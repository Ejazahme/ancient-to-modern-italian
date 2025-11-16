import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EXPERTS_DIR = Path("models/approach2")
MERGED_LORA_DIR = Path("models/approach2/merged_lora")
MERGED_FULL_DIR = Path("models/approach2/merged_full")

SNR_TOP_FRACTION = 0.5
EXPERT_NAMES = ["early_expert", "middle_expert", "late_expert"]

def find_adapter_file(dir_path: Path) -> Path:
    for name in ["adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin"]:
        f = dir_path / name
        if f.exists():
            return f
    ckpts = sorted(dir_path.glob("checkpoint-*/pytorch_model.bin"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No adapter file found in {dir_path}")

def load_adapter_state(path: Path) -> dict:
    if path.suffix == ".safetensors":
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors
    else:
        return torch.load(path, map_location="cpu")

def save_adapter_state_safetensors(state_dict: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    save_file(cpu_state, str(path))

def compute_mp_threshold(S: torch.Tensor, beta: float = 1.0) -> float:
    median_sv = torch.median(S)
    mad = torch.median(torch.abs(S - median_sv))
    sigma = 1.4826 * mad
    return (sigma * (1 + np.sqrt(beta)) ** 2).item()

def compute_layer_snr(W: torch.Tensor) -> float:
    if W is None or W.numel() == 0:
        return 0.0
    try:
        W2 = W.reshape(-1, W.shape[-1]) if W.dim() > 2 else W
        W2 = W2.float()
        _, S, _ = torch.linalg.svd(W2, full_matrices=False)
        m, n = W2.shape
        beta = min(m, n) / max(m, n)
        thr = compute_mp_threshold(S, beta)
        sig, noi = S[S > thr], S[S <= thr]
        if noi.numel() == 0:
            return 100.0
        snr = 10 * torch.log10((sig ** 2).sum() / (noi ** 2).sum())
        return float(snr)
    except Exception as e:
        return 0.0

def compute_expert_snrs(adapter_state: dict) -> dict:
    snrs = {}
    for name, W in adapter_state.items():
        if "lora" in name.lower() and W.dim() >= 2:
            snrs[name] = compute_layer_snr(W)
    return snrs

def regmean_dataless_merge(weights_list, method: str = "fisher") -> torch.Tensor:
    if not weights_list:
        return None
    if method == "uniform":
        return torch.mean(torch.stack([w.float() for w in weights_list]), dim=0).to(weights_list[0].dtype)
    if method == "fisher":
        Wf = [w.float() for w in weights_list]
        F = [w ** 2 for w in Wf]
        Fsum = torch.zeros_like(Wf[0])
        num = torch.zeros_like(Wf[0])
        for Fi, Wi in zip(F, Wf):
            Fsum += Fi
            num += Fi * Wi
        Fsum.clamp_(min=1e-8)
        return (num / Fsum).to(weights_list[0].dtype)
    raise ValueError("method must be 'fisher' or 'uniform'")

def merge_experts_snr_regmean():
    expert_adapters = {}
    for name in EXPERT_NAMES:
        path = find_adapter_file(EXPERTS_DIR / name)
        expert_adapters[name] = load_adapter_state(path)

    expert_snrs = {}
    all_keys = set()
    for name, st in expert_adapters.items():
        snrs = compute_expert_snrs(st)
        expert_snrs[name] = snrs
        all_keys.update(snrs.keys())

    avg_snrs = {k: np.mean([expert_snrs[e].get(k, 0) for e in EXPERT_NAMES]) for k in all_keys}
    sorted_snrs = sorted(avg_snrs.items(), key=lambda x: x[1], reverse=True)
    n_select = max(1, int(len(sorted_snrs) * SNR_TOP_FRACTION))
    selected = {k for k, _ in sorted_snrs[:n_select]}

    merged_adapter, merged_count, skipped = {}, 0, 0
    all_lora_keys = {k for s in expert_adapters.values() for k in s.keys() if "lora" in k.lower()}
    for k in all_lora_keys:
        if k in selected:
            Ws = [expert_adapters[e].get(k, None) for e in EXPERT_NAMES]
            if all(w is not None for w in Ws):
                merged_adapter[k] = regmean_dataless_merge(Ws, "fisher")
                merged_count += 1
            else:
                skipped += 1
        else:
            for e in EXPERT_NAMES:
                W = expert_adapters[e].get(k)
                if W is not None:
                    merged_adapter[k] = torch.zeros_like(W)
                    break

    ref_cfg = EXPERTS_DIR / EXPERT_NAMES[0] / "adapter_config.json"
    with open(ref_cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    MERGED_LORA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MERGED_LORA_DIR / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    save_adapter_state_safetensors(merged_adapter, MERGED_LORA_DIR / "adapter_model.safetensors")

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    peft_cfg = PeftConfig.from_pretrained(MERGED_LORA_DIR)
    peft_model = PeftModel.from_pretrained(base, MERGED_LORA_DIR, config=peft_cfg)
    merged_full = peft_model.merge_and_unload()
    MERGED_FULL_DIR.mkdir(parents=True, exist_ok=True)
    merged_full.save_pretrained(MERGED_FULL_DIR)
    tok.save_pretrained(MERGED_FULL_DIR)

    merge_info = {
        "base_model": BASE_MODEL,
        "experts": EXPERT_NAMES,
        "snr_top_fraction": SNR_TOP_FRACTION,
        "merged_params": merged_count,
        "method": "RegMean (Fisher dataless)",
        "selected_layers": n_select,
        "total_layers": len(all_lora_keys),
    }
    with open(MERGED_LORA_DIR / "merge_info.json", "w", encoding="utf-8") as f:
        json.dump(merge_info, f, indent=2)

if __name__ == "__main__":
    merge_experts_snr_regmean()
