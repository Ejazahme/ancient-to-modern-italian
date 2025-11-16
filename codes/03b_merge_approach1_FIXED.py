import json
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import save_file

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EXPERTS_DIR = Path("models/approach1")
MERGED_LORA_DIR = Path("models/approach1/merged_lora")
MERGED_FULL_DIR = Path("models/approach1/merged_full")

SNR_TOP_FRACTION = 0.5

EXPERT_NAMES = [
    "lexical_expert",
    "syntactic_expert",
    "semantic_expert",
]

def find_adapter_file(dir_path: Path) -> Path:
    candidates = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
    ]
    for name in candidates:
        p = dir_path / name
        if p.exists():
            return p
    ckpts = sorted(dir_path.glob("checkpoint-*/pytorch_model.bin"))
    if ckpts:
        return ckpts[-1]

    raise FileNotFoundError(f"No adapter model file found in {dir_path}")


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
    lambda_plus = sigma * (1 + np.sqrt(beta)) ** 2
    return lambda_plus.item()


def compute_layer_snr(W: torch.Tensor) -> float:
    
    if W is None or W.numel() == 0:
        return 0.0

    try:
        W2d = W
        if W2d.dim() > 2:
            W2d = W2d.reshape(-1, W2d.shape[-1])

        W2d = W2d.float()
        U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)

        m, n = W2d.shape
        beta = min(m, n) / max(m, n)
        threshold = compute_mp_threshold(S, beta)

        signal = S[S > threshold]
        noise = S[S <= threshold]

        if noise.numel() == 0:
            return 100.0

        signal_power = (signal ** 2).sum()
        noise_power = (noise ** 2).sum()
        if noise_power <= 0:
            return 100.0

        snr = 10.0 * torch.log10(signal_power / noise_power)
        return float(snr.item())
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
        stacked = torch.stack([w.float() for w in weights_list], dim=0)
        return stacked.mean(dim=0).to(weights_list[0].dtype)

    if method == "fisher":
        W_float = [w.float() for w in weights_list]
        fisher = [w ** 2 for w in W_float]
        F_sum = torch.zeros_like(W_float[0])
        for F in fisher:
            F_sum.add_(F)
        F_sum.clamp_(min=1e-8)
        num = torch.zeros_like(W_float[0])
        for F, W in zip(fisher, W_float):
            num.add_(F * W)
        merged = num / F_sum
        return merged.to(weights_list[0].dtype)

    raise ValueError(f"Unknown RegMean method: {method}")


def merge_experts_snr_regmean():
    expert_adapters = {}
    for name in EXPERT_NAMES:
        exp_dir = EXPERTS_DIR / name
        adapter_path = find_adapter_file(exp_dir)
        expert_adapters[name] = load_adapter_state(adapter_path)

    expert_snrs = {}
    all_lora_keys = set()
    for name, state in expert_adapters.items():
        snrs = compute_expert_snrs(state)
        expert_snrs[name] = snrs
        all_lora_keys.update(snrs.keys())

    if not all_lora_keys:
        raise RuntimeError("No LoRA parameters found for SNR computation.")

    avg_snrs = {}
    for key in all_lora_keys:
        vals = [expert_snrs[e].get(key, 0.0) for e in EXPERT_NAMES]
        avg_snrs[key] = float(np.mean(vals))

    sorted_by_snr = sorted(avg_snrs.items(), key=lambda x: x[1], reverse=True)
    n_select = max(1, int(len(sorted_by_snr) * SNR_TOP_FRACTION))
    selected_keys = {k for k, _ in sorted_by_snr[:n_select]}

    merged_adapter = {}
    merged_count = 0
    skipped_missing = 0

    all_lora_param_keys = set()
    for st in expert_adapters.values():
        for k in st.keys():
            if "lora" in k.lower():
                all_lora_param_keys.add(k)

    for key in all_lora_param_keys:
        if key in selected_keys:
            weights_list = []
            for exp in EXPERT_NAMES:
                W = expert_adapters[exp].get(key, None)
                if W is None:
                    weights_list = []
                    break
                weights_list.append(W)
            if weights_list:
                merged_W = regmean_dataless_merge(weights_list, method="fisher")
                merged_adapter[key] = merged_W
                merged_count += 1
            else:
                skipped_missing += 1
        else:
            for exp in EXPERT_NAMES:
                W = expert_adapters[exp].get(key, None)
                if W is not None:
                    merged_adapter[key] = torch.zeros_like(W)
                    break

    ref_exp_dir = EXPERTS_DIR / EXPERT_NAMES[0]
    adapter_config_path = ref_exp_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError("adapter_config.json not found in expert directory.")
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    MERGED_LORA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MERGED_LORA_DIR / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)

    save_adapter_state_safetensors(merged_adapter, MERGED_LORA_DIR / "adapter_model.safetensors")

    merge_info = {
        "base_model": BASE_MODEL,
        "experts": EXPERT_NAMES,
        "snr_top_fraction": SNR_TOP_FRACTION,
        "total_lora_params": len(all_lora_param_keys),
        "selected_params": n_select,
        "regmean_method": "fisher (dataless, F ~ W^2)",
        "merged_params": merged_count,
        "skipped_missing": skipped_missing,
    }
    with open(MERGED_LORA_DIR / "merge_info.json", "w", encoding="utf-8") as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    peft_config = PeftConfig.from_pretrained(MERGED_LORA_DIR)
    peft_model = PeftModel.from_pretrained(base_model, MERGED_LORA_DIR, config=peft_config)

    merged_full = peft_model.merge_and_unload()
    MERGED_FULL_DIR.mkdir(parents=True, exist_ok=True)
    merged_full.save_pretrained(MERGED_FULL_DIR)
    tokenizer.save_pretrained(MERGED_FULL_DIR)

if __name__ == "__main__":
    merge_experts_snr_regmean()
