import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

# ============================
# CONFIG
# ============================

A2_EXPERTS = [
    Path("models_hf/a2_early_expert"),
    Path("models_hf/a2_middle_expert"),
    Path("models_hf/a2_late_expert"),
]

OUTPUT_ROOT = Path("models_hf/a2_merged_lora_merges")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SNR_KEEP_RATIO = 0.5  # top 50%


# ============================
# LOAD ADAPTER STATES
# ============================

def load_adapter(folder: Path):
    return load_file(str(folder / "adapter_model.safetensors"))

expert_states = [load_adapter(p) for p in A2_EXPERTS]
keys = list(expert_states[0].keys())


# ============================
# MERGE FUNCTIONS
# ============================

def uniform_merge(tensors):
    return torch.stack(tensors, dim=0).mean(dim=0)

def fisher_merge(tensors, eps=1e-12):
    W = [t.float() for t in tensors]
    F = [w.pow(2) for w in W]
    num = torch.zeros_like(W[0])
    den = torch.zeros_like(W[0])
    for w, f in zip(W, F):
        num += f * w
        den += f
    return (num / (den + eps)).to(tensors[0].dtype)

def layer_id_from_key(key):
    marker = ".layers."
    if marker not in key:
        return -1
    tail = key.split(marker)[1]
    digits = ""
    for c in tail:
        if c.isdigit():
            digits += c
        else:
            break
    return int(digits) if digits else -1


# ============================
# SNR COMPUTATION
# ============================

def compute_snr_matrix(W):
    X = W.float()
    if X.ndim != 2:
        X = X.view(X.shape[0], -1)
    try:
        s = torch.linalg.svdvals(X)
        return float(s[0])
    except:
        return 0.0

def compute_layer_snr_map():
    layer_scores = {}
    for k in keys:
        lid = layer_id_from_key(k)
        if lid < 0:
            continue
        merged_probe = uniform_merge([st[k] for st in expert_states])
        score = compute_snr_matrix(merged_probe)
        layer_scores.setdefault(lid, []).append(score)

    for lid in layer_scores:
        layer_scores[lid] = sum(layer_scores[lid]) / len(layer_scores[lid])
    return layer_scores


# ============================
# MERGE LOGIC
# ============================

def merge_and_save(method_name, merge_fn, snr_layers=None):

    merged_state = {}

    for k in keys:
        lid = layer_id_from_key(k)

        if snr_layers is not None and lid not in snr_layers and lid != -1:
            merged_state[k] = torch.zeros_like(expert_states[0][k])
            continue

        tensors = [st[k] for st in expert_states]
        merged_state[k] = merge_fn(tensors)

    out_dir = OUTPUT_ROOT / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_file(merged_state, str(out_dir / "adapter_model.safetensors"))

    config_src = A2_EXPERTS[0] / "adapter_config.json"
    config_dst = out_dir / "adapter_config.json"
    config_dst.write_text(config_src.read_text())

    print(f"[DONE] {method_name} saved to {out_dir}")


# ============================
# RUN ALL THREE
# ============================

print("Running A2 Uniform Merge...")
merge_and_save("a2_uniform_lora", uniform_merge)

print("Running A2 Fisher Merge...")
merge_and_save("a2_fisher_lora", fisher_merge)

print("Computing SNR layers (top 50%)...")
layer_scores = compute_layer_snr_map()
sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
top_k = int(len(sorted_layers) * SNR_KEEP_RATIO)
snr_layers = set([lid for lid, _ in sorted_layers[:top_k]])

print("Running A2 Fisher + SNR (top 50%)...")
merge_and_save("a2_fisher_snr50_lora", fisher_merge, snr_layers)

print("All A2 merges completed.")