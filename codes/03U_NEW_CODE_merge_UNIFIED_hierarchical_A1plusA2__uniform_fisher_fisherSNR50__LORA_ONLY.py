import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

# 1) unified_uniform_lora      = Uniform( a1_uniform_lora , a2_uniform_lora )
# 2) unified_fisher_lora       = Fisher ( a1_fisher_lora  , a2_fisher_lora  )
# 3) unified_fisher_snr50_lora = Fisher+SNR50( a1_fisher_snr50_lora , a2_fisher_snr50_lora )
#    + Re-apply SNR selection again at unified stage (top 50% layers)
#
# INPUT PATHS
# -------------------------
A1_UNIFORM = Path("models_hf/a1_merged_lora_merges/a1_uniform_lora")
A2_UNIFORM = Path("models_hf/a2_merged_lora_merges/a2_uniform_lora")

A1_FISHER  = Path("models_hf/a1_merged_lora_merges/a1_fisher_lora")
A2_FISHER  = Path("models_hf/a2_merged_lora_merges/a2_fisher_lora")

A1_FISHER_SNR50 = Path("models_hf/a1_merged_lora_merges/a1_fisher_snr50_lora")
A2_FISHER_SNR50 = Path("models_hf/a2_merged_lora_merges/a2_fisher_snr50_lora")

# -------------------------
# OUTPUT ROOT
# -------------------------
OUTPUT_ROOT = Path("models_hf/unified_hierarchical_lora_merges")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Re-apply SNR at unified stage
SNR_KEEP_RATIO = 0.5  # top 50% transformer layers


# ============================
# UTIL: load adapter safetensors
# ============================
def load_adapter_state(folder: Path):
    st_path = folder / "adapter_model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"Missing adapter_model.safetensors in: {folder}")
    return load_file(str(st_path))

def copy_adapter_config(src_folder: Path, dst_folder: Path):
    cfg_src = src_folder / "adapter_config.json"
    if not cfg_src.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in: {src_folder}")
    (dst_folder / "adapter_config.json").write_text(cfg_src.read_text(), encoding="utf-8")


# ============================
# MERGE METHODS
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

def layer_id_from_key(key: str) -> int:
    marker = ".layers."
    if marker not in key:
        return -1
    tail = key.split(marker, 1)[1]
    digits = ""
    for c in tail:
        if c.isdigit():
            digits += c
        else:
            break
    return int(digits) if digits else -1


# ============================
# SNR PER LAYER
# ============================
def compute_snr_score(W: torch.Tensor) -> float:
    """
    Simple spectral score: top singular value of (flattened) tensor.
    """
    X = W.float()
    if X.ndim != 2:
        X = X.view(X.shape[0], -1)
    try:
        s = torch.linalg.svdvals(X)
        return float(s[0])
    except Exception:
        return 0.0

def compute_layer_snr_map_from_two_states(state1, state2, keys):
    """
    For each layer, build a probe tensor by uniform-merging (state1[key], state2[key]),
    then compute spectral score. Average over all keys in the layer.
    """
    layer_scores = {}
    for k in keys:
        lid = layer_id_from_key(k)
        if lid < 0:
            continue
        probe = uniform_merge([state1[k], state2[k]])
        score = compute_snr_score(probe)
        layer_scores.setdefault(lid, []).append(score)

    for lid in layer_scores:
        layer_scores[lid] = sum(layer_scores[lid]) / len(layer_scores[lid])
    return layer_scores


# ============================
# CORE: merge two adapter folders
# ============================
def merge_two_adapters_and_save(
    input_a: Path,
    input_b: Path,
    out_name: str,
    merge_fn,
    apply_snr: bool = False,
    snr_keep_ratio: float = 0.5,
):
    print(f"\n[START] {out_name}")
    print(f"  A: {input_a}")
    print(f"  B: {input_b}")

    stA = load_adapter_state(input_a)
    stB = load_adapter_state(input_b)

    keysA = set(stA.keys())
    keysB = set(stB.keys())
    if keysA != keysB:
        missing = list(keysA - keysB)[:10]
        extra = list(keysB - keysA)[:10]
        raise ValueError(
            f"Adapter keys mismatch between:\n{input_a}\n{input_b}\n"
            f"Missing in B (first 10): {missing}\n"
            f"Extra in B (first 10): {extra}"
        )

    keys = sorted(keysA)

    # SNR selection (unified stage), if requested
    selected_layers = None
    if apply_snr:
        layer_scores = compute_layer_snr_map_from_two_states(stA, stB, keys)
        ranked = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = max(1, int(len(ranked) * snr_keep_ratio))
        selected_layers = set([lid for lid, _ in ranked[:top_k]])
        print(f"  [SNR] keep_ratio={snr_keep_ratio:.2f} => keeping {top_k}/{len(ranked)} layers")
        print(f"  [SNR] selected layers: {sorted(selected_layers)}")

    merged_state = {}
    for k in keys:
        lid = layer_id_from_key(k)

        # if SNR enabled and key belongs to a non-selected layer => zero it
        if selected_layers is not None and lid != -1 and lid not in selected_layers:
            merged_state[k] = torch.zeros_like(stA[k])
            continue

        merged_state[k] = merge_fn([stA[k], stB[k]])

    out_dir = OUTPUT_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_file(merged_state, str(out_dir / "adapter_model.safetensors"))
    # copy config from input_a (both should match anyway)
    copy_adapter_config(input_a, out_dir)

    print(f"[DONE] {out_name} saved to {out_dir}")


# ============================
# RUN: 3 unified hierarchical merges
# ============================
if __name__ == "__main__":
    # 1) UNIFIED UNIFORM (from uniform-merged A1 and A2)
    merge_two_adapters_and_save(
        input_a=A1_UNIFORM,
        input_b=A2_UNIFORM,
        out_name="UNIFIED_hierarchical__A1uniform_plus_A2uniform__UNIFORM__LORA",
        merge_fn=uniform_merge,
        apply_snr=False
    )

    # 2) UNIFIED FISHER (from fisher-merged A1 and A2)
    merge_two_adapters_and_save(
        input_a=A1_FISHER,
        input_b=A2_FISHER,
        out_name="UNIFIED_hierarchical__A1fisher_plus_A2fisher__FISHER_allLayers__LORA",
        merge_fn=fisher_merge,
        apply_snr=False
    )

    # 3) UNIFIED FISHER + SNR50
    #    Inputs are already SNR50-pruned adapters, and we re-apply SNR50 at unified stage.
    merge_two_adapters_and_save(
        input_a=A1_FISHER_SNR50,
        input_b=A2_FISHER_SNR50,
        out_name="UNIFIED_hierarchical__A1fisherSNR50_plus_A2fisherSNR50__FISHERplus_SNR50_reapplied__LORA",
        merge_fn=fisher_merge,
        apply_snr=True,
        snr_keep_ratio=SNR_KEEP_RATIO
    )

    print("\nAll UNIFIED hierarchical LoRA merges completed.")