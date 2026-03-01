import os
from pathlib import Path
from huggingface_hub import snapshot_download

HF_USERNAME = "ejaz111"

LOCAL_ROOT = Path("models_hf")
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)


MODELS = {
    
    "a1_lexical_expert":   f"{HF_USERNAME}/archaic-italian_a1-lexical-expert-lora",
    "a1_syntactic_expert": f"{HF_USERNAME}/archaic-italian_a1-syntactic-expert-lora",
    "a1_semantic_expert":  f"{HF_USERNAME}/archaic-italian_a1-semantic-expert-lora",

    "a2_early_expert":     f"{HF_USERNAME}/archaic-italian_a2-early-expert-lora",
    "a2_middle_expert":    f"{HF_USERNAME}/archaic-italian_a2-middle-expert-lora",
    "a2_late_expert":      f"{HF_USERNAME}/archaic-italian_a2-late-expert-lora",

    
    "a1_uniform_lora":      f"{HF_USERNAME}/archaic-italian_a1-uniform-lora",
    "a1_fisher_all_lora":   f"{HF_USERNAME}/archaic-italian_a1-fisher-alllayers-lora",
    "a1_fisher_snr50_lora": f"{HF_USERNAME}/archaic-italian_a1-fisher-snr50-lora",

    
    "a2_uniform_lora":      f"{HF_USERNAME}/archaic-italian_a2-uniform-lora",
    "a2_fisher_all_lora":   f"{HF_USERNAME}/archaic-italian_a2-fisher-alllayers-lora",
    "a2_fisher_snr50_lora": f"{HF_USERNAME}/archaic-italian_a2-fisher-snr50-lora",

    
    "unified_hier_uniform_lora": f"{HF_USERNAME}/archaic-italian_unified-hier-a1plus-a2-uniform-lora",
    "unified_hier_fisher_all_lora": f"{HF_USERNAME}/archaic-italian_unified-hier-a1plus-a2-fisher-alllayers-lora",
    "unified_hier_fisher_snr50_reapplied_lora": f"{HF_USERNAME}/archaic-italian_unified-hier-a1plus-a2-fisher-snr50-reapplied-lora",
}


def download_model(logical_name: str, repo_id: str):
    target_dir = LOCAL_ROOT / logical_name
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[SKIP] Exists: {logical_name} -> {target_dir}")
        return str(target_dir)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Downloaded: {logical_name} -> {target_dir}")
        return str(target_dir)
    except Exception as e:
        print(f"[ERR] Failed: {logical_name} ({repo_id}) -> {e}")
        return None


def main():
    results = {}

    for logical_name, repo_id in MODELS.items():
        path = download_model(logical_name, repo_id)
        results[logical_name] = path is not None


if __name__ == "__main__":
    main()