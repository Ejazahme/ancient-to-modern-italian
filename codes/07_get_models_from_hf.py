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

    "a1_merged_lora":      f"{HF_USERNAME}/archaic-italian_a1-merged-lora",
    "a1_merged_full":      f"{HF_USERNAME}/archaic-italian_a1-merged-full",

    "a2_early_expert":     f"{HF_USERNAME}/archaic-italian_a2-early-expert-lora",
    "a2_middle_expert":    f"{HF_USERNAME}/archaic-italian_a2-middle-expert-lora",
    "a2_late_expert":      f"{HF_USERNAME}/archaic-italian_a2-late-expert-lora",

    "a2_merged_lora":      f"{HF_USERNAME}/archaic-italian_a2-merged-lora",
    "a2_merged_full":      f"{HF_USERNAME}/archaic-italian_a2-merged-full",

    "unified_merged_full": f"{HF_USERNAME}/archaic-italian_unified-merged-full",
}


def download_model(logical_name: str, repo_id: str):
    target_dir = LOCAL_ROOT / logical_name
    if target_dir.exists() and any(target_dir.iterdir()):
        return str(target_dir)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        return str(target_dir)
    except Exception as e:
        return None


def main():
    results = {}

    for logical_name, repo_id in MODELS.items():
        path = download_model(logical_name, repo_id)
        results[logical_name] = path is not None
    


if __name__ == "__main__":
    main()
