import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "ejaz111"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ============================================================
# UPLOAD ONLY MERGED LoRA ADAPTERS
# - We upload ONLY the new merged LoRA adapters:
#   A1: uniform / fisher / fisher+snr50
#   A2: uniform / fisher / fisher+snr50
#   UNIFIED (hierarchical): uniform / fisher / fisher+snr50(reapplied)
# ============================================================

MODELS_TO_UPLOAD = [
    # -------------------------
    # A1 merged LoRA adapters
    # -------------------------
    ("models_hf/a1_merged_lora_merges/a1_uniform_lora",       "archaic-italian_a1-uniform-lora"),
    ("models_hf/a1_merged_lora_merges/a1_fisher_lora",        "archaic-italian_a1-fisher-alllayers-lora"),
    ("models_hf/a1_merged_lora_merges/a1_fisher_snr50_lora",  "archaic-italian_a1-fisher-snr50-lora"),

    # -------------------------
    # A2 merged LoRA adapters
    # -------------------------
    ("models_hf/a2_merged_lora_merges/a2_uniform_lora",       "archaic-italian_a2-uniform-lora"),
    ("models_hf/a2_merged_lora_merges/a2_fisher_lora",        "archaic-italian_a2-fisher-alllayers-lora"),
    ("models_hf/a2_merged_lora_merges/a2_fisher_snr50_lora",  "archaic-italian_a2-fisher-snr50-lora"),

    # -------------------------
    # UNIFIED hierarchical merged LoRA adapters
    # -------------------------
    ("models_hf/unified_hierarchical_lora_merges/UNIFIED_hierarchical__A1uniform_plus_A2uniform__UNIFORM__LORA",
     "archaic-italian_unified-hier-a1plus-a2-uniform-lora"),

    ("models_hf/unified_hierarchical_lora_merges/UNIFIED_hierarchical__A1fisher_plus_A2fisher__FISHER_allLayers__LORA",
     "archaic-italian_unified-hier-a1plus-a2-fisher-alllayers-lora"),

    ("models_hf/unified_hierarchical_lora_merges/UNIFIED_hierarchical__A1fisherSNR50_plus_A2fisherSNR50__FISHERplus_SNR50_reapplied__LORA",
     "archaic-italian_unified-hier-a1plus-a2-fisher-snr50-reapplied-lora"),
]


def create_minimal_card(local_path: Path, repo_full_name: str):
    content = (
        f"# {repo_full_name}\n\n"
        f"LoRA adapter uploaded automatically.\n\n"
        f"- Base model: `{BASE_MODEL}`\n"
    )
    readme_path = local_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    return readme_path


def upload_model(local_path, repo_name):
    path = Path(local_path)
    if not path.exists():
        print(f"[SKIP] Missing local path: {path}")
        return False

    repo_full = f"{HF_USERNAME}/{repo_name}"
    api = HfApi()

    try:
        create_repo(repo_id=repo_full, private=False, exist_ok=True)
    except Exception:
        pass

    _ = create_minimal_card(path, repo_full)

    try:
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_full,
            repo_type="model",
            commit_message="Upload merged LoRA adapter"
        )
        print(f"[OK] Uploaded -> {repo_full}")
        return True
    except Exception as e:
        print(f"[ERR] Failed upload {repo_full}: {e}")
        return False


def main():
    results = {}
    for local_path, repo_name in MODELS_TO_UPLOAD:
        ok = upload_model(local_path, repo_name)
        results[repo_name] = ok

    # Save upload report
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    with open(out / "hf_upload_report_merged_lora.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()