import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "ejaz111"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ============================================================
# UPLOAD ONLY FINETUNED LoRA EXPERTS
# (NO merged models — those are handled separately)
# ============================================================

MODELS_TO_UPLOAD = [
    # -------------------------
    # Approach 1 Experts (LoRA)
    # -------------------------
    ("models_hf/a1_lexical_expert",   "archaic-italian_a1-lexical-expert-lora"),
    ("models_hf/a1_syntactic_expert", "archaic-italian_a1-syntactic-expert-lora"),
    ("models_hf/a1_semantic_expert",  "archaic-italian_a1-semantic-expert-lora"),

    # -------------------------
    # Approach 2 Experts (LoRA)
    # -------------------------
    ("models_hf/a2_early_expert",   "archaic-italian_a2-early-expert-lora"),
    ("models_hf/a2_middle_expert",  "archaic-italian_a2-middle-expert-lora"),
    ("models_hf/a2_late_expert",    "archaic-italian_a2-late-expert-lora"),
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
        print(f"[SKIP] Path not found: {path}")
        return False

    repo_full = f"{HF_USERNAME}/{repo_name}"
    api = HfApi()

    try:
        create_repo(repo_id=repo_full, private=False, exist_ok=True)
    except Exception:
        pass

    create_minimal_card(path, repo_full)

    try:
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_full,
            repo_type="model",
            commit_message="Upload finetuned LoRA expert"
        )
        print(f"[OK] Uploaded -> {repo_full}")
        return True
    except Exception as e:
        print(f"[ERROR] Upload failed for {repo_full}: {e}")
        return False


def main():
    results = {}

    for local_path, repo_name in MODELS_TO_UPLOAD:
        ok = upload_model(local_path, repo_name)
        results[repo_name] = ok

    # Save upload summary
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/hf_upload_report_finetuned_lora_experts.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()