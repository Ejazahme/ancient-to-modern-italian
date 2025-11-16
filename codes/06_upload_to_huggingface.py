import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "ejaz111"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

MODELS_TO_UPLOAD = [
    ("models/approach1/lexical_expert",   "archaic-italian_a1-lexical-expert-lora"),
    ("models/approach1/syntactic_expert", "archaic-italian_a1-syntactic-expert-lora"),
    ("models/approach1/semantic_expert",  "archaic-italian_a1-semantic-expert-lora"),
    ("models/approach1/merged_lora",      "archaic-italian_a1-merged-lora"),
    ("models/approach1/merged_full",      "archaic-italian_a1-merged-full"),
    ("models/approach2/early_expert",     "archaic-italian_a2-early-expert-lora"),
    ("models/approach2/middle_expert",    "archaic-italian_a2-middle-expert-lora"),
    ("models/approach2/late_expert",      "archaic-italian_a2-late-expert-lora"),
    ("models/approach2/merged_lora",      "archaic-italian_a2-merged-lora"),
    ("models/approach2/merged_full",      "archaic-italian_a2-merged-full"),
    ("models/unified/merged_full",        "archaic-italian_unified-merged-full"),
]

def create_minimal_card(local_path: Path, repo_full_name: str):
    content = f"# {repo_full_name}\n\nModel uploaded automatically.\n\n_Base model_: `{BASE_MODEL}`\n"
    readme_path = local_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    return readme_path


def upload_model(local_path, repo_name):
    path = Path(local_path)
    if not path.exists():
        return False

    repo_full = f"{HF_USERNAME}/{repo_name}"

    api = HfApi()
    try:
        create_repo(repo_id=repo_full, private=False, exist_ok=True)
    except Exception as e:
        pass

    readme_path = create_minimal_card(path, repo_full)

    try:
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_full,
            repo_type="model",
            commit_message="Initial upload"
        )
        return True
    except Exception as e:
        pass
        return False


def main():
    api = HfApi()
    results = {}
    for local_path, repo_name in MODELS_TO_UPLOAD:
        ok = upload_model(local_path, repo_name)
        results[repo_name] = ok
    


if __name__ == "__main__":
    main()
