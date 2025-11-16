from pathlib import Path

PROJECT_STRUCTURE = {
    "data": {
        "raw": [],
        "synthetic": [],
        "processed": {
            "approach1": {
                "lexical": [],
                "syntactic": [],
                "semantic": []
            },
            "approach2": {
                "early_1260": [],
                "middle_1310": [],
                "late_1360": []
            }
        }
    },
    "models": {
        "approach1": {
            "lexical_expert": [],
            "syntactic_expert": [],
            "semantic_expert": [],
            "merged_lora": [],
            "merged_full": []
        },
        "approach2": {
            "early_expert": [],
            "middle_expert": [],
            "late_expert": [],
            "merged_lora": [],
            "merged_full": []
        },
        "unified": {
            "merged_full": []
        }
    },
    "models_hf": [],
    "outputs": {
        "gpt4_1_translations": [],
        "gpt4_1_translations_fewshot": [],
        "gemini2_5_translations": [],
        "gemini2_5_translations_fewshots": [],
        "stratified_evaluation": [],
        "generative_evaluation": []
    },
    "results": {
        "llm-as-a-judge": {
            "finetuned": []
        }
    },
    "logs": {
        "training": []
    }
}

def create_directory_structure(base_path=".", structure=None):
    if structure is None:
        structure = PROJECT_STRUCTURE
    
    for name, content in structure.items():
        path = Path(base_path) / name
        path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, dict):
            create_directory_structure(path, content)

def main():
    create_directory_structure()

if __name__ == "__main__":
    main()
