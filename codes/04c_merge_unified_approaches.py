import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

APPROACH1_PATH = Path("models/approach1/merged_full")
APPROACH2_PATH = Path("models/approach2/merged_full")

OUTPUT_DIR = Path("models/unified/merged_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8


def load_model_state(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    return sd


def fisher_weighted_merge(state_dicts):
    if len(state_dicts) < 2:
        raise ValueError("need at least two models to merge.")

    ref_keys = list(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if sd.keys() != state_dicts[0].keys():
            raise ValueError("state dict keys differ between models; check inputs.")

    merged = {}
    num_models = len(state_dicts)

    for k in ref_keys:
        tensors = [sd[k] for sd in state_dicts]

        
        if tensors[0].is_floating_point():
            
            W = [t.to(torch.float32) for t in tensors]
            F = [t ** 2 for t in W]

            
            F_sum = torch.zeros_like(W[0])
            num = torch.zeros_like(W[0])
            for Fi, Wi in zip(F, W):
                F_sum.add_(Fi)
                num.add_(Fi * Wi)

            F_sum.clamp_(min=EPS)
            merged_t = num / F_sum

            
            merged[k] = merged_t.to(torch.float16)
        else:
            
            merged[k] = tensors[0]

    return merged


def main():
    sd1 = load_model_state(APPROACH1_PATH)
    sd2 = load_model_state(APPROACH2_PATH)

    merged_sd = fisher_weighted_merge([sd1, sd2])

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    missing, unexpected = base_model.load_state_dict(merged_sd, strict=False)

    if missing:
        pass
    if unexpected:
        pass

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    
    merge_info = {
        "base_model": BASE_MODEL,
        "sources": [
            str(APPROACH1_PATH),
            str(APPROACH2_PATH),
        ],
        "method": "Dataless RegMean-style Fisher merge over full merged models",
        "eps": EPS,
        "note": (
            "Inputs are already SNR+RegMean merged_full models from Approach 1 "
            "and Approach 2; this script fuses them into a single unified model."
        ),
    }
    with open(OUTPUT_DIR / "merge_info.json", "w", encoding="utf-8") as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)

    


if __name__ == "__main__":
    main()
