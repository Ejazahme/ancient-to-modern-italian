import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from evaluate import load
from bert_score import score as bert_score
import sacrebleu


# =========================
# CONFIG
# =========================
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
MODELS_ROOT = Path("models_hf")

# Paper-friendly output folder for "experts only"
OUTPUT_ROOT = Path("outputs/stratified_evaluation_experts_only")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
MAX_NEW_TOKENS = 128

GENERATION_CONFIG = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    "a1_lexical_test":   Path("data/processed/approach1/lexical/test.jsonl"),
    "a1_syntactic_test": Path("data/processed/approach1/syntactic/test.jsonl"),
    "a1_semantic_test":  Path("data/processed/approach1/semantic/test.jsonl"),
    "a2_early_1260_test":  Path("data/processed/approach2/early_1260/test.jsonl"),
    "a2_middle_1310_test": Path("data/processed/approach2/middle_1310/test.jsonl"),
    "a2_late_1360_test":   Path("data/processed/approach2/late_1360/test.jsonl"),
    "real_97_test": Path("data/processed/test_ground_truth.jsonl"),
}

# Evaluate each expert on its "own" split + real_97
EVAL_MAP = {
    "A1__LexicalExpert__LoRA":   ["a1_lexical_test", "real_97_test"],
    "A1__SyntacticExpert__LoRA": ["a1_syntactic_test", "real_97_test"],
    "A1__SemanticExpert__LoRA":  ["a1_semantic_test", "real_97_test"],
    "A2__Early1260Expert__LoRA":   ["a2_early_1260_test", "real_97_test"],
    "A2__Middle1310Expert__LoRA":  ["a2_middle_1310_test", "real_97_test"],
    "A2__Late1360Expert__LoRA":    ["a2_late_1360_test", "real_97_test"],
}


# =========================
# PROMPT
# =========================
def format_prompt(text: str):
    return f"""### Istruzione:
Modernizza la seguente frase italiana arcaica in italiano contemporaneo.

### Frase arcaica:
{text}

### Frase moderna:
"""


# =========================
# DATA LOADING
# =========================
def load_dataset(path: Path):
    if not path.exists():
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            src = item.get("source") or item.get("archaic") or item.get("input")
            tgt = item.get("target") or item.get("modern") or item.get("reference") or item.get("output")
            if not src:
                continue
            data.append({"id": item.get("id"), "source": src, "reference": tgt})
    return data


# =========================
# MODEL LOADING (BASE + ADAPTER ONLY)
# =========================
def load_model_adapter(adapter_path: str | Path):
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # sanity check
    _ = PeftConfig.from_pretrained(str(adapter_path))
    model = PeftModel.from_pretrained(base, str(adapter_path), torch_dtype=torch.float16)

    model.eval()
    tokenizer.padding_side = "right"
    return model, tokenizer


# =========================
# GENERATION
# =========================
def batched_generate(model, tokenizer, dataset, desc):
    preds = []
    prompts = [format_prompt(d["source"]) for d in dataset]

    encodings = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    num_samples = len(dataset)

    for i in tqdm(range(0, num_samples, BATCH_SIZE), desc=desc, leave=False):
        batch_input = {k: v[i:i+BATCH_SIZE].to(model.device) for k, v in encodings.items()}

        with torch.no_grad():
            if model.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model.generate(**batch_input, **GENERATION_CONFIG, pad_token_id=tokenizer.eos_token_id)
            else:
                outputs = model.generate(**batch_input, **GENERATION_CONFIG, pad_token_id=tokenizer.eos_token_id)

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for j, gen in enumerate(texts):
            pred = gen.split("### Frase moderna:")[-1].strip().split("\n")[0]
            preds.append({
                "id": dataset[i+j].get("id"),
                "source": dataset[i+j]["source"],
                "prediction": pred,
                "reference": dataset[i+j]["reference"],
            })
    return preds


# =========================
# METRICS
# =========================
def compute_metrics(preds):
    preds_text = [p["prediction"] for p in preds]
    refs_text = [p["reference"] for p in preds if p["reference"]]

    m = {}
    if not refs_text:
        return {"has_reference": False}

    try:
        m["bleu"] = sacrebleu.corpus_bleu(preds_text, [refs_text]).score
        m["chrf++"] = sacrebleu.corpus_chrf(preds_text, [refs_text]).score
    except Exception as e:
        m["bleu_error"] = str(e)

    try:
        rouge_metric = load("rouge")
        rouge = rouge_metric.compute(predictions=preds_text, references=refs_text)
        m["rougeL"] = rouge["rougeL"] * 100
    except Exception as e:
        m["rouge_error"] = str(e)

    try:
        meteor_metric = load("meteor")
        meteor = meteor_metric.compute(predictions=preds_text, references=refs_text)
        m["meteor"] = meteor["meteor"]
    except Exception as e:
        m["meteor_error"] = str(e)

    token_f1s = []
    for p, r in zip(preds_text, refs_text):
        ps, rs = set(p.lower().split()), set(r.lower().split())
        inter = ps & rs
        prec = len(inter)/len(ps) if ps else 0
        rec = len(inter)/len(rs) if rs else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        token_f1s.append(f1)
    m["token_f1"] = float(np.mean(token_f1s))

    try:
        P, R, F1 = bert_score(
            preds_text,
            refs_text,
            lang="it",
            model_type="bert-base-multilingual-cased",
            device=DEVICE,
        )
        m["bertscore_f1"] = float(F1.mean().item())
    except Exception as e:
        m["bertscore_error"] = str(e)

    m["has_reference"] = True
    m["num_predictions"] = len(preds_text)
    return m


# =========================
# EVALUATION
# =========================
def evaluate(model_name, adapter_path, allowed_datasets):
    model, tokenizer = load_model_adapter(adapter_path)
    metrics_summary = []

    for ds in allowed_datasets:
        ds_path = DATASETS[ds]
        dataset = load_dataset(ds_path)
        if not dataset:
            continue

        preds = batched_generate(model, tokenizer, dataset, f"{model_name}->{ds}")
        metrics = compute_metrics(preds)

        out_dir = OUTPUT_ROOT / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}__on__{ds}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": model_name,
                "model_type": "adapter",
                "adapter_path": str(adapter_path),
                "base_model": BASE_MODEL,
                "dataset_name": ds,
                "dataset_path": str(ds_path),
                "num_samples": len(dataset),
                "metrics": metrics,
                "predictions": preds
            }, f, indent=2, ensure_ascii=False)

        metrics_summary.append({"model": model_name, "dataset": ds, "metrics": metrics})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics_summary


# =========================
# MODELS: FINETUNED EXPERTS ONLY (NO MERGES)
# =========================
MODELS = [
    ("A1__LexicalExpert__LoRA",   MODELS_ROOT / "a1_lexical_expert"),
    ("A1__SyntacticExpert__LoRA", MODELS_ROOT / "a1_syntactic_expert"),
    ("A1__SemanticExpert__LoRA",  MODELS_ROOT / "a1_semantic_expert"),
    ("A2__Early1260Expert__LoRA",  MODELS_ROOT / "a2_early_expert"),
    ("A2__Middle1310Expert__LoRA", MODELS_ROOT / "a2_middle_expert"),
    ("A2__Late1360Expert__LoRA",   MODELS_ROOT / "a2_late_expert"),
]


def main():
    all_results = []
    for name, adapter_path in MODELS:
        if not adapter_path.exists():
            print(f"[SKIP] Missing adapter folder: {adapter_path}")
            continue

        allowed = EVAL_MAP.get(name, [])
        if not allowed:
            print(f"[SKIP] No EVAL_MAP entry for: {name}")
            continue

        results = evaluate(name, adapter_path, allowed)
        all_results.extend(results)

    # Paper-friendly summary print
    for r in all_results:
        m = r["metrics"]
        if not m.get("has_reference"):
            continue
        print(
            f"{r['model']:30s} | {r['dataset']:22s} | "
            f"BLEU={m.get('bleu',0):.2f} | chrF={m.get('chrf++',0):.2f} | "
            f"ROUGE-L={m.get('rougeL',0):.2f} | METEOR={m.get('meteor',0):.4f} | "
            f"BERT={m.get('bertscore_f1',0)*100:.2f} | TokF1={m.get('token_f1',0)*100:.2f}"
        )


if __name__ == "__main__":
    main()