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
# GLOBAL CONFIG
# =========================
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

OUTPUT_ROOT = Path("outputs/stratified_evaluation_paper")
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

# =========================
# DATASETS
# =========================
DATASETS = {
    "a1_lexical_test":   Path("data/processed/approach1/lexical/test.jsonl"),
    "a1_syntactic_test": Path("data/processed/approach1/syntactic/test.jsonl"),
    "a1_semantic_test":  Path("data/processed/approach1/semantic/test.jsonl"),
    "a2_early_1260_test":  Path("data/processed/approach2/early_1260/test.jsonl"),
    "a2_middle_1310_test": Path("data/processed/approach2/middle_1310/test.jsonl"),
    "a2_late_1360_test":   Path("data/processed/approach2/late_1360/test.jsonl"),
    "real_97_test": Path("data/processed/test_ground_truth.jsonl"),
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
# IO
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
# MODEL LOADING
# =========================
def load_model(model_name: str, mtype: str, model_path: str | Path):
    """
    mtype:
      - "base":  base Mistral from HF id
      - "adapter": LoRA adapter folder (PEFT) applied on base model
      - "full": full model folder (not used here, but kept compatible)
    """
    if mtype == "base":
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif mtype == "adapter":
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _ = PeftConfig.from_pretrained(str(model_path))  # sanity check
        model = PeftModel.from_pretrained(base, str(model_path), torch_dtype=torch.float16)

    else:  # "full"
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
            preds_text, refs_text,
            lang="it",
            model_type="bert-base-multilingual-cased",
            device=DEVICE
        )
        m["bertscore_f1"] = float(F1.mean().item())
    except Exception as e:
        m["bertscore_error"] = str(e)

    m["has_reference"] = True
    m["num_predictions"] = len(preds_text)
    return m

# =========================
# EVAL DRIVER
# =========================
def evaluate(model_name, mtype, model_path, allowed_datasets):
    model, tokenizer = load_model(model_name, mtype, model_path)
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

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{model_name}__on__{ds}__{ts}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": model_name,
                "model_type": mtype,
                "model_path": str(model_path),
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
# MODELS (CLEAR PAPER NAMES)
# =========================
MODELS = [
    # Base model (full baseline, no LoRA)
    ("BASE__Mistral7BInstruct_v0_2", "base", BASE_MODEL_ID),

    # A1 LoRA merges
    ("A1__Uniform__LoRA", "adapter", Path("models_hf/a1_merged_lora_merges/a1_uniform_lora")),
    ("A1__FisherAllLayers__LoRA", "adapter", Path("models_hf/a1_merged_lora_merges/a1_fisher_lora")),
    ("A1__FisherPlusSNR50__LoRA", "adapter", Path("models_hf/a1_merged_lora_merges/a1_fisher_snr50_lora")),

    # A2 LoRA merges
    ("A2__Uniform__LoRA", "adapter", Path("models_hf/a2_merged_lora_merges/a2_uniform_lora")),
    ("A2__FisherAllLayers__LoRA", "adapter", Path("models_hf/a2_merged_lora_merges/a2_fisher_lora")),
    ("A2__FisherPlusSNR50__LoRA", "adapter", Path("models_hf/a2_merged_lora_merges/a2_fisher_snr50_lora")),

    # Unified hierarchical LoRA merges
    ("UNIFIED__HierA1plusA2__Uniform__LoRA", "adapter",
     Path("models_hf/unified_hierarchical_lora_merges/UNIFIED_hierarchical__A1uniform_plus_A2uniform__UNIFORM__LORA")),
    ("UNIFIED__HierA1plusA2__FisherAllLayers__LoRA", "adapter",
     Path("models_hf/unified_hierarchical_lora_merges/UNIFIED_hierarchical__A1fisher_plus_A2fisher__FISHER_allLayers__LORA")),
    ("UNIFIED__HierA1plusA2__FisherPlusSNR50_Reapplied__LoRA", "adapter",
     Path("models_hf/unified_hierarchical_lora_merges/UNIFIED_hierarchical__A1fisherSNR50_plus_A2fisherSNR50__FISHERplus_SNR50_reapplied__LORA")),
]

# =========================
# WHAT EACH MODEL EVALUATES ON
# =========================
EVAL_MAP = {
    "BASE__Mistral7BInstruct_v0_2": list(DATASETS.keys()),

    "A1__Uniform__LoRA": ["a1_lexical_test", "a1_syntactic_test", "a1_semantic_test", "real_97_test"],
    "A1__FisherAllLayers__LoRA": ["a1_lexical_test", "a1_syntactic_test", "a1_semantic_test", "real_97_test"],
    "A1__FisherPlusSNR50__LoRA": ["a1_lexical_test", "a1_syntactic_test", "a1_semantic_test", "real_97_test"],

    "A2__Uniform__LoRA": ["a2_early_1260_test", "a2_middle_1310_test", "a2_late_1360_test", "real_97_test"],
    "A2__FisherAllLayers__LoRA": ["a2_early_1260_test", "a2_middle_1310_test", "a2_late_1360_test", "real_97_test"],
    "A2__FisherPlusSNR50__LoRA": ["a2_early_1260_test", "a2_middle_1310_test", "a2_late_1360_test", "real_97_test"],

    "UNIFIED__HierA1plusA2__Uniform__LoRA": list(DATASETS.keys()),
    "UNIFIED__HierA1plusA2__FisherAllLayers__LoRA": list(DATASETS.keys()),
    "UNIFIED__HierA1plusA2__FisherPlusSNR50_Reapplied__LoRA": list(DATASETS.keys()),
}

def main():
    all_results = []

    for name, mtype, path in MODELS:
        if mtype != "base":
            if not Path(path).exists():
                print(f"[SKIP] Missing path for {name}: {path}")
                continue

        allowed = EVAL_MAP.get(name, [])
        if not allowed:
            print(f"[SKIP] No EVAL_MAP entry for {name}")
            continue

        results = evaluate(name, mtype, path, allowed)
        all_results.extend(results)

    # Printed summary (stable formatting)
    for r in all_results:
        m = r["metrics"]
        if not m.get("has_reference"):
            continue
        print(
            f"{r['model']:55s} | {r['dataset']:22s} | "
            f"BLEU={m.get('bleu',0):.2f} | chrF={m.get('chrf++',0):.2f} | "
            f"ROUGE-L={m.get('rougeL',0):.2f} | METEOR={m.get('meteor',0):.4f} | "
            f"BERT={m.get('bertscore_f1',0)*100:.2f} | TokF1={m.get('token_f1',0)*100:.2f}"
        )

if __name__ == "__main__":
    main()