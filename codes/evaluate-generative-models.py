import os
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from evaluate import load
from bert_score import score as bert_score
import sacrebleu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_FOLDERS = {
    "gpt4_1": Path("outputs/gpt4_1_translations"),
    "gpt4_1_fewshot": Path("outputs/gpt4_1_translations_fewshot"),
    "gemini_2_5": Path("outputs/gemini2_5_translations"),
    "gemini_2_5_fewshot": Path("outputs/gemini2_5_translations_fewshots"),
}

OUTPUT_ROOT = Path("outputs/generative_evaluation")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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
        rouge_refs = [[r] for r in refs_text]

        rouge = rouge_metric.compute(
            predictions=preds_text,
            references=rouge_refs
        )
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
        rec  = len(inter)/len(rs) if rs else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        token_f1s.append(f1)
    m["token_f1"] = float(np.mean(token_f1s))

    try:
        P, R, F1 = bert_score(
            preds_text,
            refs_text,
            lang="it",
            model_type="bert-base-multilingual-cased",
            device=DEVICE
        )
        m["bertscore_f1"] = float(F1.mean().item())
    except Exception as e:
        m["bertscore_error"] = str(e)

    m["num_predictions"] = len(preds_text)
    m["has_reference"] = True
    return m


def load_translation_file(path: Path):
    """Load single generative output file."""
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        return data.get("predictions", [])
    except:
        return []


def main():
    all_results = []

    for model_tag, folder in INPUT_FOLDERS.items():
        if not folder.exists():
            continue

        out_model_dir = OUTPUT_ROOT / model_tag
        out_model_dir.mkdir(exist_ok=True, parents=True)

        for f in folder.glob("*.json"):
            preds = load_translation_file(f)
            if not preds:
                continue

            metrics = compute_metrics(preds)

            output_name = f.name.replace(".json", "__metrics.json")
            out_path = out_model_dir / output_name

            json.dump({
                "model_tag": model_tag,
                "input_file": str(f),
                "num_samples": len(preds),
                "metrics": metrics,
                "predictions": preds
            }, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

            all_results.append((model_tag, f.name, metrics))

    for model_tag, fname, m in all_results:
        if not m.get("has_reference"):
            continue
        


if __name__ == "__main__":
    main()
