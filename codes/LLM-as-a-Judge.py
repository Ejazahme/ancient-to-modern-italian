import os
import json
import time
import random
import statistics
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

MODEL_JUDGE = "gpt-4.1"
TEMPERATURE = 0.3
MAX_TOKENS = 800

API_KEY="OPEN AI KEY"
INPUT_FOLDER = Path("outputs")
OUTPUT_ROOT = Path("results/llm-as-a-judge/finetuned")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "llm_translation_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "Faithfulness": {
                    "type": "object",
                    "properties": {
                        "feedback": {"type": "string"},
                        "score": {"type": "integer", "minimum": 1, "maximum": 5}
                    },
                    "required": ["feedback", "score"],
                    "additionalProperties": False
                },
                "Fluency": {
                    "type": "object",
                    "properties": {
                        "feedback": {"type": "string"},
                        "score": {"type": "integer", "minimum": 1, "maximum": 5}
                    },
                    "required": ["feedback", "score"],
                    "additionalProperties": False
                },
                "Style": {
                    "type": "object",
                    "properties": {
                        "feedback": {"type": "string"},
                        "score": {"type": "integer", "minimum": 1, "maximum": 5}
                    },
                    "required": ["feedback", "score"],
                    "additionalProperties": False
                },
                "Overall": {
                    "type": "object",
                    "properties": {
                        "feedback": {"type": "string"},
                        "score": {"type": "integer", "minimum": 1, "maximum": 5}
                    },
                    "required": ["feedback", "score"],
                    "additionalProperties": False
                }
            },
            "required": ["Faithfulness", "Fluency", "Style", "Overall"],
            "additionalProperties": False
        }
    }
}

RUBRICS = {
    "Faithfulness": """Evaluate meaning preservation and modernization accuracy.
5 – Excellent: Meaning fully preserved and properly modernized.
4 – Good: Minor nuance loss or slightly literal.
3 – Fair: Understandable but partial misinterpretation.
2 – Poor: Major omission or distortion.
1 – Fail: Meaning lost or wrong.""",

    "Fluency": """Evaluate grammar, syntax, and comprehensibility.
5 – Excellent: Flawless, idiomatic modern Italian.
4 – Good: Minor slips but readable.
3 – Fair: Awkward phrasing or rigid syntax.
2 – Poor: Frequent grammatical errors.
1 – Fail: Broken or unreadable.""",

    "Style": """Evaluate naturalness and human-likeness of the translation.
5 – Excellent: Feels human, elegant, and modern.
4 – Good: Smooth, slightly mechanical.
3 – Fair: Understandable but stiff.
2 – Poor: Robotic tone or phrasing.
1 – Fail: Stylistically incoherent.""",

    "Overall": """Evaluate the holistic translation quality combining all above criteria.
5 – Excellent: Faithful, fluent, and natural.
4 – Good: Minor issues not affecting comprehension.
3 – Fair: Acceptable but mechanical.
2 – Poor: Noticeable problems.
1 – Fail: Unusable translation."""
}

def build_prompt(item):
    return f"""
You are a bilingual Italian linguist specialized in translating and evaluating
texts from medieval Italian (13th–15th century) to modern Italian.

Evaluate the following translation according to the rubrics below.
Use the "Reference Translation" only as a semantic anchor — not a gold standard.
Do not paraphrase or retranslate; only judge the given text.

Provide feedback ≤50 words per criterion, strictly following the rubrics.

### Archaic Sentence
{item['source']}

### Model Translation
{item['prediction']}

### Reference Translation (semantic anchor)
{item.get('reference', 'N/A')}

### Rubrics
1. Faithfulness – {RUBRICS['Faithfulness']}
2. Fluency – {RUBRICS['Fluency']}
3. Style – {RUBRICS['Style']}
4. Overall – {RUBRICS['Overall']}
""".strip()

def _mk_client():
    # Create a client per worker thread (safer than sharing one instance)
    return OpenAI(api_key=API_KEY)

def judge_item(client: OpenAI, item, retries=5):
    prompt = build_prompt(item)

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_JUDGE,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format=JUDGE_SCHEMA,
                messages=[
                    {"role": "system", "content": "You are a neutral translation evaluator. Output must strictly follow the provided JSON schema."},
                    {"role": "user", "content": prompt},
                ],
            )
            return json.loads(resp.choices[0].message.content)

        except Exception as e:
            # Exponential backoff + jitter (helps with 429 / transient errors)
            sleep_s = min(60, (2 ** (attempt - 1)) * 2) + random.uniform(0, 1.5)
            print(f"[judge_item] attempt={attempt}/{retries} failed: {type(e).__name__}: {e} | sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)

    return None

def evaluate_dataset(path: Path, model_name: str):
    client = _mk_client()

    data = json.load(open(path, "r", encoding="utf-8"))
    preds = data.get("predictions", [])
    dataset = data.get("dataset_name", path.stem)

    results = []
    for item in tqdm(preds, desc=f"{model_name}/{dataset}", leave=False):
        evaluation = judge_item(client, item)
        if not evaluation:
            continue

        results.append({
            "id": item.get("id"),
            "source": item["source"],
            "prediction": item["prediction"],
            "reference": item.get("reference"),
            "scores": {k: evaluation[k]["score"] for k in evaluation},
            "feedback": {k: evaluation[k]["feedback"] for k in evaluation},
        })

    summary = {}
    for k in RUBRICS:
        vals = [r["scores"][k] for r in results if r["scores"].get(k)]
        if vals:
            summary[k] = {
                "mean": round(statistics.mean(vals), 3),
                "std": round(statistics.pstdev(vals), 3),
            }

    out_dir = OUTPUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "dataset_name": dataset,
            "rubrics": list(RUBRICS.keys()),
            "num_samples": len(results),
            "results": results,
            "summary": summary,
        }, f, indent=2, ensure_ascii=False)

    return str(out_file)

def collect_stratified_files():
    files = []
    for json_path in INPUT_FOLDER.rglob("*.json"):
        model_name = json_path.parent.name
        files.append((model_name, json_path))
    return files

def main(max_workers=4):
    files = collect_stratified_files()
    print(f"Found {len(files)} files. Running with max_workers={max_workers}")

    # Submit each file as a parallel job
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for model_name, fpath in files:
            futures.append(ex.submit(evaluate_dataset, fpath, model_name))

        for fut in as_completed(futures):
            try:
                out = fut.result()
                print(f"[DONE] wrote: {out}")
            except Exception as e:
                print(f"[ERROR] file job failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main(max_workers=4)