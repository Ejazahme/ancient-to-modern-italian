import os
import json
import time
import statistics
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

API_KEY = "OPEN API KEY"

MODEL_JUDGE = "gpt-4.1"
TEMPERATURE = 0.3
MAX_TOKENS = 800

client = OpenAI(api_key=API_KEY)

INPUT_FOLDER = Path("outputs/stratified_evaluation")
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

def judge_item(item):
    prompt = build_prompt(item)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_JUDGE,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format=JUDGE_SCHEMA,
                messages=[
                    {"role": "system", "content": "You are a neutral translation evaluator. Output must strictly follow the provided JSON schema."},
                    {"role": "user", "content": prompt}
                ]
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            
            time.sleep(5)
    return None

def evaluate_dataset(path, model_name):
    data = json.load(open(path, "r", encoding="utf-8"))
    preds = data.get("predictions", [])
    dataset = data.get("dataset_name", Path(path).stem)

    results = []

    for item in tqdm(preds):
        evaluation = judge_item(item)
        if not evaluation:
            continue
        results.append({
            "id": item.get("id"),
            "source": item["source"],
            "prediction": item["prediction"],
            "reference": item.get("reference"),
            "scores": {k: evaluation[k]["score"] for k in evaluation},
            "feedback": {k: evaluation[k]["feedback"] for k in evaluation}
        })

    summary = {}
    for k in RUBRICS:
        vals = [r["scores"][k] for r in results if r["scores"].get(k)]
        if vals:
            summary[k] = {
                "mean": round(statistics.mean(vals), 3),
                "std": round(statistics.pstdev(vals), 3)
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
            "summary": summary
        }, f, indent=2, ensure_ascii=False)


def collect_stratified_files():
    files = []
    for subdir in INPUT_FOLDER.iterdir():
        if not subdir.is_dir():
            continue
        model_name = subdir.name
        for f in subdir.glob("*.json"):
            files.append((model_name, f))
    return files

def main():
    files = collect_stratified_files()
    for model_name, fpath in files:
        evaluate_dataset(fpath, model_name)

if __name__ == "__main__":
    main()
