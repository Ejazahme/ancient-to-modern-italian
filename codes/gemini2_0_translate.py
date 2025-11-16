from google import genai
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from pydantic import BaseModel, Field
import json, time, re

API_KEY = "gemini api key" 
MODEL_NAME = "gemini-2.0-flash"         
TEMPERATURE = 0.4
MAX_RETRIES = 3
OUTPUT_ROOT = Path("outputs/gemini2_5_translations")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "a1_lexical_test":   Path("data/processed/approach1/lexical/test.jsonl"),
    "a1_syntactic_test": Path("data/processed/approach1/syntactic/test.jsonl"),
    "a1_semantic_test":  Path("data/processed/approach1/semantic/test.jsonl"),
    "a2_early_1260_test":  Path("data/processed/approach2/early_1260/test.jsonl"),
    "a2_middle_1310_test": Path("data/processed/approach2/middle_1310/test.jsonl"),
    "a2_late_1360_test":   Path("data/processed/approach2/late_1360/test.jsonl"),
    "real_97_test":         Path("data/processed/test_ground_truth.jsonl"),
}

client = genai.Client(api_key=API_KEY)

translation_schema = {
    "type": "object",
    "properties": {
        "translation": {
            "type": "string",
            "description": "Modern Italian translation of the input sentence."
        }
    },
    "required": ["translation"]
}

def load_dataset(path: Path):
    if not path.exists():
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            src = item.get("source") or item.get("archaic") or item.get("input")
            ref = item.get("reference") or item.get("target")
            if not src:
                continue
            data.append({"id": item.get("id"), "source": src, "reference": ref})
    return data


def build_prompt(text: str) -> str:
    """Exact same semantics as GPT-4.1 prompt (no markdown)."""
    return (
        "You are a professional Italian linguist specialized in transforming "
        "archaic (13th–15th century) Italian into natural 2025 Italian. "
        "Translate the following archaic sentence into fluent modern Italian, "
        "preserving meaning and stylistic naturalness. "
        "Output only the translation text, no explanations. "
        f"Archaic sentence: {text}"
    )


def extract_text(response):
    for attr in ["text", "output_text"]:
        if getattr(response, attr, None):
            return getattr(response, attr).strip()
    if response.candidates:
        parts = response.candidates[0].content.parts
        if parts and getattr(parts[0], "text", None):
            return parts[0].text.strip()
    return None


def safe_generate(prompt: str):
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": 256,
                    "response_mime_type": "application/json",
                    "response_json_schema": translation_schema,
                },
            )

            text = extract_text(response)
            if not text:
                raise ValueError("Empty response text (no valid field found)")

            try:
                data = json.loads(text)
                return data.get("translation")
            except json.JSONDecodeError:
                return text.strip()

        except Exception as e:
            msg = str(e)
            match = re.search(r"retryDelay['\"]?: ['\"]?(\d+)s", msg)
            if match:
                delay = int(match.group(1)) + 3
                print(f"⏳ Quota reached. Waiting {delay}s before retrying...")
                time.sleep(delay)
                continue

            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                time.sleep(60)
                continue

            time.sleep(5)
    return None


def main():
    for ds_name, ds_path in DATASETS.items():
        dataset = load_dataset(ds_path)
        if not dataset:
            continue

        results = []

        for item in tqdm(dataset, desc=f"Translating {ds_name}"):
            translation = safe_generate(build_prompt(item["source"]))
            if translation:
                results.append({
                    "id": item.get("id"),
                    "source": item["source"],
                    "prediction": translation,
                    "reference": item.get("reference"),
                })
            time.sleep(5)

        out_file = OUTPUT_ROOT / f"gemini2_5__on__{ds_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": MODEL_NAME,
                "dataset_name": ds_name,
                "dataset_path": str(ds_path),
                "num_samples": len(results),
                "predictions": results,
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
