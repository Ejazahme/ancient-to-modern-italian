import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

API_KEY = "OPEN AI KEY"
MODEL_NAME = "gpt-4.1"
MAX_RETRIES = 3
TEMPERATURE = 0.4
OUTPUT_ROOT = Path("outputs/gpt4_1_translations")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "a1_lexical_test":   Path("data/processed/approach1/lexical/test.jsonl"),
    "a1_syntactic_test": Path("data/processed/approach1/syntactic/test.jsonl"),
    "a1_semantic_test":  Path("data/processed/approach1/semantic/test.jsonl"),
    "a2_early_1260_test":  Path("data/processed/approach2/early_1260/test.jsonl"),
    "a2_middle_1310_test": Path("data/processed/approach2/middle_1310/test.jsonl"),
    "a2_late_1360_test":   Path("data/processed/approach2/late_1360/test.jsonl"),
    "real_97_test": Path("data/processed/test_ground_truth.jsonl"),
}

TRANSLATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "archaic_to_modern_translation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "translation": {
                    "type": "string",
                    "description": "Modern Italian translation of the input sentence"
                }
            },
            "required": ["translation"],
            "additionalProperties": False
        }
    }
}

def load_dataset(path):
    if not path.exists():
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            src = item.get("source") or item.get("archaic") or item.get("input")
            ref = item.get("reference") or item.get("target") or None
            if not src:
                continue
            data.append({"id": item.get("id"), "source": src, "reference": ref})
    return data


def build_prompt(text):
    return f"""You are a professional Italian linguist specialized in transforming
archaic (13th–15th century) Italian into natural 2025 Italian.

### Task
Translate the following archaic Italian sentence into fluent modern Italian,
preserving meaning and stylistic naturalness.

### Archaic Sentence:
{text}

### Output rules
- Keep the meaning identical
- Use modern vocabulary and syntax
- Output only the translation text, no explanations
- The translation must fit in the 'translation' JSON field
"""


def translate_with_retry(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=256,
                response_format=TRANSLATION_SCHEMA,
                messages=[
                    {"role": "system", "content": "Translate archaic Italian to modern Italian using the defined JSON schema."},
                    {"role": "user", "content": prompt}
                ]
            )
            data = json.loads(response.choices[0].message.content)
            return data["translation"]
        except Exception as e:
            time.sleep(5)
    return None


def main():

    client = OpenAI(api_key=API_KEY)

    for ds_name, ds_path in DATASETS.items():
        dataset = load_dataset(ds_path)
        if not dataset:
            continue

        translated = []

        for item in tqdm(dataset, desc=f"Translating {ds_name}"):
            prompt = build_prompt(item["source"])
            translation = translate_with_retry(client, prompt)
            if not translation:
                continue

            translated.append({
                "id": item.get("id"),
                "source": item["source"],
                "prediction": translation,
                "reference": item.get("reference")
            })
            time.sleep(0.5)

        out_file = OUTPUT_ROOT / f"gpt4_1__on__{ds_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "gpt-4.1",
                "dataset_name": ds_name,
                "dataset_path": str(ds_path),
                "num_samples": len(translated),
                "predictions": translated
            }, f, indent=2, ensure_ascii=False)

        time.sleep(1)


if __name__ == "__main__":
    main()