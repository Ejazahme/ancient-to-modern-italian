from google import genai
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from pydantic import BaseModel, Field
import json, time, re

API_KEY = "A"  
MODEL_NAME = "gemini-2.0-flash"                     
TEMPERATURE = 0.4
MAX_RETRIES = 3
OUTPUT_ROOT = Path("outputs/gemini2_5_translations_fewshots")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
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


def build_prompt_fewshot(text):
    examples = """
### Example 1
"source": "sono due già non in una carne, ma in uno spirito, cioè Iddio, e l' anima. Onde in altro luogo dice S. Paolo: Chi s' accosta a Dio è uno spirito", "target": "sono due già non in una carne, ma in uno spirito, cioè Dio e l'anima. Onde in un altro luogo dice San Paolo: Chi si accosta a Dio è uno spirito"


### Example 2
"source": "Altressì uno amante chiamando merzé alla sua donna dice parole e ragioni molte, et ella si difende in suo dire.", "target": "Allo stesso modo, un uomo innamorato, mentre chiede misericordia alla donna che ama, le rivolge molte parole e argomentazioni, ed essa risponde difendendo la propria posizione con le sue parole."


### Example 3
"source": "Andò nel campo de' Cartaginesi e tutta la legione trasse seco.", "target": "Trasse con sé tutta la legione e andò nel campo dei Cartaginesi."

### Example 4
"source": "la moltitudine de' quali tu ài potuto vedere e riguardare lo studio e poco dinanzi udire le voci, e lle cui mani e lance apena posso ritenere.", "target": "La moltitudine della quale hai potuto vedere e osservare l'impegno e poco fa ascoltare le voci, e le cui mani e lance riesco a stento a trattenere."
### Example 5

"source": "I vendimenti de' morti et le presure de' vivi fece la frode d'uno feroce re.", "reference": "Le vendite dei beni dei morti e le prigionie dei vivi furono causate dall’inganno di un re crudele."


"""
    return f"""You are a professional Italian linguist specialized in transforming
archaic (13th–15th century) Italian into natural 2025 Italian.

Learn from the examples below and translate the following archaic sentence
into fluent modern Italian while preserving meaning and tone.

{examples}

### Sentence to translate:
{text}

### Output format
Respond strictly in JSON as:
{{"translation": "<modern translation>"}}
"""

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
            translation = safe_generate(build_prompt_fewshot(item["source"]))
            if translation:
                results.append({
                    "id": item.get("id"),
                    "source": item["source"],
                    "prediction": translation,
                    "reference": item.get("reference"),
                })
            time.sleep(5)

        out_file = OUTPUT_ROOT / f"gemini2_5_fewshot_on__{ds_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
