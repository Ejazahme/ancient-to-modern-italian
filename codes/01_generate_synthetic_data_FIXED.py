import os
import json
import time
import random
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("dev.env") 
API_KEY =""
MODEL_NAME = "gpt-4.1"
MAX_RETRIES = 3
SAMPLES_PER_SCENARIO = (8, 12)  
SAMPLES_PER_PERIOD = 50 
OUTPUT_DIR = Path("data/synthetic")
RAW_DATA_PATH = Path("data/raw/task2-archaic2modern_ita.xlsx")

LEXICAL_SCENARIOS = [
    ("Lexical substitution", "Replace obsolete words with contemporary equivalents"),
    ("Verb modernization", "Replace obsolete verbs (veggere, udire) with modern equivalents"),
    ("Pronoun modernization", "Replace archaic pronouns (egli, ella, cotesto) with modern forms"),
    ("Auxiliary correction", "Modernize obsolete auxiliary verbs (esser→essere, aver→avere)"),
    ("Morphological simplification", "Simplify archaic inflections and verb forms"),
]

SYNTACTIC_SCENARIOS = [
    ("Syntactic reordering", "Reorder clauses for modern word order and readability"),
    ("Clause simplification", "Reduce redundant relative clauses or nested phrasing"),
    ("Sentence merging", "Merge fragmented archaic sentences for smoother flow"),
    ("Poetic syntax normalization", "Normalize inverted poetic word order to standard syntax"),
    ("Subject restoration", "Add explicit subjects when omitted in archaic structures"),
]

SEMANTIC_SCENARIOS = [
    ("Semantic disambiguation", "Clarify ambiguous or polysemic archaic expressions"),
    ("Idiomatic modernization", "Translate idioms into modern equivalents preserving figurative meaning"),
    ("Remove Latinisms", "Replace Latin-derived connectors with modern Italian"),
    ("Temporal modernization", "Update outdated temporal expressions (onde, giammai, poiché)"),
    ("Explicitation", "Expand implicit or elliptical expressions into explicit modern syntax"),
]

SCENARIO_CATEGORIES = {
    **{s[0]: "lexical" for s in LEXICAL_SCENARIOS},
    **{s[0]: "syntactic" for s in SYNTACTIC_SCENARIOS},
    **{s[0]: "semantic" for s in SEMANTIC_SCENARIOS}
}

ALL_SCENARIOS = LEXICAL_SCENARIOS + SYNTACTIC_SCENARIOS + SEMANTIC_SCENARIOS

TEMPORAL_PERIODS = {
    "early_1260": {
        "range": "1260-1310",
        "characteristics": [
            "Heavy Latinisms and Latin-derived syntax",
            "Archaic verb endings (-oe, -ae endings common)",
            "Frequent use of 'fue' instead of 'fu'",
            "Medieval vocabulary (veggere, udire, cotesto)",
            "Inverted word order typical of Latin influence",
            "Era of early Florentine writers before Dante"
        ],
        "example_features": "elli fue grande, veggere, cotesto, onde"
    },
    "middle_1310": {
        "range": "1310-1360",
        "characteristics": [
            "Dante and Petrarch's influence visible",
            "Transitional forms (mixing archaic and newer patterns)",
            "Some Latin influence but more Italian structure",
            "Evolving pronouns (egli becoming more common)",
            "Mix of fue/fu verb forms",
            "Golden age of Tuscan literary language"
        ],
        "example_features": "egli era, some modernization, pero, giammai"
    },
    "late_1360": {
        "range": "1360-1415",
        "characteristics": [
            "Boccaccio's clearer prose style emerging",
            "Moving toward Renaissance Italian",
            "Less Latin syntax, more natural Italian word order",
            "Modern verb forms more common (fu over fue)",
            "Reduced archaic pronouns",
            "Pre-Renaissance transitional period"
        ],
        "example_features": "Modern-like structure, fu, simplified syntax"
    }
}

MODERNIZATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "archaic_to_modern",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
                "scenario": {"type": "string"},
                "author": {"type": "string"},
                "date": {"type": "string"},
                "region": {"type": "string"}
            },
            "required": ["source", "target", "scenario", "author", "date", "region"],
            "additionalProperties": False
        }
    }
}


def load_source_data():

    df = pd.read_excel(RAW_DATA_PATH)
    df = df.dropna(subset=["Sentence"]).copy()

    sentences_with_metadata = []
    for _, row in df.iterrows():
        sentences_with_metadata.append({
            "sentence": str(row["Sentence"]).strip(),
            "author": str(row.get("Author", "Unknown")),
            "date": str(row.get("Date", "Unknown")),
            "region": str(row.get("Region", "Unknown"))
        })

    return sentences_with_metadata

def categorize_by_period(sentences_with_metadata):

    def extract_year(date_str):
        match = re.search(r'(\d{4})', str(date_str))
        return int(match.group(1)) if match else None

    categorized = {
        "early_1260": [],
        "middle_1310": [],
        "late_1360": [],
        "unknown": []
    }

    for item in sentences_with_metadata:
        year = extract_year(item["date"])

        if year and 1260 <= year < 1310:
            categorized["early_1260"].append(item)
        elif year and 1310 <= year < 1360:
            categorized["middle_1310"].append(item)
        elif year and 1360 <= year <= 1415:
            categorized["late_1360"].append(item)
        else:
            categorized["unknown"].append(item)

    return categorized

def build_linguistic_prompt(data_item, scenario_name, scenario_desc):
    """Build scenario-specific prompt (Approach 1)"""
    return f"""You are an expert in historical Italian linguistics specializing in modernizing 13th-15th century texts.

**Task**: Modernize the following archaic Italian sentence according to the specific scenario.

**Modernization Scenario**: {scenario_name}
**Scenario Objective**: {scenario_desc}

**Original Sentence**:
"{data_item['sentence']}"

**Metadata**:
- Author: {data_item['author']}
- Period: {data_item['date']}
- Region: {data_item['region']}

**Instructions**:
1. Apply ONLY transformations relevant to "{scenario_name}"
2. Preserve meaning and semantic content
3. Ensure output is natural 2025 Italian
4. Return ONLY the modernized sentence in the target field
5. Do NOT include explanations in the target field

**CRITICAL**: The "target" field must contain ONLY the modernized Italian sentence."""

def build_temporal_prompt(data_item, period_key, period_info):
    """Build period-aware prompt (Approach 2)"""
    characteristics = "\n  - ".join(period_info["characteristics"])

    return f"""You are an expert in diachronic Italian linguistics specializing in {period_info['range']}.

**Task**: Modernize this archaic Italian sentence from {period_info['range']} to 2025 Italian.

**Period Context ({period_info['range']})**:
Linguistic characteristics of this era:
  - {characteristics}

Typical features: {period_info['example_features']}

**Original Sentence**:
"{data_item['sentence']}"

**Metadata**:
- Author: {data_item['author']}
- Date: {data_item['date']} (within {period_info['range']})
- Region: {data_item['region']}

**Instructions**:
1. Recognize period-specific archaic features (see characteristics above)
2. Transform according to this era's linguistic patterns
3. Preserve original meaning exactly
4. Output natural 2025 Italian
5. Return ONLY the modernized sentence in target field (no explanations!)

**CRITICAL**: The "target" field must contain ONLY the modern Italian sentence."""

def safe_schema_call(prompt, data_item, scenario_name):
    client = OpenAI(api_key=API_KEY)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.7,
                max_tokens=512,
                response_format=MODERNIZATION_SCHEMA,
                messages=[
                    {"role": "system", "content": "Generate JSON conforming to schema. Target field has ONLY the modernized sentence."},
                    {"role": "user", "content": prompt}
                ]
            )

            result = json.loads(response.choices[0].message.content)

            result["source"] = data_item["sentence"]
            result["author"] = data_item["author"]
            result["date"] = data_item["date"]
            result["region"] = data_item["region"]
            result["scenario"] = scenario_name

            return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)

    return None

def generate_linguistic_data(sentences):
   

    records = []

    for scenario_name, scenario_desc in tqdm(ALL_SCENARIOS, desc="Linguistic scenarios"):
        n_samples = random.randint(*SAMPLES_PER_SCENARIO)
        chosen = random.sample(sentences, min(len(sentences), n_samples))

        for data_item in tqdm(chosen, desc=f"  {scenario_name}", leave=False):
            prompt = build_linguistic_prompt(data_item, scenario_name, scenario_desc)
            result = safe_schema_call(prompt, data_item, scenario_name)

            if result and all(k in result for k in ["source", "target"]):
                result["category"] = SCENARIO_CATEGORIES.get(scenario_name, "unknown")
                records.append(result)

            time.sleep(0.5)

    return records

def generate_temporal_data(categorized_sentences):
    
    records = []

    for period_key, period_info in TEMPORAL_PERIODS.items():
        base_sentences = categorized_sentences[period_key]

        if len(base_sentences) == 0:
            continue

        n_needed = max(SAMPLES_PER_PERIOD, len(base_sentences) * 3)
        n_per_sentence = (n_needed + len(base_sentences) - 1) // len(base_sentences)

        for data_item in tqdm(base_sentences, desc=f"  {period_key}"):
            for _ in range(n_per_sentence):
                prompt = build_temporal_prompt(data_item, period_key, period_info)
                result = safe_schema_call(prompt, data_item, f"temporal_{period_key}")

                if result and all(k in result for k in ["source", "target"]):
                    result["category"] = "temporal"
                    result["period"] = period_key
                    records.append(result)

                time.sleep(0.5)

    return records

def main():

    if not API_KEY:
        return

    sentences = load_source_data()
    categorized = categorize_by_period(sentences)

    linguistic_records = generate_linguistic_data(sentences)

    temporal_records = generate_temporal_data(categorized)

    all_records = linguistic_records + temporal_records

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "synthetic_all.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
