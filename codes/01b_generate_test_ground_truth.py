"""
Generate Ground Truth for 97 Test Sentences
This creates modern translations for the original corpus
ONLY FOR EVALUATION - NOT used in training!
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from openai import OpenAI

API_KEY =""
MODEL_NAME = "gpt-4.1"
MAX_RETRIES = 3
RAW_DATA_PATH = Path("data/raw/task2-archaic2modern_ita.xlsx")
OUTPUT_DIR = Path("data/processed")

TEST_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "test_ground_truth",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "modern_translation": {
                    "type": "string",
                    "description": "Modern Italian translation (2025)"
                }
            },
            "required": ["modern_translation"],
            "additionalProperties": False
        }
    }
}

def load_test_sentences():
    """Load the 97 original sentences"""
    df = pd.read_excel(RAW_DATA_PATH)
    df = df.dropna(subset=["Sentence"]).copy()
    
    test_sentences = []
    for idx, row in df.iterrows():
        test_sentences.append({
            "id": idx,
            "source": str(row["Sentence"]).strip(),
            "author": str(row.get("Author", "Unknown")),
            "date": str(row.get("Date", "Unknown")),
            "region": str(row.get("Region", "Unknown"))
        })

    return test_sentences

def generate_ground_truth(test_sentence):
    """Generate modern translation for one test sentence"""
    
    prompt = f"""You are an expert translator of historical Italian to modern Italian.

**TASK**: Translate this 13th-15th century Italian sentence to natural modern Italian (2025).

**Archaic Sentence**:
"{test_sentence['source']}"

**Metadata**:
- Author: {test_sentence['author']}
- Date: {test_sentence['date']}
- Region: {test_sentence['region']}

**Instructions**:
1. Provide a faithful, natural modern Italian translation
2. Preserve the original meaning completely
3. Use contemporary vocabulary and syntax
4. Make it sound like native 2025 Italian
5. Return ONLY the translation in the "modern_translation" field

**CRITICAL**: The translation must be accurate and natural."""

    client = OpenAI(api_key=API_KEY)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.3,  
                max_tokens=256,
                response_format=TEST_SCHEMA,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional Italian translator. Provide accurate, natural modern Italian translations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["modern_translation"]
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
    
    return None

def main():
    """Generate ground truth for all 97 test sentences"""
    if not API_KEY:
        return

    test_sentences = load_test_sentences()

    test_data = []
    
    for test_sent in tqdm(test_sentences, desc="Translating"):
        modern = generate_ground_truth(test_sent)
        
        if modern:
            test_data.append({
                "id": test_sent["id"],
                "source": test_sent["source"],
                "reference": modern,
                "author": test_sent["author"],
                "date": test_sent["date"],
                "region": test_sent["region"]
            })
        else:
            pass
        
        time.sleep(0.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "test_ground_truth.jsonl"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
