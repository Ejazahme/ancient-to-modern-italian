# From Ancient to Modern Italian

This project addresses the task of **translating archaic Italian (13th–15th century) into contemporary Italian** using neural machine translation models.

The goal is to modernize historical Italian text while preserving its original meaning, improving readability for modern readers.

---

## Approach

The project explores **multi-expert neural translation models** fine-tuned for archaic-to-modern Italian transformation. The system leverages specialized experts that capture different linguistic characteristics of historical Italian and combines them into unified translation models.

Key ideas include:

* Modeling lexical, syntactic, and semantic modernization
* Handling strong temporal variation in historical Italian
* Evaluating fine-tuned models against strong generative LLM baselines

---

## Data

The evaluation is performed on a curated dataset of **archaic Italian sentences** paired with modern Italian translations. The dataset covers multiple historical periods and linguistic styles.

---

## Results

Models are evaluated using standard machine translation metrics, including BLEU, chrF++, ROUGE-L, METEOR, BERTScore, and token-level F1, with both quantitative and qualitative analyses.

Detailed experimental results and analyses are provided in the accompanying documentation.

---

## Repository Structure

```
.
├── data/          # Raw, synthetic, and processed datasets
├── outputs/       # Evaluation results and model predictions
├── documents_and_results_summary/
└── README.md
```

---

## License

This project is released under the **Apache License 2.0**.

---
