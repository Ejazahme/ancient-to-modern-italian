# From Ancient to Modern Italian

## Project Overview

This project implements a comprehensive pipeline for translating archaic Italian text (13th-15th century) into contemporary Italian using two distinct multi-expert ensemble approaches. The models were fine-tuned on RunPod using NVIDIA RTX A4000 GPUs (16GB VRAM) and employ LoRA-based parameter-efficient fine-tuning of Mistral-7B-Instruct-v0.2.

## Approach Summary

### Approach 1: Linguistic Multi-Expert Ensemble

Three specialized experts trained on different linguistic transformation categories:

- **Lexical Expert**: Handles obsolete word replacements, verb modernization, pronoun updates, and morphological simplification
- **Syntactic Expert**: Manages clause reordering, sentence structure normalization, and poetic syntax adjustments
- **Semantic Expert**: Addresses semantic disambiguation, idiomatic modernization, and temporal expression updates

### Approach 2: Temporal-Stratified Era-Specific Ensemble

Three experts trained on period-specific linguistic characteristics:

- **Early Medieval Expert (1260-1310)**: Heavy Latinisms, archaic verb endings, early Florentine patterns
- **Middle Period Expert (1310-1360)**: Transitional forms influenced by Dante and Petrarch
- **Late Medieval Expert (1360-1415)**: Pre-Renaissance structures with clearer prose style

Both approaches utilize SNR-based parameter selection combined with RegMean merging to create unified models.

## Requirements

### Hardware

- GPU: NVIDIA RTX A4000 (16GB VRAM) or equivalent for training
- RAM: 30GB minimum
- Storage: Approximately 50GB for models and datasets
- CPU-only evaluation is supported but significantly slower

### Software

Python 3.8+ with dependencies listed in `requirements.txt`

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Initial Setup

### Required Data

You only need the original test dataset:

- **File**: `task2-archaic2modern_ita.xlsx`
- **Location**: Place in `data/raw/`
- **Content**: 97 archaic Italian sentences with metadata (author, date, region)

### Directory Structure

The following structure will be created automatically during execution:

```
.
├── data/
│   ├── raw/                          # Original Excel file
│   ├── synthetic/                    # GPT-4 generated synthetic data
│   ├── processed/                    # Train/validation/test splits
│   │   ├── approach1/
│   │   │   ├── lexical/             # Lexical expert data
│   │   │   ├── syntactic/           # Syntactic expert data
│   │   │   └── semantic/            # Semantic expert data
│   │   ├── approach2/
│   │   │   ├── early_1260/          # Early period data
│   │   │   ├── middle_1310/         # Middle period data
│   │   │   └── late_1360/           # Late period data
│   │   └── test_ground_truth.jsonl  # 97 original test sentences
│   ├── synthetic_approach1.jsonl    # Linguistic scenario data
│   └── synthetic_approach2.jsonl    # Temporal period data
├── models/
│   ├── approach1/
│   │   ├── lexical_expert/          # Trained LoRA adapter
│   │   ├── syntactic_expert/        # Trained LoRA adapter
│   │   ├── semantic_expert/         # Trained LoRA adapter
│   │   ├── merged_lora/             # SNR+RegMean merged LoRA
│   │   └── merged_full/             # Full merged model (base + LoRA)
│   ├── approach2/
│   │   ├── early_expert/            # Trained LoRA adapter
│   │   ├── middle_expert/           # Trained LoRA adapter
│   │   ├── late_expert/             # Trained LoRA adapter
│   │   ├── merged_lora/             # SNR+RegMean merged LoRA
│   │   └── merged_full/             # Full merged model
│   └── unified/
│       └── merged_full/             # Unified model (Approach 1 + 2)
├── models_hf/                        # Downloaded models from Hugging Face
├── outputs/
│   ├── finetuned_model_evaluation/  # Fine-tuned model evaluation results
│   ├── gpt4.1_and_gemini_2.0_models_evaluation/ # LLM baseline evaluation results
│   └── llm_as_a_judge_results/      # LLM-as-a-Judge qualitative assessments
├── documents_and_results_summary/
│   ├── nlp-hw-report.pdf            # Complete project report
│   ├── summary_fine_tuned_models.txt
│   ├── summary_gpt_4_1_and_gemini_models.txt
│   └── summary_fine_tuned_models_of_llm_as_a_judge.txt
└── logs/
    └── training/                     # Training logs for each expert
```

## Workflow

### Option 1: Train Models from Scratch

This workflow assumes you have access to GPU resources (RunPod recommended) and OpenAI API access for synthetic data generation.

#### Step 1: Generate Synthetic Training Data

```bash
python 01_generate_synthetic_data_FIXED.py
```

**Requirements**:

- OpenAI API key (set in script or environment variable)
- Original Excel file in `data/raw/task2-archaic2modern_ita.xlsx`

**Outputs**:

- `data/synthetic/synthetic_all.jsonl` (combined synthetic data)

**Description**: Uses GPT-4.1 to generate synthetic archaic-to-modern Italian translation pairs based on linguistic scenarios (Approach 1) and temporal periods (Approach 2).

#### Step 2: Generate Test Ground Truth

```bash
python 01b_generate_test_ground_truth.py
```

**Requirements**: OpenAI API key

**Outputs**:

- `data/processed/test_ground_truth.jsonl` (97 test sentences with modern translations)

**Description**: Creates reference translations for the 97 original sentences. This data is used only for evaluation, never for training.

#### Step 3: Split Synthetic Data by Approach

```bash
python 01c_split_synthetic_data.py
```

**Outputs**:

- `data/synthetic_approach1.jsonl` (linguistic scenarios)
- `data/synthetic_approach2.jsonl` (temporal periods)

#### Step 4: Preprocess and Create Train/Val/Test Splits

```bash
python 02_preprocess_data.py
```

**Outputs**:

- `data/processed/approach1/{lexical,syntactic,semantic}/{train,val,test}.jsonl`
- `data/processed/approach2/{early_1260,middle_1310,late_1360}/{train,val,test}.jsonl`

**Description**: Cleans data, removes duplicates, and creates stratified splits (80/10/10) for each expert category.

#### Step 5: Train Approach 1 Experts

```bash
python 03_train_approach1.py
```

**Requirements**: GPU with 16GB+ VRAM (tested on RunPod with RTX A4000)

**Outputs**:

- `models/approach1/lexical_expert/`
- `models/approach1/syntactic_expert/`
- `models/approach1/semantic_expert/`

**Training Configuration**:

- Base model: Mistral-7B-Instruct-v0.2
- LoRA rank: 16, alpha: 32
- Batch size: 1 with gradient accumulation (6 steps)
- Learning rate: 2e-4
- Epochs: 5
- Max sequence length: 256

**Duration**: Approximately 2-3 hours per expert on RTX A4000

#### Step 6: Merge Approach 1 Experts

```bash
python 03b_merge_approach1_FIXED.py
```

**Outputs**:

- `models/approach1/merged_lora/` (merged LoRA adapter)
- `models/approach1/merged_full/` (base + merged LoRA)

**Description**: Applies SNR-based parameter selection (top 50% by signal-to-noise ratio) and RegMean merging using Fisher weighting.

#### Step 7: Train Approach 2 Experts

```bash
python 04_train_approach2.py
```

**Outputs**:

- `models/approach2/early_expert/`
- `models/approach2/middle_expert/`
- `models/approach2/late_expert/`

**Training Configuration**: Identical to Approach 1

#### Step 8: Merge Approach 2 Experts

```bash
python 04b_merge_approach2_FIXED.py
```

**Outputs**:

- `models/approach2/merged_lora/`
- `models/approach2/merged_full/`

#### Step 9: Create Unified Model (Approach 1 + 2)

```bash
python 04c_merge_unified_approaches.py
```

**Outputs**:

- `models/unified/merged_full/`

**Description**: Merges the two fully-merged models (Approach 1 and Approach 2) using Fisher-weighted RegMean at the parameter level.

### Option 2: Use Pre-Trained Models from Hugging Face

If you want to skip training and use the already fine-tuned models uploaded to Hugging Face:

#### Step 1: Download Pre-Trained Models

```bash
python 07_get_models_from_hf.py
```

**Requirements**: Internet connection, Hugging Face account (optional for public models)

**Outputs**:
All models downloaded to `models_hf/` directory:

- `a1_lexical_expert/`
- `a1_syntactic_expert/`
- `a1_semantic_expert/`
- `a1_merged_lora/`
- `a1_merged_full/`
- `a2_early_expert/`
- `a2_middle_expert/`
- `a2_late_expert/`
- `a2_merged_lora/`
- `a2_merged_full/`
- `unified_merged_full/`

**Note**: Update the `HF_USERNAME` variable in the script to match your Hugging Face username where models are hosted.

#### Step 2: Evaluate Downloaded Models

```bash
python 08_evaluate_models_stratified.py
```

**Requirements**:

- GPU recommended (works on CPU but slower)
- Test data in `data/processed/`

**Outputs**:

- `outputs/stratified_evaluation/{model_name}/{dataset_name}_metrics.json`

**Metrics Computed**:

- BLEU
- chrF++
- ROUGE-L
- METEOR
- Token-level F1
- BERTScore

### Option 3: Upload Trained Models to Hugging Face

If you have trained models locally and want to share them:

```bash
python 06_upload_to_huggingface.py
```

**Requirements**:

- Hugging Face account and API token
- Trained models in `models/` directory

**Description**: Uploads all individual experts and merged models to Hugging Face Hub with standardized naming.

## Baseline Comparison: Generative LLMs

### GPT-4.1 Translation

#### Zero-Shot

```bash
python gpt4_1_translate.py
```

**Requirements**: OpenAI API key

**Outputs**: `outputs/gpt4_1_translations/*.json`

#### Few-Shot

```bash
python gpt4_1_translate_fewshot.py
```

**Requirements**: OpenAI API key

**Outputs**: `outputs/gpt4_1_translations_fewshot/*.json`

### Gemini 2.0 Translation

#### Zero-Shot

```bash
python gemini2_0_translate.py
```

**Requirements**: Google Gemini API key

**Outputs**: `outputs/gemini2_5_translations/*.json`

#### Few-Shot

```bash
python gemini2_0_fewshot_translate.py
```

**Requirements**: Google Gemini API key

**Outputs**: `outputs/gemini2_5_translations_fewshots/*.json`

### Evaluate Generative Model Outputs

```bash
python evaluate-generative-models.py
```

**Outputs**: `outputs/generative_evaluation/{model_tag}/*.json`

**Description**: Computes the same metrics (BLEU, chrF++, ROUGE-L, METEOR, Token-F1, BERTScore) for GPT-4.1 and Gemini 2.0 translations.

## LLM-as-a-Judge Evaluation

For qualitative assessment of fine-tuned model outputs:

```bash
python LLM-as-a-Judge.py
```

**Requirements**:

- OpenAI API key
- Evaluation files in `outputs/stratified_evaluation/`

**Outputs**: `results/llm-as-a-judge/finetuned/{model_name}_judged.json`

**Evaluation Criteria**:

- **Faithfulness** (1-5): Meaning preservation and modernization accuracy
- **Fluency** (1-5): Grammar, syntax, and comprehensibility
- **Style** (1-5): Naturalness and human-likeness
- **Overall** (1-5): Holistic translation quality

**Judge Model**: GPT-4.1 with structured rubric-based assessment

## Configuration

### API Keys

Set the following API keys in the respective scripts before running:

**OpenAI** (required for synthetic data generation, GPT-4.1 baseline, LLM-as-a-Judge):

- Files: `01_generate_synthetic_data_FIXED.py`, `01b_generate_test_ground_truth.py`, `gpt4_1_translate.py`, `gpt4_1_translate_fewshot.py`, `LLM-as-a-Judge.py`
- Variable: `API_KEY`

**Google Gemini** (required for Gemini baseline):

- Files: `gemini2_0_translate.py`, `gemini2_0_fewshot_translate.py`
- Variable: `API_KEY`

**Hugging Face** (optional, for model upload/download):

- Set via `huggingface-cli login` or environment variable `HF_TOKEN`

### Training Hyperparameters

All training scripts use consistent hyperparameters optimized for 16GB VRAM:

- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- LoRA config: r=16, alpha=32, dropout=0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Batch size: 1 per device
- Gradient accumulation: 6 steps
- Learning rate: 2e-4 with linear warmup (20 steps)
- Epochs: 5
- Max sequence length: 256 tokens
- Precision: FP16 with gradient checkpointing

## Results Directory Structure

After running all evaluation scripts, results are organized as follows:

```
outputs/
├── finetuned_model_evaluation/     # Fine-tuned model results
│   ├── a1_lexical_expert/
│   ├── a1_syntactic_expert/
│   ├── a1_semantic_expert/
│   ├── a1_merged_lora/
│   ├── a1_merged_full/
│   ├── a2_early_expert/
│   ├── a2_middle_expert/
│   ├── a2_late_expert/
│   ├── a2_merged_lora/
│   ├── a2_merged_full/
│   └── unified_merged_full/
├── gpt4.1_and_gemini_2.0_models_evaluation/  # LLM baseline results
│   ├── gpt4_1_translations/
│   ├── gpt4_1_translations_fewshot/
│   ├── gemini2_5_translations/
│   └── gemini2_5_translations_fewshots/
└── llm_as_a_judge_results/
    ├── LLM_as_a_judge_finetuned_model_evaluation/
    └── LLM_as_a_judge_gpt4.1_and_gemini_2.0_models_evaluation/

documents_and_results_summary/
├── nlp-hw-report.pdf
├── summary_fine_tuned_models.txt
├── summary_gpt_4_1_and_gemini_models.txt
└── summary_fine_tuned_models_of_llm_as_a_judge.txt
```

Each metrics file contains:

- Quantitative scores (BLEU, chrF++, ROUGE-L, METEOR, BERTScore, Token-F1)
- Per-sample predictions with references
- Metadata (model name, dataset, timestamp)
- File naming format: `{model_name}__on__{test_dataset}__YYYYMMDD_HHMMSS.json` or `{model_name}__on__{test_dataset}.json`

**Available Test Datasets:**

- `real_97_test` - 97 authentic archaic Italian sentences (primary evaluation)
- `a1_lexical_test`, `a1_syntactic_test`, `a1_semantic_test` - Approach 1 expert-specific tests
- `a2_early_1260_test`, `a2_middle_1310_test`, `a2_late_1360_test` - Approach 2 temporal period tests

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length` to 128 or 192

**2. Missing Data Files**

- Ensure `task2-archaic2modern_ita.xlsx` is in `data/raw/`
- Run synthetic data generation scripts in order (01, 01b, 01c, 02)

**3. API Rate Limits**

- Add `time.sleep()` between API calls
- Reduce batch sizes in translation scripts

**4. Model Download Failures**

- Check internet connection
- Verify Hugging Face username in `07_get_models_from_hf.py`
- Ensure sufficient disk space (approximately 50GB)

**5. Evaluation Crashes**

- Ensure test data exists in expected locations
- Check GPU availability for BERTScore computation
- Verify model paths in evaluation scripts

## Pre-Generated Data and Results (For Exploration Without Running Code)

If you want to explore the repository and understand the project without running any scripts, we provide pre-generated data, model evaluation results, and comprehensive documentation.

### Available Pre-Generated Data

#### 1. Synthetic Training Data (`data/` directory)

The following synthetic datasets are already available:

- **`data/synthetic_approach1.jsonl`**: Synthetic training data for Approach 1 (Linguistic Multi-Expert Ensemble)

  - Contains examples categorized by linguistic transformation types (lexical, syntactic, semantic)
  - Generated using GPT-4.1 based on linguistic scenarios

- **`data/synthetic_approach2.jsonl`**: Synthetic training data for Approach 2 (Temporal-Stratified Ensemble)

  - Contains examples from different historical periods (Early 1260-1310, Middle 1310-1360, Late 1360-1415)
  - Generated using GPT-4.1 based on temporal characteristics

- **`data/processed/test_ground_truth.jsonl`**: Real test dataset ("real_97_test")

  - Contains 97 authentic archaic Italian sentences from the original Excel file
  - Includes modern Italian translations used as ground truth for evaluation
  - Never used for training, only for testing and evaluation

- **`data/synthetic/synthetic_all.jsonl`**: Combined synthetic dataset
  - All synthetic examples before splitting into Approach 1 and Approach 2
  - Includes metadata: author, date, region, category, and scenario

**Additional Processed Test Datasets:**

In `data/processed/approach1/` and `data/processed/approach2/`, you'll find expert-specific test sets:

- `approach1/lexical/test.jsonl` - Lexical transformation test set
- `approach1/syntactic/test.jsonl` - Syntactic transformation test set
- `approach1/semantic/test.jsonl` - Semantic transformation test set
- `approach2/early_1260/test.jsonl` - Early Medieval period test set
- `approach2/middle_1310/test.jsonl` - Middle period test set
- `approach2/late_1360/test.jsonl` - Late Medieval period test set

Each directory also contains `train.jsonl` and `val.jsonl` for training and validation.

#### 2. Model Evaluation Results (`outputs/` directory)

All model evaluation results are available as JSON files, organized by model type:

##### Fine-Tuned Models Evaluation (`outputs/finetuned_model_evaluation/`)

Contains evaluation metrics for all fine-tuned models on different test sets:

**Approach 1 Models:**

- `a1_lexical_expert/`: Lexical expert model results
- `a1_syntactic_expert/`: Syntactic expert model results
- `a1_semantic_expert/`: Semantic expert model results
- `a1_merged_lora/`: Merged LoRA adapter results
- `a1_merged_full/`: Fully merged Approach 1 model results

**Approach 2 Models:**

- `a2_early_expert/`: Early Medieval period expert results
- `a2_middle_expert/`: Middle period expert results
- `a2_late_expert/`: Late Medieval period expert results
- `a2_merged_lora/`: Merged LoRA adapter results
- `a2_merged_full/`: Fully merged Approach 2 model results

**Unified Model:**

- `unified_merged_full/`: Results from the unified model combining both approaches

Each model directory contains JSON files with timestamped results:

- **File naming**: `{model_name}__on__{test_dataset}__YYYYMMDD_HHMMSS.json`
- **Example**: `a1_lexical_expert__on__real_97_test__20251111_184910.json`

**Test datasets evaluated:**

- `real_97_test` - The 97 authentic archaic Italian sentences (all models)
- `a1_lexical_test` - Lexical expert-specific test set
- `a1_syntactic_test` - Syntactic expert-specific test set
- `a1_semantic_test` - Semantic expert-specific test set
- `a2_early_1260_test` - Early Medieval period test set
- `a2_middle_1310_test` - Middle period test set
- `a2_late_1360_test` - Late Medieval period test set

**Each JSON file contains:**

- BLEU, chrF++, ROUGE-L, METEOR, BERTScore, and Token-F1 scores
- Source sentences, model predictions, and reference translations
- Model metadata (name, type, path)
- Dataset information and timestamps

##### LLM Baseline Evaluations (`outputs/gpt4.1_and_gemini_2.0_models_evaluation/`)

Contains evaluation results for commercial LLM baselines:

- **`gpt4_1_translations/`**: GPT-4.1 zero-shot translation results
- **`gpt4_1_translations_fewshot/`**: GPT-4.1 few-shot translation results
- **`gemini2_5_translations/`**: Gemini 2.0 zero-shot translation results
- **`gemini2_5_translations_fewshots/`**: Gemini 2.0 few-shot translation results

**File naming**: `{model_name}__on__{test_dataset}.json`

- **Example**: `gpt4_1__on__real_97_test.json`, `gemini2_5__on__a1_lexical_test.json`

Each directory contains evaluation results on all 7 test datasets:

- `real_97_test` (main evaluation dataset)
- `a1_lexical_test`, `a1_syntactic_test`, `a1_semantic_test`
- `a2_early_1260_test`, `a2_middle_1310_test`, `a2_late_1360_test`

Each JSON file contains translations, reference texts, and quantitative metrics for comparison with fine-tuned models.

##### LLM-as-a-Judge Results (`outputs/llm_as_a_judge_results/`)

Qualitative evaluation results using GPT-4.1 as a judge:

- **`LLM_as_a_judge_finetuned_model_evaluation/`**: Judgments for all fine-tuned models

  - Subdirectories for each model: `a1_lexical_expert/`, `a1_syntactic_expert/`, `a1_semantic_expert/`, etc.
  - Each subdirectory contains JSON files for applicable test datasets
  - Includes scores for Faithfulness, Fluency, Style, and Overall quality (1-5 scale)
  - Provides detailed reasoning and feedback for each evaluation criterion

- **`LLM_as_a_judge_gpt4.1_and_gemini_2.0_models_evaluation/`**: Judgments for baseline LLMs
  - Subdirectories: `gpt4_1/`, `gpt4_1_fewshot/`, `gemini2_5/`, `gemini2_5_fewshot/`
  - Contains comparative assessments of GPT-4.1 and Gemini 2.0 translations
  - Each subdirectory has JSON files for all 7 test datasets
  - Same qualitative scoring criteria as fine-tuned models

#### 3. Documentation and Result Summaries (`documents_and_results_summary/` directory)

Comprehensive documentation of the project and all evaluation results:

- **`nlp-hw-report.pdf`**:

  - **Complete project report** with detailed methodology, experiments, and findings
  - Includes background on archaic Italian translation challenges
  - Explains both multi-expert ensemble approaches in detail
  - Comprehensive analysis of results with tables and visualizations
  - **Recommended starting point** for understanding the entire project

- **`summary_fine_tuned_models.txt`**:

  - Tabular summary of all fine-tuned model performances
  - Shows BLEU, chrF++, ROUGE-L, METEOR, BERTScore, and Token-F1 for each model
  - Organized by model name and test dataset
  - Easy comparison across individual experts, merged models, and unified models

- **`summary_gpt_4_1_and_gemini_models.txt`**:

  - Summary of GPT-4.1 and Gemini 2.0 baseline model performances
  - Comparison between zero-shot and few-shot approaches
  - Performance metrics across all test datasets
  - Analysis of commercial LLM capabilities on archaic Italian translation

- **`summary_fine_tuned_models_of_llm_as_a_judge.txt`**:
  - Qualitative assessment summary from LLM-as-a-Judge evaluations
  - Aggregated scores for Faithfulness, Fluency, Style, and Overall quality
  - Insights on translation quality from a human-centric perspective
  - Comparative analysis across all models

These files provide an easy-to-read overview of all experimental results without needing to parse JSON files or run evaluation scripts.

### Quick Start for Exploration

**For a comprehensive understanding:**

1. **Read the Full Report**: Start with `documents_and_results_summary/nlp-hw-report.pdf` for complete project details
2. **Review Result Summaries**: Check the `.txt` files in `documents_and_results_summary/` for tabular metric summaries
3. **Explore Raw Data**:
   - `data/synthetic_approach1.jsonl` and `data/synthetic_approach2.jsonl` - Training data examples
   - `data/processed/test_ground_truth.jsonl` - Real test sentences (97 samples)
   - `data/processed/approach1/` and `approach2/` - Expert-specific test sets
4. **Examine Model Outputs**:
   - `outputs/finetuned_model_evaluation/` - Detailed results for all fine-tuned models
   - `outputs/gpt4.1_and_gemini_2.0_models_evaluation/` - Commercial LLM baseline results
   - `outputs/llm_as_a_judge_results/` - Qualitative assessments with human-centric scores
5. **Compare Approaches**:
   - Approach 1 (linguistic experts: lexical, syntactic, semantic)
   - Approach 2 (temporal experts: early, middle, late periods)
   - Unified model (combination of both approaches)
   - Commercial LLMs (GPT-4.1 and Gemini 2.0, zero-shot vs few-shot)

**Understanding the evaluation:**

- Each model was tested on 7 different test sets (real_97 + 6 expert-specific sets)
- Metrics include BLEU, chrF++, ROUGE-L, METEOR, BERTScore, and Token-F1
- LLM-as-a-Judge provides qualitative scores (1-5) for Faithfulness, Fluency, Style, and Overall quality

All results are timestamped and include complete metric breakdowns, making it easy to understand model performance without running any code.

## Acknowledgments

This project was conducted as part of the **Multilingual Natural Language Processing** course in the Master's program in Artificial Intelligence and Robotics at **La Sapienza University of Rome** during the academic year **2024-2025**. The homework assignment was titled "From Ancient to Modern Italian."

Models were fine-tuned on RunPod infrastructure using NVIDIA RTX A4000 GPUs. We acknowledge the use of OpenAI GPT-4.1 and Google Gemini 2.0 for baseline comparisons and synthetic data generation.
