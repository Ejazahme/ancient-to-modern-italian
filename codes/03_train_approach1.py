import os
import json
import gc
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

DATA_BASE = Path("data/processed/approach1")
OUTPUT_BASE = Path("models/approach1")
LOGS_BASE = Path("logs/training")

TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 6,   
    "max_seq_length": 256,              
    "learning_rate": 2e-4,
    "num_train_epochs": 5,             
    "warmup_steps": 20,
    "fp16": True,
    "optim": "adamw_torch",
}

SCHEDULING_CONFIG = {
    "eval_steps": 100,
    "save_steps": 300,
    "logging_steps": 10,
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

EXPERTS = [
    {
        "name": "lexical_expert",
        "data_subdir": "lexical",
        "description": "Lexical-Morphological Expert",
    },
    {
        "name": "syntactic_expert",
        "data_subdir": "syntactic",
        "description": "Syntactic-Structural Expert",
    },
    {
        "name": "semantic_expert",
        "data_subdir": "semantic",
        "description": "Semantic-Pragmatic Expert",
    },
]

def load_jsonl_dataset(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")
    return load_dataset("json", data_files=str(file_path))["train"]


def format_prompt(example):
    return f"""### Istruzione:
Modernizza la seguente frase italiana arcaica in italiano contemporaneo.

### Frase arcaica:
{example['source']}

### Frase moderna:
{example['target']}"""


def preprocess_function(examples, tokenizer, max_length: int):
    texts = [
        format_prompt({"source": s, "target": t})
        for s, t in zip(examples["source"], examples["target"])
    ]
    enc = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model):
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def build_training_arguments(expert_name: str, output_dir: Path):
    sig = inspect.signature(TrainingArguments)
    valid_params = set(sig.parameters.keys())

    kwargs = {"output_dir": str(output_dir)}
    if "logging_dir" in valid_params:
        kwargs["logging_dir"] = str(LOGS_BASE / expert_name)

    for k, v in TRAINING_CONFIG.items():
        if k in valid_params:
            kwargs[k] = v

    try:
        from transformers.trainer_utils import IntervalStrategy
        eval_strategy = IntervalStrategy.STEPS
        save_strategy = IntervalStrategy.STEPS
    except Exception:
        eval_strategy = "steps"
        save_strategy = "steps"

    if "evaluation_strategy" in valid_params:
        kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in valid_params:
        kwargs["eval_strategy"] = eval_strategy

    if "save_strategy" in valid_params:
        kwargs["save_strategy"] = save_strategy
    if "report_to" in valid_params:
        kwargs["report_to"] = "none"
    if "load_best_model_at_end" in valid_params:
        kwargs["load_best_model_at_end"] = False
    if "overwrite_output_dir" in valid_params:
        kwargs["overwrite_output_dir"] = True
    if "lr_scheduler_type" in valid_params:
        kwargs["lr_scheduler_type"] = "linear"
    for k, v in SCHEDULING_CONFIG.items():
        if k in valid_params:
            kwargs[k] = v
    if "gradient_checkpointing" in valid_params:
        kwargs["gradient_checkpointing"] = True

    return TrainingArguments(**kwargs)

def train_expert(expert_config):
    data_dir = DATA_BASE / expert_config["data_subdir"]
    out_dir = OUTPUT_BASE / expert_config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_BASE.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    model = setup_lora(model)

    train_dataset = load_jsonl_dataset(data_dir / "train.jsonl")
    val_dataset = load_jsonl_dataset(data_dir / "val.jsonl")

    max_len = TRAINING_CONFIG["max_seq_length"]
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_len),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_len),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    args = build_training_arguments(expert_config["name"], out_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(out_dir)

    info = {
        "expert": expert_config["name"],
        "base_model": BASE_MODEL,
        "lora_config": LORA_CONFIG,
        "training_config": TRAINING_CONFIG,
        "scheduling_config": SCHEDULING_CONFIG,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    with open(out_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    for exp in EXPERTS:
        try:
            train_expert(exp)
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue


if __name__ == "__main__":
    main()
