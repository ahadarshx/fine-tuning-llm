import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


SYSTEM_PROMPT = (
    "You are a careful teaching assistant for MIT 6.034 Artificial Intelligence. "
    "Answer only when the question is covered by the course material you learned. "
    "Give a direct, factual explanation. If the material does not support an answer, say that it is outside the course or that you are uncertain."
)


class CourseDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.items = []
        for row in rows:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["instruction"]},
            ]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": row["response"]}
            ]
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=True, max_length=max_length
            )["input_ids"]
            encoded = tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=max_length
            )
            labels = encoded["input_ids"].copy()
            prompt_length = min(len(prompt_ids), len(labels))
            labels[:prompt_length] = [-100] * prompt_length
            if all(label == -100 for label in labels):
                continue
            self.items.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": labels,
                }
            )
        if not self.items:
            raise ValueError("No trainable examples remain after tokenization.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


@dataclass
class CausalDataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)
        max_length = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with 4-bit QLoRA.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default="data/processed/train.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen-course-lora-v2")
    parser.add_argument("--validation-data", default="data/processed/validation.jsonl")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run python check_gpu.py first.")
    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} does not exist. Run prepare_data.py first.")
    if not Path(args.validation_data).exists():
        raise SystemExit(
            f"{args.validation_data} does not exist. Run prepare_data.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    rows = load_dataset("json", data_files=args.data, split="train")
    validation_rows = load_dataset(
        "json", data_files=args.validation_data, split="train"
    )
    train_dataset = CourseDataset(rows, tokenizer, args.max_length)
    validation_dataset = CourseDataset(validation_rows, tokenizer, args.max_length)
    print(f"Tokenized training examples: {len(train_dataset)}")
    print(f"Tokenized validation examples: {len(validation_dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_pin_memory=True,
        logging_first_step=True,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalDataCollator(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1,
                early_stopping_threshold=0.01,
            )
        ],
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter saved to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
