import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are a careful teaching assistant for MIT 6.034 Artificial Intelligence. "
    "Answer only when the question is covered by the course material you learned. "
    "Give a direct, factual explanation. If the material does not support an "
    "answer, say that it is outside the course or that you are uncertain."
)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9*()^+\-/]+", text.lower())


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = tokens(prediction)
    expected = tokens(reference)
    if not predicted or not expected:
        return 0.0
    previous = [0] * (len(expected) + 1)
    for predicted_token in predicted:
        current = [0]
        for index, expected_token in enumerate(expected, start=1):
            if predicted_token == expected_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def keyword_recall(prediction: str, required_terms: list[str]) -> float:
    if not required_terms:
        return 1.0
    normalized = prediction.lower()
    matched = sum(term.lower() in normalized for term in required_terms)
    return matched / len(required_terms)


def load_questions(path: Path, limit: int) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit > 0 else rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate factual course answers.")
    parser.add_argument("--model", choices=["base", "adapter"], required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", default="outputs/qwen-course-lora-v2")
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("data/evaluation/questions.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run python check_gpu.py first.")
    if not args.questions_file.exists():
        raise SystemExit(f"Evaluation questions not found: {args.questions_file}")
    if args.model == "adapter" and not Path(args.adapter_dir).exists():
        raise SystemExit("Adapter not found. Train the v2 adapter first.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    if args.model == "adapter":
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    results = []
    for index, row in enumerate(load_questions(args.questions_file, args.limit), start=1):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
            )
        prediction = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        rouge = rouge_l_f1(prediction, row["reference_answer"])
        keyword = keyword_recall(prediction, row.get("required_terms", []))
        results.append(
            {
                **row,
                "prediction": prediction,
                "rouge_l_f1": round(rouge, 4),
                "required_term_recall": round(keyword, 4),
                "needs_human_review": True,
                "human_factuality_0_to_2": None,
            }
        )
        print(
            f"[{index:02}] terms={keyword:.2f} rougeL={rouge:.2f} | "
            f"{row['question'][:64]}"
        )

    average_rouge = sum(row["rouge_l_f1"] for row in results) / len(results)
    average_terms = sum(row["required_term_recall"] for row in results) / len(results)
    output_path = Path(f"outputs/evaluation_v2_{args.model}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "average_rouge_l_f1": average_rouge,
                "average_required_term_recall": average_terms,
                "human_review_complete": False,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nAverage required-term recall: {average_terms:.3f}")
    print(f"Average ROUGE-L F1: {average_rouge:.3f}")
    print(f"Review every answer in: {output_path.resolve()}")
    print("Do not promote an adapter until human factuality review is complete.")


if __name__ == "__main__":
    main()
