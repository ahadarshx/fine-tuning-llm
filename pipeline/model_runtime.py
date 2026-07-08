import os
import threading
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_ID = os.getenv("ADAPTER_ID", "outputs/qwen-course-lora-v2")
SYSTEM_PROMPT = (
    "You are an MIT 6.034 Artificial Intelligence course assistant.\n\n"
    "Only answer questions that are directly related to MIT 6.034 course topics, "
    "including search, CSPs, game playing, learning, decision trees, Naive Bayes, "
    "neural networks, SVMs, logic, and natural language understanding.\n\n"
    "If a question is outside the MIT 6.034 course material, respond exactly:\n"
    "\"This question is outside the MIT 6.034 course corpus and I cannot answer it reliably.\"\n\n"
    "Do not provide medical, financial, legal, sports, or general-world knowledge advice."
)


class CourseModel:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to serve this model.")
        if ADAPTER_ID.startswith(("outputs/", ".", "/")) and not Path(ADAPTER_ID).exists():
            raise FileNotFoundError(
                f"Adapter not found at {ADAPTER_ID}. Train it or set ADAPTER_ID "
                "to a Hugging Face adapter repository."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        self.model.eval()
        self.generation_lock = threading.Lock()

    def generate(self, message: str, history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in (history or [])[-6:]:
            role, content = item.get("role"), item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content[:4000]})
        messages.append({"role": "user", "content": message[:4000]})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536
        ).to("cuda")
        with self.generation_lock, torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()


_runtime: CourseModel | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> CourseModel:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = CourseModel()
    return _runtime
