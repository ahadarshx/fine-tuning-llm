import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from prepare_data import (
    OUT_OF_SCOPE_EXAMPLES,
    SUPPORTED_EXTENSIONS,
    assert_dataset_quality,
    clean_text,
    markdown_sections,
    normalize_title,
    paragraph_chunks,
    read_document,
    write_jsonl,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a teaching assistant for the MIT 6.034 Artificial Intelligence course. You write study questions and answers strictly from the lecture note section you are given. Never use outside knowledge and never invent facts that are not supported by the section."
)

USER_PROMPT = """\
Below is one section of MIT 6.034 course notes.

Topic: {topic}

--- SECTION START ---
{body}
--- SECTION END ---

Write {count} distinct question-answer pairs that a student might ask about \
this section. Make the questions varied in form and intent, for example:
- a definition question ("What is ...?")
- a motivation question ("Why is ... useful?" / "What problem does ... solve?")
- a mechanism question ("How does ... work?" / "How does ... reduce search?")
- a comparison or trade-off question, when the section supports one
- a worked or example-based question, when the section gives an example

Rules:
- Phrase questions the natural way a student would, not by quoting the heading.
- Each answer must be self-contained, 2-5 sentences, and grounded ONLY in the \
section text above. Do not reference "the section" or "the notes".
- Do not repeat the same question or the same answer.
- If the section is too thin to support {count} good pairs, return fewer.

Return ONLY a JSON array, no prose, no markdown fences:
[{{"question": "...", "answer": "..."}}, ...]
"""


class LocalLLM:
    """A locally-downloaded chat model run with transformers.

    The model is fetched from the Hugging Face hub on first use and cached under
    ~/.cache/huggingface, so subsequent runs are fully offline. On CUDA it can
    load in 4-bit (bitsandbytes) to fit a laptop GPU; otherwise it falls back to
    fp16 on GPU or fp32 on CPU.
    """

    def __init__(self, model_id: str, load_4bit: bool, device: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SystemExit(
                "generate_data.py needs transformers + torch. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        resolved = device
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA GPU is available.")
        self.device = resolved

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {"low_cpu_mem_usage": True}
        if resolved == "cuda":
            load_kwargs["device_map"] = {"": 0}
            load_kwargs["torch_dtype"] = torch.float16
            if load_4bit:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
        else:
            load_kwargs["torch_dtype"] = torch.float32

        print(f"Loading {model_id} on {resolved}{' (4-bit)' if load_4bit and resolved == 'cuda' else ''} ...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if "device_map" not in load_kwargs:
            self.model.to(resolved)
        self.model.eval()

    def generate(self, messages: list[dict], temperature: float, max_new_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.model.device)
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()


def call_llm(
    llm: LocalLLM,
    topic: str,
    body: str,
    count: int,
    temperature: float,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(topic=topic, body=body, count=count)},
    ]
    return llm.generate(messages, temperature, max_new_tokens)


def extract_pairs(raw: str) -> list[dict]:
    """Pull a list of {question, answer} dicts out of a model response."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    pairs: list[dict] = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if len(question) >= 8 and len(answer.split()) >= 8:
            pairs.append({"question": question, "answer": answer})
    return pairs


class GenerationCache:
    """On-disk cache of raw LLM responses keyed by model + section + count."""

    def __init__(self, path: Path):
        self.path = path
        self.store: dict[str, str] = {}
        if path.exists():
            try:
                self.store = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.store = {}

    @staticmethod
    def key(model: str, topic: str, body: str, count: int) -> str:
        payload = f"{model}\x1f{count}\x1f{topic}\x1f{body}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str) -> None:
        self.store[key] = value

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.store, ensure_ascii=False), encoding="utf-8")


def section_pairs(
    llm: LocalLLM,
    cache: GenerationCache,
    model: str,
    topic: str,
    body: str,
    count: int,
    temperature: float,
    max_new_tokens: int,
) -> list[dict]:
    key = cache.key(model, topic, body, count)
    cached = cache.get(key)
    if cached is not None:
        return extract_pairs(cached)
    raw = call_llm(llm, topic, body, count, temperature, max_new_tokens)
    cache.set(key, raw)
    return extract_pairs(raw)


def make_examples(
    llm: LocalLLM,
    cache: GenerationCache,
    model: str,
    path: Path,
    text: str,
    max_words: int,
    questions_per_chunk: int,
    temperature: float,
    max_new_tokens: int,
    seen_responses: set[str],
) -> list[dict]:
    fallback = normalize_title(path.stem.replace("_", " "))
    sections = (
        markdown_sections(text, fallback)
        if path.suffix.lower() == ".md"
        else [(fallback, text)]
    )
    examples: list[dict] = []
    for topic, body in sections:
        for chunk in paragraph_chunks(body, max_words):
            pairs = section_pairs(
                llm, cache, model, topic, chunk,
                questions_per_chunk, temperature, max_new_tokens,
            )
            for pair in pairs:
                answer = pair["answer"]
                fingerprint = answer.strip().lower()
                if fingerprint in seen_responses:
                    continue  # keep responses globally unique for the audit
                seen_responses.add(fingerprint)
                examples.append(
                    {
                        "id": hashlib.sha1(
                            f"{path.name}:{topic}:{pair['question']}:{answer}".encode("utf-8")
                        ).hexdigest()[:12],
                        "instruction": pair["question"],
                        "response": answer,
                        "source": path.name,
                        "topic": topic,
                        "kind": "course",
                    }
                )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a natural Q&A course dataset with a local Hugging Face LLM."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Hugging Face model id to download and run locally.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="Where to run the model (auto: CUDA if available, else CPU).",
    )
    parser.add_argument(
        "--load-4bit", action="store_true",
        help="Load in 4-bit via bitsandbytes (CUDA only) to fit a small GPU.",
    )
    parser.add_argument("--max-words", type=int, default=180, help="Max words per section chunk.")
    parser.add_argument(
        "--questions-per-chunk", type=int, default=2,
        help="Question-answer pairs requested per section chunk.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--cache", type=Path, default=None,
        help="LLM response cache path (default: <output-dir>/.llm_cache.json).",
    )
    parser.add_argument(
        "--limit-docs", type=int, default=0,
        help="Process only the first N documents (for a quick smoke test).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    documents = sorted(
        path
        for path in args.raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if args.limit_docs:
        documents = documents[: args.limit_docs]
    if len(documents) < 3:
        raise SystemExit("Use at least three course documents for source-level splits.")

    llm = LocalLLM(args.model, args.load_4bit, args.device)
    cache = GenerationCache(args.cache or args.output_dir / ".llm_cache.json")
    seen_responses: set[str] = set()

    examples_by_document: dict[str, list[dict]] = {}
    try:
        for path in documents:
            text = clean_text(read_document(path))
            if len(text.split()) < 50:
                print(f"Skipped {path.name}: too little usable text.")
                continue
            rows = make_examples(
                llm, cache, args.model, path, text,
                args.max_words, args.questions_per_chunk,
                args.temperature, args.max_new_tokens, seen_responses,
            )
            examples_by_document[path.name] = rows
            print(f"{path.name:42} {len(rows):4} Q&A pairs")
    finally:
        cache.flush()  # persist whatever we generated, even on interruption

    total = sum(len(rows) for rows in examples_by_document.values())
    if total == 0:
        raise SystemExit(
            "The LLM produced no usable pairs. Try a larger --model or raise "
            "--max-new-tokens, then rerun (cached sections will be skipped)."
        )

    document_names = list(examples_by_document)
    random.Random(args.seed).shuffle(document_names)
    validation_count = max(1, round(len(document_names) * 0.12))
    test_count = max(1, round(len(document_names) * 0.12))
    validation_docs = set(document_names[:validation_count])
    test_docs = set(document_names[validation_count : validation_count + test_count])

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for source, examples in examples_by_document.items():
        split = (
            "validation"
            if source in validation_docs
            else "test"
            if source in test_docs
            else "train"
        )
        splits[split].extend(examples)

    for index, (instruction, response) in enumerate(OUT_OF_SCOPE_EXAMPLES):
        splits["train"].append(
            {
                "id": f"scope-{index + 1}",
                "instruction": instruction,
                "response": response,
                "source": "scope_examples",
                "topic": "Out-of-scope handling",
                "kind": "abstention",
            }
        )

    assert_dataset_quality(splits)
    for split, rows in splits.items():
        random.Random(args.seed).shuffle(rows)
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)

    manifest = {
        "generator": "generate_data.py",
        "model": args.model,
        "seed": args.seed,
        "max_words": args.max_words,
        "questions_per_chunk": args.questions_per_chunk,
        "temperature": args.temperature,
        "documents": {
            "train": sorted(set(document_names) - validation_docs - test_docs),
            "validation": sorted(validation_docs),
            "test": sorted(test_docs),
        },
        "examples": {name: len(rows) for name, rows in splits.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n===== DATASET SUMMARY =====")
    print(f"Model:     {args.model}")
    print(f"Documents: {len(document_names)}")
    for split in ("train", "validation", "test"):
        print(
            f"{split.title():10} {len(manifest['documents'][split]):2} documents, "
            f"{len(splits[split]):3} examples"
        )
    print(f"Saved to:  {args.output_dir.resolve()}")
    print("Next: inspect JSONL files, then run python audit_data.py")


if __name__ == "__main__":
    main()
