import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prepared course data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    split_rows = {
        split: load_jsonl(args.data_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    seen: dict[str, str] = {}
    failures: list[str] = []

    for split, rows in split_rows.items():
        sources = Counter(row["source"] for row in rows)
        kinds = Counter(row.get("kind", "unknown") for row in rows)
        response_counts = Counter(row["response"].strip() for row in rows)
        duplicate_count = sum(count - 1 for count in response_counts.values() if count > 1)
        print(
            f"{split:10} rows={len(rows):3} unique_responses={len(response_counts):3} "
            f"duplicates={duplicate_count:2} sources={len(sources):2} kinds={dict(kinds)}"
        )
        if duplicate_count:
            failures.append(f"{split} contains {duplicate_count} duplicate responses")

        for row in rows:
            digest = hashlib.sha256(row["response"].strip().encode("utf-8")).hexdigest()
            previous = seen.get(digest)
            if previous and previous != split:
                failures.append(f"response leakage between {previous} and {split}")
            seen[digest] = split

    source_sets = {
        split: {row["source"] for row in rows if row["source"] != "scope_examples"}
        for split, rows in split_rows.items()
    }
    if source_sets["train"] & source_sets["validation"]:
        failures.append("source leakage between train and validation")
    if source_sets["train"] & source_sets["test"]:
        failures.append("source leakage between train and test")
    if source_sets["validation"] & source_sets["test"]:
        failures.append("source leakage between validation and test")

    if failures:
        print("\nAUDIT FAILED")
        for failure in sorted(set(failures)):
            print(f"- {failure}")
        raise SystemExit(1)
    print("\nAUDIT PASSED: no duplicate responses or cross-split source leakage.")


if __name__ == "__main__":
    main()
