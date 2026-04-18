"""
Uzbek-only chat dataset preparation for Ai_training-model-4.

Pulls instruction/Q&A data from public HuggingFace Uzbek datasets and writes
ChatML JSONL files (chat_train.jsonl + chat_val.jsonl).

Sources (in order of preference):
  - behbudiy/alpaca-cleaned-uz       — translated Alpaca (instruction/output)
  - behbudiy/translation-instruction-uzbek
  - risqaliyevds/uzbek_instruct      — Uzbek instruction set
  - tahrirchi/uz-books               — Uzbek books (used as continuation pretraining-style Q&A)
  - wikipedia (uz)                   — Uzbek Wikipedia (extract title/intro pairs)

Each output line:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
  python scripts/prepare_dataset.py \\
      --output-dir data \\
      --sources alpaca-uz,translation-uz,risqaliyevds,wiki-uz \\
      --max-samples 200000 \\
      --val-ratio 0.01
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Iterator, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_mostly_uzbek(text: str) -> bool:
    """Crude filter: at least 30% of chars are Uzbek-Latin or Cyrillic, very few
    runs of plain English-only words."""
    if not text:
        return False
    # Quick allow: contains Uzbek-specific letters or apostrophes used in Uzbek Latin
    uz_chars = set("o'g'O'G'oʻgʻOʻGʻchshChShъыэЪЫЭ")
    if any(ch in text for ch in uz_chars):
        return True
    # Cyrillic share
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    if cyr / max(1, len(text)) > 0.3:
        return True
    # Latin-only English heuristic: large share of common English words = reject
    eng_markers = (" the ", " and ", " is ", " of ", " for ", " with ", " that ")
    score = sum(text.lower().count(m) for m in eng_markers)
    return score < 2  # tolerate occasional borrowings


def _make_pair(user: str, assistant: str) -> Optional[dict]:
    user = _norm(user)
    assistant = _norm(assistant)
    if not user or not assistant:
        return None
    if len(user) < 2 or len(assistant) < 2:
        return None
    if len(user) > 4000 or len(assistant) > 4000:
        return None
    if not (_is_mostly_uzbek(user) or _is_mostly_uzbek(assistant)):
        return None
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


# ---------------------------------------------------------------------------
# Source loaders (lazy — datasets library imported inside)
# ---------------------------------------------------------------------------

def _load_hf(name: str, split: str = "train", config: Optional[str] = None,
             streaming: bool = True):
    from datasets import load_dataset
    return load_dataset(name, config, split=split, streaming=streaming,
                        trust_remote_code=True)


def src_alpaca_uz(limit: int) -> Iterator[dict]:
    """behbudiy/alpaca-cleaned-uz — Uzbek translation of Alpaca."""
    print("[SRC] behbudiy/alpaca-cleaned-uz ...")
    try:
        ds = _load_hf("behbudiy/alpaca-cleaned-uz")
    except Exception as e:
        print(f"  [SKIP] {e}")
        return
    n = 0
    for ex in ds:
        instr = ex.get("instruction") or ex.get("instruction_uz") or ""
        inp = ex.get("input") or ex.get("input_uz") or ""
        out = ex.get("output") or ex.get("output_uz") or ""
        user = (instr + ("\n\n" + inp if inp else "")).strip()
        pair = _make_pair(user, out)
        if pair:
            yield pair
            n += 1
            if n >= limit:
                break
    print(f"  [SRC] alpaca-uz -> {n}")


def src_translation_uz(limit: int) -> Iterator[dict]:
    """behbudiy/translation-instruction-uzbek."""
    print("[SRC] behbudiy/translation-instruction-uzbek ...")
    try:
        ds = _load_hf("behbudiy/translation-instruction-uzbek")
    except Exception as e:
        print(f"  [SKIP] {e}")
        return
    n = 0
    for ex in ds:
        instr = ex.get("instruction") or ex.get("prompt") or ""
        out = ex.get("output") or ex.get("response") or ex.get("completion") or ""
        pair = _make_pair(instr, out)
        if pair:
            yield pair
            n += 1
            if n >= limit:
                break
    print(f"  [SRC] translation-uz -> {n}")


def src_risqaliyevds(limit: int) -> Iterator[dict]:
    """risqaliyevds/uzbek_instruct or similar Uzbek instruct datasets."""
    candidates = [
        "risqaliyevds/uzbek_instruct",
        "risqaliyevds/Uzbek-Instruct",
        "risqaliyevds/uzbek_instruct_data",
    ]
    for name in candidates:
        print(f"[SRC] {name} ...")
        try:
            ds = _load_hf(name)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue
        n = 0
        for ex in ds:
            instr = ex.get("instruction") or ex.get("question") or ex.get("prompt") or ""
            inp = ex.get("input") or ""
            out = ex.get("output") or ex.get("answer") or ex.get("response") or ""
            user = (instr + ("\n\n" + inp if inp else "")).strip()
            pair = _make_pair(user, out)
            if pair:
                yield pair
                n += 1
                if n >= limit:
                    break
        print(f"  [SRC] {name} -> {n}")
        if n > 0:
            return  # success, stop trying alternatives


def src_wiki_uz(limit: int) -> Iterator[dict]:
    """Uzbek Wikipedia → Q&A pair (title -> intro paragraph)."""
    print("[SRC] wikipedia (uz) ...")
    name_attempts = [
        ("wikimedia/wikipedia", "20231101.uz"),
        ("wikipedia", "20220301.uz"),
    ]
    ds = None
    for name, conf in name_attempts:
        try:
            ds = _load_hf(name, config=conf)
            break
        except Exception as e:
            print(f"  [TRY] {name}/{conf}: {e}")
    if ds is None:
        print("  [SKIP] no wikipedia uz available")
        return
    n = 0
    templates = [
        "{t} haqida nima bilasan?",
        "{t} nima?",
        "{t} kim?",
        "{t} ni qisqacha tushuntirib ber.",
        "Menga {t} haqida ma'lumot ber.",
    ]
    for ex in ds:
        title = (ex.get("title") or "").strip()
        text = (ex.get("text") or "").strip()
        if not title or not text or len(text) < 80:
            continue
        intro = text.split("\n\n", 1)[0]
        intro = " ".join(intro.split())[:1200]
        prompt = random.choice(templates).format(t=title)
        pair = _make_pair(prompt, intro)
        if pair:
            yield pair
            n += 1
            if n >= limit:
                break
    print(f"  [SRC] wiki-uz -> {n}")


def src_uz_books(limit: int) -> Iterator[dict]:
    """tahrirchi/uz-books — chunk into short prompt/continuation pairs."""
    print("[SRC] tahrirchi/uz-books ...")
    try:
        ds = _load_hf("tahrirchi/uz-books")
    except Exception as e:
        print(f"  [SKIP] {e}")
        return
    n = 0
    for ex in ds:
        text = (ex.get("text") or ex.get("content") or "").strip()
        if len(text) < 200:
            continue
        # take 2 paragraph windows: first paragraph as "prompt", second as "answer"
        parts = [p for p in text.split("\n\n") if len(p) > 60]
        for i in range(0, len(parts) - 1, 2):
            prompt = "Davom ettir: " + parts[i].strip()[:600]
            answer = parts[i + 1].strip()[:1200]
            pair = _make_pair(prompt, answer)
            if pair:
                yield pair
                n += 1
                if n >= limit:
                    break
        if n >= limit:
            break
    print(f"  [SRC] uz-books -> {n}")


SOURCES = {
    "alpaca-uz":      src_alpaca_uz,
    "translation-uz": src_translation_uz,
    "risqaliyevds":   src_risqaliyevds,
    "wiki-uz":        src_wiki_uz,
    "uz-books":       src_uz_books,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data")
    p.add_argument("--sources",
                   default="alpaca-uz,translation-uz,risqaliyevds,wiki-uz",
                   help="comma separated, choose from " + ",".join(SOURCES))
    p.add_argument("--per-source-limit", type=int, default=80000,
                   help="hard cap per source")
    p.add_argument("--max-samples", type=int, default=300000,
                   help="overall cap (after dedup)")
    p.add_argument("--val-ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SOURCES]
    if unknown:
        print(f"[ERR] unknown sources: {unknown}. Allowed: {list(SOURCES)}")
        return 2

    seen_keys = set()
    rows: List[dict] = []
    for name in requested:
        loader = SOURCES[name]
        for pair in loader(args.per_source_limit):
            key = pair["messages"][0]["content"][:200] + "|" + \
                  pair["messages"][1]["content"][:200]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(pair)
            if len(rows) >= args.max_samples:
                break
        if len(rows) >= args.max_samples:
            break

    if not rows:
        print("[ERR] no samples collected")
        return 1

    random.shuffle(rows)
    n_val = max(200, int(len(rows) * args.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    train_path = os.path.join(args.output_dir, "chat_train.jsonl")
    val_path = os.path.join(args.output_dir, "chat_val.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[OK] train = {len(train_rows):,}  -> {train_path}")
    print(f"[OK] val   = {len(val_rows):,}  -> {val_path}")
    print(f"[OK] total = {len(rows):,}  (sources: {requested})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
