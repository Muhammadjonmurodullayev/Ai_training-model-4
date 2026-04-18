"""
Train SentencePiece BPE tokenizer on chat JSONL data.

Reads ChatML JSONL (one {"messages": [...]} per line), renders each
conversation into a flat ChatML string, writes them to a temp text
corpus, and trains a SentencePiece BPE model with the special tokens
defined in app.chat.chat_template.

Usage:
    python scripts/train_tokenizer.py \
        --train data/chat_train.jsonl \
        --val   data/chat_val.jsonl \
        --output checkpoints/chat/chat_vocab \
        --vocab-size 5000

The output is two files:  chat_vocab.model  +  chat_vocab.vocab
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Make `model` package importable when run from repo root
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from model.chat_template import SPECIAL_TOKENS, render_chatml  # noqa: E402

import sentencepiece as spm  # noqa: E402


def stream_corpus(jsonl_paths: list[Path], corpus_fh) -> int:
    n = 0
    for p in jsonl_paths:
        if not p.exists():
            print(f"  [WARN] missing: {p}", file=sys.stderr)
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = obj.get("messages")
                if not isinstance(msgs, list):
                    continue
                # Render full ChatML (no generation prompt)
                text = render_chatml(msgs, add_generation_prompt=False)
                # SP wants one sentence per line; replace newlines with space variant?
                # Better: keep newlines. SP can handle them.
                corpus_fh.write(text.replace("\n", " ⏎ ") + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--output", required=True, type=Path,
                    help="prefix path; produces <output>.model and <output>.vocab")
    ap.add_argument("--vocab-size", type=int, default=5000)
    ap.add_argument("--character-coverage", type=float, default=1.0)
    ap.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--max-sentence-length", type=int, default=8192)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    paths = [args.train] + ([args.val] if args.val else [])
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        n = stream_corpus(paths, tmp)
    print(f"  [CORPUS] {n:,} conversations -> {tmp_path}")

    # SP cannot use <pad>/<unk>/<s>/</s> as user_defined; they're built-in.
    # Only ChatML markers are user-defined.
    user_defined = [t for t in SPECIAL_TOKENS
                    if t not in ("<pad>", "<unk>", "<s>", "</s>")]

    print(f"  [SP] training {args.model_type} vocab={args.vocab_size} ...")
    spm.SentencePieceTrainer.train(
        input=str(tmp_path),
        model_prefix=str(args.output),
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<s>",   eos_piece="</s>",
        user_defined_symbols=user_defined,
        max_sentence_length=args.max_sentence_length,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,
        normalization_rule_name="nmt_nfkc",
        remove_extra_whitespaces=False,
    )
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.output) + ".model")
    print(f"  [DONE] vocab_size={sp.GetPieceSize()}")
    for t in SPECIAL_TOKENS:
        tid = sp.PieceToId(t)
        print(f"    {t:20s} -> {tid}")


if __name__ == "__main__":
    main()
