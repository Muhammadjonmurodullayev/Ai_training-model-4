"""
Train MiniTransformer (chat) on ChatML JSONL data.

Features:
    * ChatML rendering with assistant-only loss masking (label = -100 on
      system/user tokens, so the model learns only to produce assistant turns).
    * AdamW + linear warmup + cosine decay LR schedule.
    * Mixed-precision training (AMP) on CUDA.
    * Gradient accumulation, gradient clipping.
    * Resume from checkpoint.
    * Validation loss + early-stopping-style "best" checkpoint.

Usage:
    python scripts/train_chat.py \
        --train       data/chat_train.jsonl \
        --val         data/chat_val.jsonl \
        --tokenizer   checkpoints/chat/chat_vocab.model \
        --output-dir  checkpoints/chat \
        --epochs      3 \
        --batch-size  16 \
        --lr          3e-4 \
        --max-seq-len 256 \
        --embed-dim   256 --num-heads 8 --num-layers 5 --ff-dim 768 \
        --warmup-steps 200

Saved checkpoint format (compatible with ChatService._load):
    {
        "model_config":     dataclasses.asdict(TransformerConfig),
        "model_state_dict": ...,
        "epoch":            int,
        "step":             int,
        "val_loss":         float,
        "best_val_loss":    float,
        "tokenizer_path":   str,
    }
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Make `model` package importable when run from repo root
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from model.model import MiniTransformer, TransformerConfig  # noqa: E402
from model.chat_template import (                            # noqa: E402
    SPECIAL_TOKENS, render_chatml, ROLE_USER, ROLE_SYSTEM, ROLE_ASSISTANT,
)
from model.chat_tokenizer import ChatTokenizer               # noqa: E402

IGNORE_INDEX = -100


# ─── Dataset ───────────────────────────────────────────────────────────────

class ChatMLDataset(Dataset):
    """
    Tokenizes ChatML conversations on-the-fly. Builds (input_ids, labels)
    where labels are IGNORE_INDEX for system/user tokens (loss only on
    assistant content + the trailing <|im_end|> for each assistant turn).

    Strategy:
        For each message, we encode "<|im_start|>{role}\n{content}<|im_end|>\n"
        independently; only assistant chunks contribute non-masked labels.
    """

    def __init__(self, path: Path, tokenizer: ChatTokenizer, max_seq_len: int):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[list[dict]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    self.samples.append(msgs)
        if not self.samples:
            raise RuntimeError(f"No samples loaded from {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_turn(self, role: str, content: str, *, is_first: bool) -> list[int]:
        # Match what render_chatml produces: turns joined by "\n"
        prefix = "" if is_first else "\n"
        text = f"{prefix}<|im_start|>{role}\n{content}<|im_end|>"
        return self.tok.encode(text, add_special_tokens=False)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        msgs = self.samples[idx]
        ids: list[int] = []
        labels: list[int] = []
        for i, m in enumerate(msgs):
            role = m["role"]
            content = m["content"]
            tok_ids = self._encode_turn(role, content, is_first=(i == 0))
            ids.extend(tok_ids)
            if role == ROLE_ASSISTANT:
                labels.extend(tok_ids)            # train on full assistant turn
            else:
                labels.extend([IGNORE_INDEX] * len(tok_ids))

        # Truncate from the start to keep most-recent context (for chat) -
        # but for SFT we keep from start to preserve the prompt+answer.
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        # Shift for causal LM: predict next token. We feed input_ids[:-1]
        # and target labels[1:]. Done in collate to allow padding first.
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_collate(pad_id: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        # We need at least 2 tokens to shift
        max_len = max(max_len, 2)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["input_ids"].size(0)
            input_ids[i, :n] = b["input_ids"]
            labels[i, :n] = b["labels"]
        # Causal-LM shift: model sees tokens[:-1], predicts tokens[1:]
        return {
            "input_ids": input_ids[:, :-1].contiguous(),
            "labels":    labels[:, 1:].contiguous(),
        }
    return collate


# ─── LR schedule ───────────────────────────────────────────────────────────

def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ─── Train / eval loops ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(input_ids)
        loss = crit(logits.view(-1, logits.size(-1)), labels.view(-1))
        n = (labels != IGNORE_INDEX).sum().item()
        total_loss += loss.item()
        total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


def save_ckpt(path: Path, *, model: nn.Module, cfg: TransformerConfig,
              epoch: int, step: int, val_loss: float, best_val_loss: float,
              tokenizer_path: str) -> None:
    """Atomic-ish save with fsync so that Drive (FUSE) actually flushes to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save({
            "model_config": dataclasses.asdict(cfg),
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "tokenizer_path": tokenizer_path,
        }, fh)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass  # fsync not supported on some FUSE backends
    os.replace(tmp, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  [SAVED] {path}  ({size_mb:.1f} MB)", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path,
                    help="Path to chat_vocab.model (SentencePiece)")
    ap.add_argument("--output-dir", default=Path("checkpoints/chat"), type=Path)

    # Model arch
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--ff-dim", type=int, default=768)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Optim
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--betas", default="0.9,0.95")

    # IO
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500,
                    help="steps; also eval at end of each epoch")
    ap.add_argument("--save-every", type=int, default=200,
                    help="steps; force-save chat_last.pt to output_dir (Drive-safe)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and not args.no_amp
    print(f"  [ENV] device={device}  amp={use_amp}  torch={torch.__version__}")

    # ─── Tokenizer ──
    if not args.tokenizer.exists():
        sys.exit(f"  [FAIL] tokenizer not found: {args.tokenizer}")
    tok = ChatTokenizer(model_path=str(args.tokenizer))
    print(f"  [TOK] vocab={tok.vocab_size}  mode={tok.mode}")

    # ─── Datasets ──
    print(f"  [DATA] loading {args.train} ...")
    train_ds = ChatMLDataset(args.train, tok, args.max_seq_len)
    print(f"  [DATA] loading {args.val} ...")
    val_ds = ChatMLDataset(args.val, tok, args.max_seq_len)
    print(f"  [DATA] train={len(train_ds):,}  val={len(val_ds):,}")

    collate = make_collate(pad_id=tok.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=max(0, args.num_workers // 2),
        pin_memory=(device == "cuda"),
    )

    # ─── Model ──
    cfg = TransformerConfig(
        vocab_size=tok.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        padding_idx=tok.pad_id,
    )
    model = MiniTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [MODEL] {n_params/1e6:.2f}M params  cfg={cfg}")

    # ─── Optim ──
    betas = tuple(float(x) for x in args.betas.split(","))
    no_decay = {"bias", "LayerNorm.weight", "ln1.weight", "ln2.weight",
                "ln_final.weight"}
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=betas,
    )
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: lr_lambda(s, args.warmup_steps, total_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ─── Resume ──
    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        start_epoch = ck.get("epoch", 0)
        global_step = ck.get("step", 0)
        best_val = ck.get("best_val_loss", float("inf"))
        print(f"  [RESUME] from {args.resume}  epoch={start_epoch} step={global_step}")

    # Resolve to absolute path so we know EXACTLY where checkpoints land.
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "chat_last.pt"
    best_path = args.output_dir / "chat_best.pt"
    is_drive = "/drive/" in str(args.output_dir) or str(args.output_dir).startswith("/content/drive")
    print(f"  [OUT ] dir       = {args.output_dir}", flush=True)
    print(f"  [OUT ] best_path = {best_path}", flush=True)
    print(f"  [OUT ] last_path = {last_path}", flush=True)
    print(f"  [OUT ] on Drive? = {is_drive}", flush=True)
    smoke = args.output_dir / ".write_test"
    try:
        smoke.write_text("ok")
        smoke.unlink()
        print("  [OUT ] write test PASSED", flush=True)
    except Exception as e:
        sys.exit(f"  [FAIL] cannot write to output dir: {e}")

    # ─── Train ──
    model.train()
    t0 = time.time()
    accum_loss, accum_n = 0.0, 0
    for epoch in range(start_epoch, args.epochs):
        for step_in_epoch, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits = model(input_ids)
                loss = crit(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum
            accum_n += 1

            do_step = ((step_in_epoch + 1) % args.grad_accum == 0) or \
                      (step_in_epoch + 1 == len(train_loader))
            if do_step:
                if args.grad_clip:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = accum_loss / max(1, accum_n)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    tps = (global_step * args.batch_size * args.grad_accum
                           * args.max_seq_len) / max(1, elapsed)
                    print(f"  ep {epoch+1}/{args.epochs}  step {global_step}/{total_steps}  "
                          f"loss {avg:.4f}  lr {lr:.2e}  {tps:,.0f} tok/s")
                    accum_loss, accum_n = 0.0, 0

# Frequent "last" checkpoint save (Drive-safe; survives disconnect)
                if args.save_every and global_step % args.save_every == 0:
                    save_ckpt(last_path, model=model, cfg=cfg,
                              epoch=epoch+1, step=global_step,
                              val_loss=float("nan"), best_val_loss=best_val,
                              tokenizer_path=str(args.tokenizer))

                if global_step % args.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  [EVAL ] step {global_step}  val_loss {val_loss:.4f}  "
                              f"ppl {math.exp(min(val_loss, 20)):.2f}", flush=True)
                    if val_loss < best_val:
                        best_val = val_loss
                        save_ckpt(best_path, model=model, cfg=cfg,
                                  epoch=epoch+1, step=global_step,
                                  val_loss=val_loss, best_val_loss=best_val,
                                  tokenizer_path=str(args.tokenizer))
                        print(f"  [SAVE ] best -> {best_path}  (val {val_loss:.4f})", flush=True)
                    # Always update last after eval too
                    save_ckpt(last_path, model=model, cfg=cfg,
                              epoch=epoch+1, step=global_step,
                              val_loss=val_loss, best_val_loss=best_val,
                              tokenizer_path=str(args.tokenizer))

        # End of epoch eval + last
        val_loss = evaluate(model, val_loader, device)
        print(f"  [EPOCH] {epoch+1} done  val_loss {val_loss:.4f}  "
              f"ppl {math.exp(min(val_loss, 20)):.2f}")
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(best_path, model=model, cfg=cfg,
                      epoch=epoch+1, step=global_step,
                      val_loss=val_loss, best_val_loss=best_val,
                      tokenizer_path=str(args.tokenizer))
            print(f"  [SAVE ] best -> {best_path}  (val {val_loss:.4f})")
        save_ckpt(last_path, model=model, cfg=cfg,
                  epoch=epoch+1, step=global_step,
                  val_loss=val_loss, best_val_loss=best_val,
                  tokenizer_path=str(args.tokenizer))
        print(f"  [SAVE ] last -> {last_path}")

    print(f"\n  [DONE] best_val_loss={best_val:.4f}  total_time={(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
