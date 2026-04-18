"""
Chat tokenizer wrapper.

Loads a SentencePiece model trained on uz+en+code corpus.
Provides encode/decode + special token IDs (im_start, im_end, etc.)

If sentencepiece is not installed or no model file exists,
falls back to a minimal multilingual char tokenizer covering
ASCII + Latin Extended + Cyrillic + Uzbek diacritics.
"""

import os
from typing import List, Optional

# Try sentencepiece import
try:
    import sentencepiece as spm
    SP_AVAILABLE = True
except ImportError:
    SP_AVAILABLE = False
    spm = None

from .chat_template import SPECIAL_TOKENS


class ChatTokenizer:
    """
    Wrapper around SentencePiece for chat.

    Special tokens:
        <pad>, <unk>, <s>, </s>,
        <|im_start|>, <|im_end|>, <|endoftext|>
    """

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<s>"
    EOS = "</s>"
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    EOT = "<|endoftext|>"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.sp: Optional["spm.SentencePieceProcessor"] = None
        self._fallback_vocab: Optional[List[str]] = None

        if model_path and os.path.exists(model_path) and SP_AVAILABLE:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_path)
            self.mode = "sentencepiece"
        else:
            # Build minimal multilingual fallback vocab
            self._build_fallback()
            self.mode = "fallback-char"

    # ─── Vocab properties ────────────────────────────

    @property
    def vocab_size(self) -> int:
        if self.sp is not None:
            return self.sp.GetPieceSize()
        return len(self._fallback_vocab) if self._fallback_vocab else 0

    @property
    def pad_id(self) -> int:
        return self._token_id(self.PAD, default=0)

    @property
    def unk_id(self) -> int:
        return self._token_id(self.UNK, default=1)

    @property
    def bos_id(self) -> int:
        return self._token_id(self.BOS, default=2)

    @property
    def eos_id(self) -> int:
        return self._token_id(self.EOS, default=3)

    @property
    def im_start_id(self) -> int:
        return self._token_id(self.IM_START, default=4)

    @property
    def im_end_id(self) -> int:
        return self._token_id(self.IM_END, default=5)

    @property
    def eot_id(self) -> int:
        return self._token_id(self.EOT, default=6)

    def _token_id(self, tok: str, default: int) -> int:
        if self.sp is not None:
            tid = self.sp.PieceToId(tok)
            return tid if tid >= 0 else default
        if self._fallback_vocab and tok in self._fallback_vocab:
            return self._fallback_vocab.index(tok)
        return default

    # ─── Encode / Decode ─────────────────────────────

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if self.sp is not None:
            ids = self.sp.EncodeAsIds(text)
            if add_special_tokens:
                ids = [self.bos_id] + ids + [self.eos_id]
            return ids
        # fallback
        return self._fallback_encode(text, add_special_tokens)

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if self.sp is not None:
            if skip_special:
                special_ids = {
                    self.pad_id, self.bos_id, self.eos_id,
                    self.im_start_id, self.im_end_id, self.eot_id,
                }
                ids = [i for i in ids if i not in special_ids]
            try:
                return self.sp.DecodeIds(list(ids))
            except Exception:
                return "".join(self.sp.IdToPiece(i) for i in ids if i >= 0)
        return self._fallback_decode(ids, skip_special)

    # ─── Fallback char tokenizer (used until SP model is trained) ─

    def _build_fallback(self) -> None:
        """Multilingual char vocab — ASCII + Latin extended + Cyrillic + Uz diacritics."""
        chars: List[str] = []
        # Special tokens first (fixed positions)
        chars.extend(SPECIAL_TOKENS)
        # ASCII printable
        for c in range(32, 127):
            chars.append(chr(c))
        # Newline, tab
        chars.extend(["\n", "\t"])
        # Latin Extended-A (covers Uzbek Latin: o', g', etc.)
        for c in range(0x00A0, 0x017F):
            chars.append(chr(c))
        # Special Uzbek chars
        chars.extend(["ʻ", "ʼ", "ʹ", "ʽ", "‘", "’", "ў", "ғ", "қ", "ҳ"])
        # Cyrillic basic
        for c in range(0x0400, 0x0500):
            chars.append(chr(c))
        # Common punctuation
        chars.extend(["…", "—", "–", "«", "»", "“", "”"])
        # Deduplicate preserving order
        seen = set()
        out = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                out.append(c)
        self._fallback_vocab = out

    def _fallback_encode(self, text: str, add_special_tokens: bool) -> List[int]:
        assert self._fallback_vocab is not None
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)
        i = 0
        # Greedy match for special multi-char tokens
        special_sorted = sorted(
            [t for t in SPECIAL_TOKENS if t not in ("<pad>", "<unk>")],
            key=len, reverse=True,
        )
        while i < len(text):
            matched = False
            for sp in special_sorted:
                if text.startswith(sp, i):
                    ids.append(self._fallback_vocab.index(sp))
                    i += len(sp)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            if ch in self._fallback_vocab:
                ids.append(self._fallback_vocab.index(ch))
            else:
                ids.append(self.unk_id)
            i += 1
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def _fallback_decode(self, ids: List[int], skip_special: bool) -> str:
        assert self._fallback_vocab is not None
        special_set = set(SPECIAL_TOKENS) if skip_special else set()
        out: List[str] = []
        for i in ids:
            if 0 <= i < len(self._fallback_vocab):
                tok = self._fallback_vocab[i]
                if tok in special_set:
                    continue
                out.append(tok)
        return "".join(out)

    # ─── Info ────────────────────────────────────────

    def info(self) -> dict:
        return {
            "mode": self.mode,
            "vocab_size": self.vocab_size,
            "model_path": self.model_path,
            "sp_available": SP_AVAILABLE,
            "special_token_ids": {
                "pad": self.pad_id,
                "bos": self.bos_id,
                "eos": self.eos_id,
                "im_start": self.im_start_id,
                "im_end": self.im_end_id,
                "eot": self.eot_id,
            },
        }
