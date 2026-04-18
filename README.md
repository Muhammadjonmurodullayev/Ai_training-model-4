# Ai_training-model-4 — O'zbekcha chat AI

**MiniTransformer** asosida **faqat o'zbek tilida** ishlovchi chat-tuned model trainingi.

- **Model:** MiniTransformer (RoPE + SwiGLU + Pre-LN + DeepNorm scaling)
- **Til:** O'zbek (Latin + Cyrillic)
- **Vocab:** 16 000 tokens (SentencePiece BPE)
- **Default config:** `embed_dim=512`, `num_heads=8`, `num_layers=8`, `ff_dim=2048`, `seq_len=512` → ~30M params
- **Training:** ~4-5 soat T4/L4 GPU da
- **Format:** ChatML (`<|im_start|>user … <|im_end|> <|im_start|>assistant … <|im_end|>`)

## Datasetlar (HuggingFace)

| Manba | Tavsif | Hajm |
|---|---|---|
| `behbudiy/alpaca-cleaned-uz` | Tarjima qilingan Alpaca instruktsiyalari | ~50k |
| `behbudiy/translation-instruction-uzbek` | Tarjima va instruktsiya | ~80k |
| `risqaliyevds/uzbek_instruct` | Mahalliy o'zbekcha instruktsiyalar | ~30k |
| `wikipedia (uz)` | Wikipedia ga asoslangan QA | ~100k |
| `tahrirchi/uz-books` | Davom ettirish promptlari (ixtiyoriy) | ~50k |

## Tezkor boshlash (Google Colab)

```python
# 1) Drive mount
from google.colab import drive
drive.mount('/content/drive')

# 2) Repo
!git clone https://github.com/Muhammadjonmurodullayev/Ai_training-model-4.git
%cd Ai_training-model-4
!pip install -q -r requirements.txt

# 3) Dataset
!python scripts/prepare_dataset.py \
  --output-dir data \
  --sources alpaca-uz,translation-uz,risqaliyevds,wiki-uz \
  --max-samples 250000 --val-ratio 0.01

# 4) Tokenizer
!python scripts/train_tokenizer.py \
  --train data/chat_train.jsonl --val data/chat_val.jsonl \
  --output data/chat_vocab --vocab-size 16000

# 5) Training (4-5 soat) — natija to'g'ridan-to'g'ri Drive ga
DRIVE = "/content/drive/MyDrive/Ai_chat_checkpoints_v4"
import os; os.makedirs(DRIVE, exist_ok=True)

!python scripts/train_chat.py \
  --train data/chat_train.jsonl --val data/chat_val.jsonl \
  --tokenizer data/chat_vocab.model \
  --output-dir "$DRIVE" \
  --embed-dim 512 --num-heads 8 --num-layers 8 --ff-dim 2048 \
  --max-seq-len 512 --batch-size 16 --grad-accum 4 \
  --epochs 6 --lr 3e-4 --warmup-steps 500 --log-every 50 --eval-every 1000
```

## Lokal test

```bash
python scripts/prepare_dataset.py --output-dir data --sources alpaca-uz --per-source-limit 200
python scripts/train_tokenizer.py --train data/chat_train.jsonl --val data/chat_val.jsonl --output data/chat_vocab --vocab-size 2000
python scripts/train_chat.py --train data/chat_train.jsonl --val data/chat_val.jsonl --tokenizer data/chat_vocab.model --output-dir output --embed-dim 128 --num-heads 4 --num-layers 2 --epochs 1 --batch-size 4
```

## Output

Training tugagach `--output-dir` ichida:
- `chat_best.pt` — eng yaxshi val_loss li model (~120 MB)
- `chat_last.pt` — oxirgi epoch (~120 MB)
- `chat_vocab.model` — tokenizer (qo'lda ko'chirish kerak)

`Ai_chat` asosiy loyihaning `backend/checkpoints/chat/` papkasiga ko'chiring va admin paneldan reload qiling.
