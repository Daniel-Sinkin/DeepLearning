import random
from pathlib import Path
from collections import Counter
from itertools import islice

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

from src.transformer import Transformer
from src.common import get_default_configs

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BOS, EOS, PAD = "<bos>", "<eos>", "<pad>"
MIN_FREQ = 1  # we want all tokens to be available to overfit
MAX_SENT_LEN = 25
TRAIN_SAMPLES = 100  # SMALL for overfitting
BATCH_SIZE = 10
EPOCHS = 300  # more epochs to guarantee overfitting


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def clean_pair(example):
    de, en = example["translation"]["de"], example["translation"]["en"]
    if len(de.split()) <= MAX_SENT_LEN and len(en.split()) <= MAX_SENT_LEN:
        return {"de": de.lower(), "en": en.lower()}


print("Loading OPUS-Books…")
raw = load_dataset("opus_books", "de-en", split="train").shuffle(seed=SEED)
pairs = [clean_pair(ex) for ex in islice(raw, 500)]
pairs = [p for p in pairs if p is not None][:TRAIN_SAMPLES]


def tok(text: str) -> list[str]:
    return text.strip().split()


def build_vocab(sentences, min_freq):
    counter = Counter(token for s in sentences for token in tok(s))
    return {
        PAD: 0,
        BOS: 1,
        EOS: 2,
        **{w: i + 3 for i, (w, c) in enumerate(counter.items()) if c >= min_freq},
    }


vocab_de = build_vocab([p["de"] for p in pairs], MIN_FREQ)
vocab_en = build_vocab([p["en"] for p in pairs], MIN_FREQ)


def encode(text, vocab):
    tokens = [vocab.get(t, vocab[PAD]) for t in tok(text)]
    return [vocab[BOS]] + tokens + [vocab[EOS]]


class TinyTranslationDS(Dataset):
    def __getitem__(self, idx):
        src = torch.tensor(encode(pairs[idx]["de"], vocab_de))
        tgt = torch.tensor(encode(pairs[idx]["en"], vocab_en))
        return src.clamp(max=len(vocab_de) - 1), tgt.clamp(max=len(vocab_en) - 1)

    def __len__(self):
        return len(pairs)


def collate(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=vocab_de[PAD])
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=vocab_en[PAD])
    return src, tgt, src.eq(vocab_de[PAD]), tgt.eq(vocab_en[PAD])


loader = DataLoader(
    TinyTranslationDS(), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
cfgs = get_default_configs()
model = Transformer(
    d_model=128,
    n_head=8,
    d_ff=256,
    n_layer=3,
    dropout=0.1,
    source_vocab_size=len(vocab_de),
    target_vocab_size=len(vocab_en),
    pad_id=vocab_en[PAD],
    configs=cfgs,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab_en[PAD])

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
print("Training to overfit on small dataset…")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for src, tgt, src_pad, tgt_pad in loader:
        optimizer.zero_grad()
        logits = model(src, tgt[:, :-1], src_pad, tgt_pad[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    if epoch % 20 == 0 or avg_loss < 0.05:
        print(f"Epoch {epoch}: loss = {avg_loss:.4f}")
    if avg_loss < 0.01:
        print("Early stopping: Model has overfit.")
        break


def greedy(src_tensor, src_pad, max_len=MAX_SENT_LEN):
    model.eval()
    tgt = torch.tensor([[vocab_en[BOS]]], device=DEVICE)
    for _ in range(max_len):
        logits = model(src_tensor, tgt, src_pad, tgt.eq(vocab_en[PAD]))
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        tgt = torch.cat([tgt, next_tok], dim=1)
        if next_tok.item() == vocab_en[EOS]:
            break
    return tgt.squeeze().tolist()


def detokenize(ids, vocab):
    inv = {v: k for k, v in vocab.items()}
    return " ".join(
        inv.get(i, f"<unk:{i}>")
        for i in ids
        if i not in (vocab[BOS], vocab[EOS], vocab[PAD])
    )


print("\n--- Greedy Output on Training Set ---")
for i in range(5):
    src_text = pairs[i]["de"]
    tgt_text = pairs[i]["en"]
    src_tensor = torch.tensor(encode(src_text, vocab_de)).unsqueeze(0)
    decoded = greedy(src_tensor, src_tensor.eq(vocab_de[PAD]))
    print(f"\n[{i+1}]")
    print("DE:", src_text)
    print("GT:", tgt_text)
    print("EN:", detokenize(decoded, vocab_en))
