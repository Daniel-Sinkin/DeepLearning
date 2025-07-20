"""
Tiny CPU-friendly German→English translation trainer

Dependencies (all pure-Python wheels):
    pip install torch==2.2.2 datasets==2.19.0 tqdm

No torchtext / torchdata / spacy required.
"""

from collections import Counter
from itertools import islice
from pathlib import Path
import random
import math
import os
from types import NotImplementedType

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

from src.transformer import Transformer
from src.common import get_default_configs

# --------------------------------------------------------------------------- #
#  Hyper-parameters & constants                                               #
# --------------------------------------------------------------------------- #
BOS, EOS, PAD = "<bos>", "<eos>", "<pad>"
MIN_FREQ = 5
MAX_SENT_LEN = 25  # drop longer sentences (keeps it light)
TRAIN_SAMPLES = 30_000  # subset – fits in RAM & trains fast on CPU
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cpu")
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------------------------------- #
#  1. Load OPUS-Books via HuggingFace                                         #
# --------------------------------------------------------------------------- #
print("Downloading OPUS-Books…")
data = load_dataset("opus_books", "de-en", split="train")  # 31 k examples
data = data.shuffle(seed=SEED)


def clean_pair(example):
    de, en = example["translation"]["de"], example["translation"]["en"]
    if len(de.split()) <= MAX_SENT_LEN and len(en.split()) <= MAX_SENT_LEN:
        return {"de": de.lower(), "en": en.lower()}


pairs = [clean_pair(ex) for ex in islice(data, TRAIN_SAMPLES)]
pairs = [p for p in pairs if p is not None]  # drop long ones


# --------------------------------------------------------------------------- #
#  2. Tokenisation & vocabulary                                               #
# --------------------------------------------------------------------------- #
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

id_de, id_en = vocab_de.__getitem__, vocab_en.__getitem__

print(f"German vocab: {len(vocab_de):,}  |  English vocab: {len(vocab_en):,}")


def encode(text, vocab):
    tokens = [vocab.get(t, vocab[PAD]) for t in tok(text)]
    tokens = [vocab[BOS]] + tokens + [vocab[EOS]]
    return [min(t, len(vocab) - 1) for t in tokens]


# --------------------------------------------------------------------------- #
#  3. Dataset / DataLoader                                                    #
# --------------------------------------------------------------------------- #
class TranslationDS(Dataset):
    def __len__(self):
        return len(pairs)

    def __getitem__(self, idx):
        src_encoded = torch.tensor(encode(pairs[idx]["de"], vocab_de))
        tgt_encoded = torch.tensor(encode(pairs[idx]["en"], vocab_en))

        # Defensive: Clamp out-of-bounds indices (optional)
        src_encoded = src_encoded.clamp(max=len(vocab_de) - 1)
        tgt_encoded = tgt_encoded.clamp(max=len(vocab_en) - 1)

        return src_encoded, tgt_encoded


def collate(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=vocab_de[PAD])
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=vocab_en[PAD])
    return src, tgt, src.eq(vocab_de[PAD]), tgt.eq(vocab_en[PAD])


loader = DataLoader(
    TranslationDS(), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
)

# --------------------------------------------------------------------------- #
#  4. Build Transformer                                                       #
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

optim = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab_en[PAD])

print(f"Model source vocab: {len(vocab_de)} | target vocab: {len(vocab_en)}")
print(f"Target embedding size: {model.target_embedding.num_embeddings}")


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


print("Training…")
for epoch in range(1, EPOCHS + 1):

    model.train()
    total = 0
    for src, tgt, src_pad, tgt_pad in tqdm(loader, desc=f"Epoch {epoch}"):
        if src.max() >= model.source_embedding.num_embeddings:
            raise ValueError("Source index out of embedding bounds")
        if tgt.max() >= model.target_embedding.num_embeddings:
            raise ValueError("Target index out of embedding bounds")

        optim.zero_grad()
        logits = model(src, tgt[:, :-1], src_pad, tgt_pad[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optim.step()
        total += loss.item()
    print(f"  avg loss: {total / len(loader):.3f}")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"weights_epoch{epoch}.pt")


# --------------------------------------------------------------------------- #
#  7. Demo                                                                    #
# --------------------------------------------------------------------------- #
demo_src = random.choice(pairs)["de"]
demo_tensor = torch.tensor(encode(demo_src, vocab_de), device=DEVICE).unsqueeze(0)
print(
    f"Max index in demo_tensor: {demo_tensor.max().item()} | vocab size: {len(vocab_de)}"
)
decoded = greedy(demo_tensor, demo_tensor.eq(vocab_de[PAD]))

print(f"Generated tokens: {decoded}")
print(f"Max vocab index: {max(vocab_en.values())}")


def detokenize(ids, vocab):
    inv = {v: k for k, v in vocab.items()}
    return " ".join(
        inv.get(i, f"<unk:{i}>")  # fall back to a placeholder token
        for i in ids
        if i not in (vocab[BOS], vocab[EOS], vocab[PAD])
    )


print("\n--- Demo Translations ---")
for i in range(5):
    demo_src = random.choice(pairs)["de"]
    demo_tensor = torch.tensor(encode(demo_src, vocab_de), device=DEVICE).unsqueeze(0)
    decoded = greedy(demo_tensor, demo_tensor.eq(vocab_de[PAD]))

    print(f"\nSample {i + 1}")
    print("DE:", demo_src)
    print("EN (greedy):", detokenize(decoded, vocab_en))

torch.save(model.state_dict(), "transformer_translate_weights.pt")
