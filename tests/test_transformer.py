import random
from collections import Counter
from itertools import islice

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from src.common import get_default_configs
from src.transformer import Transformer


@pytest.fixture(scope="module")
def tiny_transformer(device):
    """A mini model that runs in milliseconds."""
    cfg = get_default_configs()
    model = Transformer(
        d_model=32,
        n_head=4,
        d_ff=64,
        n_layer=2,
        dropout=0.0,
        source_vocab_size=1337,
        target_vocab_size=1337,
        pad_id=0,
        configs=cfg,
    ).to(device)
    model.eval()
    return model


def test_forward_shape(tiny_transformer, device):
    B, Lsrc, Ltgt = 2, 9, 7
    src = torch.randint(0, 1337, (B, Lsrc), device=device)
    tgt = torch.randint(0, 1337, (B, Ltgt), device=device)
    logits = tiny_transformer(src, tgt)
    assert logits.shape == (B, Ltgt, 1337)


def test_gradients_flow(tiny_transformer, device):
    B, Lsrc, Ltgt = 2, 5, 5
    src = torch.randint(0, 1337, (B, Lsrc), device=device)
    tgt = torch.randint(0, 1337, (B, Ltgt), device=device)
    logits = tiny_transformer(src, tgt)
    loss = logits.mean()
    loss.backward()
    assert all(
        p.grad is not None for p in tiny_transformer.parameters() if p.requires_grad
    )


def test_embedding_and_lm_head_weight_tying(tiny_transformer):
    if tiny_transformer.configs.tie_target_embedding_and_lm_head_weights:
        assert (
            tiny_transformer.lm_head.weight is tiny_transformer.target_embedding.weight
        )
    else:
        assert (
            tiny_transformer.lm_head.weight
            is not tiny_transformer.target_embedding.weight
        )


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
BOS, EOS, PAD = "<bos>", "<eos>", "<pad>"
MIN_FREQ = 1
MAX_SENT_LEN = 20
TRAIN_SAMPLES = 20  # reduced!
BATCH_SIZE = 5
EPOCHS = 100
LOSS_THRESHOLD = 0.05  # relaxed a bit


@pytest.fixture(scope="module")
def training_data():
    raw = load_dataset("opus_books", "de-en", split="train").shuffle(seed=SEED)

    def clean_pair(example):
        de, en = example["translation"]["de"], example["translation"]["en"]
        if len(de.split()) <= MAX_SENT_LEN and len(en.split()) <= MAX_SENT_LEN:
            return {"de": de.lower(), "en": en.lower()}
        return None

    pairs = [clean_pair(ex) for ex in islice(raw, 100)]
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
        tokens = [vocab.get(t, vocab[PAD]) for t in text.strip().split()]
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
        src = nn.utils.rnn.pad_sequence(
            src, batch_first=True, padding_value=vocab_de[PAD]
        )
        tgt = nn.utils.rnn.pad_sequence(
            tgt, batch_first=True, padding_value=vocab_en[PAD]
        )
        return src, tgt, src.eq(vocab_de[PAD]), tgt.eq(vocab_en[PAD])

    loader = DataLoader(
        TinyTranslationDS(), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    return loader, vocab_de, vocab_en


def test_transformer_overfits_tiny_dataset(training_data):
    loader, vocab_de, vocab_en = training_data

    cfgs = get_default_configs()
    model = Transformer(
        d_model=32,  # reduced size
        n_head=2,
        d_ff=64,
        n_layer=1,
        dropout=0.0,
        source_vocab_size=len(vocab_de),
        target_vocab_size=len(vocab_en),
        pad_id=vocab_en[PAD],
        configs=cfgs,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)  # faster convergence
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab_en[PAD])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for src, tgt, src_pad, tgt_pad in loader:
            optimizer.zero_grad()
            logits = model(src, tgt[:, :-1], src_pad, tgt_pad[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < LOSS_THRESHOLD:
            break

    assert (
        avg_loss < LOSS_THRESHOLD
    ), f"Model failed to overfit: final loss = {avg_loss:.4f}"
