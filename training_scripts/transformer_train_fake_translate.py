# train_fake_translate.py

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random

from src.transformer import Transformer
from src.common import get_default_configs

PAD_ID = 0
END_ID = 999
SRC_VOCAB_SIZE = 1000
TGT_VOCAB_SIZE = 1000


CATEGORIES = {
    "DET": list(range(10, 20)),
    "NOUN": list(range(100, 200)),
    "VERB": list(range(200, 300)),
    "ADJ": list(range(300, 400)),
    "ADV": list(range(400, 500)),
}

SOURCE_PATTERNS = [
    ["DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "END"],
    ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "END"],
    ["DET", "NOUN", "VERB", "ADV", "DET", "NOUN", "END"],
]

TARGET_PATTERNS = [
    ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "END"],
    ["ADV", "VERB", "DET", "NOUN", "ADJ", "END"],
    ["DET", "NOUN", "ADV", "VERB", "DET", "NOUN", "END"],
]


def random_token(category: str) -> int:
    return random.choice(CATEGORIES[category])


class StructuredTranslationDataset(Dataset):
    def __init__(self, n_samples: int):
        self.pairs = []
        for _ in range(n_samples):
            src_pattern = random.choice(SOURCE_PATTERNS)
            tgt_pattern = random.choice(TARGET_PATTERNS)

            src = [random_token(cat) for cat in src_pattern[:-1]] + [END_ID]
            tgt = [random_token(cat) for cat in tgt_pattern[:-1]] + [END_ID]

            self.pairs.append((src, tgt))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])


def collate_batch(batch):
    src_seqs, tgt_seqs = zip(*batch)
    max_src = max(len(s) for s in src_seqs)
    max_tgt = max(len(s) for s in tgt_seqs)

    def pad(seqs, max_len):
        return torch.stack(
            [torch.cat([s, torch.full((max_len - len(s),), PAD_ID)]) for s in seqs]
        )

    src = pad(src_seqs, max_src)
    tgt = pad(tgt_seqs, max_tgt)
    src_mask = src.eq(PAD_ID)
    tgt_mask = tgt.eq(PAD_ID)
    return src, tgt, src_mask, tgt_mask


def train():
    torch.manual_seed(0)
    device = torch.device("cpu")

    dataset = StructuredTranslationDataset(1000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

    configs = get_default_configs()
    model = Transformer(
        d_model=128,
        n_head=8,
        d_ff=256,
        n_layer=3,
        dropout=0.1,
        source_vocab_size=SRC_VOCAB_SIZE,
        target_vocab_size=TGT_VOCAB_SIZE,
        pad_id=PAD_ID,
        configs=configs,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(6):
        total_loss = 0.0
        for src, tgt, src_mask, tgt_mask in loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            logits = model(src, tgt, src_mask, tgt_mask)
            B, L, V = logits.shape
            loss = loss_fn(logits.view(B * L, V), tgt.view(B * L))

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    evaluate(model, device)


def evaluate(model, device):
    model.eval()
    src_pattern = ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "END"]
    tgt_pattern = ["ADV", "VERB", "DET", "NOUN", "ADJ", "END"]

    src = [random_token(cat) for cat in src_pattern[:-1]] + [END_ID]
    tgt = [random_token(cat) for cat in tgt_pattern[:-1]] + [END_ID]

    src_tensor = torch.tensor([src], device=device)
    tgt_tensor = torch.tensor([tgt], device=device)
    src_mask = src_tensor.eq(PAD_ID)
    tgt_mask = tgt_tensor.eq(PAD_ID)

    with torch.no_grad():
        logits = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
        pred = logits.argmax(dim=-1)

    print("SRC: ", src)
    print("TGT: ", tgt)
    print("PRED:", pred[0].tolist())


if __name__ == "__main__":
    train()
