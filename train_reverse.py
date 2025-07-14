# train_reverse.py

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random

from src.transformer import Transformer
from src.common import get_default_configs

PAD_ID = 0
VOCAB_SIZE = 100
MAX_LEN = 12


class TinyReverseDataset(Dataset):
    def __init__(self, n_examples: int):
        self.samples = []
        for _ in range(n_examples):
            length = random.randint(3, MAX_LEN)
            tokens = [random.randint(1, VOCAB_SIZE - 1) for _ in range(length)]
            self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        input_ids = seq
        target_ids = list(reversed(seq))
        return torch.tensor(input_ids), torch.tensor(target_ids)


def collate_batch(batch):
    input_batch, target_batch = zip(*batch)
    input_lens = [len(x) for x in input_batch]
    max_len = max(input_lens)

    def pad(seqs):
        return torch.stack(
            [
                torch.cat([seq, torch.full((max_len - len(seq),), PAD_ID)])
                for seq in seqs
            ]
        )

    src = pad(input_batch).long()
    tgt = pad(target_batch).long()

    src_kpm = src.eq(PAD_ID)
    tgt_kpm = tgt.eq(PAD_ID)

    return src, tgt, src_kpm, tgt_kpm


def train():
    torch.manual_seed(0)
    device = torch.device("cpu")

    dataset = TinyReverseDataset(512)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)

    configs = get_default_configs()
    model = Transformer(
        d_model=64,
        n_head=4,
        d_ff=128,
        n_layer=2,
        dropout=0.1,
        source_vocab_size=VOCAB_SIZE,
        target_vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        configs=configs,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for src, tgt, src_mask, tgt_mask in loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            logits = model(src, tgt, src_mask, tgt_mask)
            B, L, V = logits.shape
            loss = criterion(logits.view(B * L, V), tgt.view(B * L))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    print("Training done.")
    evaluate(model, device)


def evaluate(model, device):
    model.eval()
    ex = [14, 9, 22, 88]
    src = torch.tensor([ex], device=device)
    tgt = torch.tensor([list(reversed(ex))], device=device)
    src_mask = src.eq(PAD_ID)
    tgt_mask = tgt.eq(PAD_ID)

    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask)
        preds = logits.argmax(dim=-1)

    print("Input:    ", ex)
    print("Target:   ", list(reversed(ex)))
    print("Predicted:", preds[0].tolist())


if __name__ == "__main__":
    train()
