"""Token datasets for training language models.

TokenDataset: streaming IterableDataset from WebDataset .tar shards.
EvalDataset: fixed-size Dataset for evaluation.
Both pack token sequences into fixed-size blocks with BOS/EOS wrapping.
"""

import random
import tarfile
from pathlib import Path

import torch
import webdataset as wds
from torch.utils.data import Dataset, IterableDataset

from smbs.config import MAX_TOKENS, SEED, SHUFFLE_BUFFER


class TokenDataset(IterableDataset):
    """Streaming token dataset that packs sequences into fixed-size blocks."""

    def __init__(self, tokens_dir: str, bos_token_id: int, eos_token_id: int):
        self.urls = sorted([str(p) for p in Path(tokens_dir).glob("*.tar")])
        self.block_size = MAX_TOKENS
        self.shuffle_buffer = SHUFFLE_BUFFER
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        random.seed(SEED)
        random.shuffle(self.urls)

    def __iter__(self):
        dataset = (
            wds.WebDataset(  # type: ignore
                self.urls,
                resampled=True,
                shardshuffle=False,
                nodesplitter=wds.shardlists.split_by_node,
                handler=wds.warn_and_continue,  # type: ignore
            )
            .shuffle(self.shuffle_buffer)
            .decode()
        )
        dataset = dataset.compose(wds.shardlists.split_by_worker)

        buffer = []

        for sample in dataset:
            tokens = sample.get("tokens.npy")
            if tokens is None:
                continue

            token_list = [self.bos_token_id] + tokens.tolist() + [self.eos_token_id]
            buffer.extend(token_list)

            while len(buffer) >= self.block_size:
                block = buffer[: self.block_size]
                buffer = buffer[self.block_size :]
                yield {"input_ids": torch.tensor(block, dtype=torch.long)}


class EvalDataset(Dataset):
    """Fixed-size evaluation dataset that pre-loads blocks into memory."""

    def __init__(self, tokens_dir: str, bos_token_id: int, eos_token_id: int, num_blocks: int = 3000):
        self.urls = sorted([str(p) for p in Path(tokens_dir).glob("*.tar")])
        self.block_size = MAX_TOKENS
        self.num_blocks = num_blocks
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        random.seed(SEED)
        random.shuffle(self.urls)

        dataset = wds.WebDataset(  # type: ignore
            self.urls,
            resampled=True,
            shardshuffle=False,
            nodesplitter=wds.shardlists.split_by_node,
            handler=wds.warn_and_continue,  # type: ignore
        ).decode()

        buffer = []
        self.blocks = []

        for sample in dataset:
            if len(self.blocks) >= self.num_blocks:
                break

            tokens = sample.get("tokens.npy")
            if tokens is None:
                continue

            token_list = [self.bos_token_id] + tokens.tolist() + [self.eos_token_id]
            buffer.extend(token_list)

            while len(buffer) >= self.block_size:
                block = buffer[: self.block_size]
                buffer = buffer[self.block_size :]
                self.blocks.append(torch.tensor(block, dtype=torch.long))

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        return {"input_ids": self.blocks[idx]}


def collate_fn(batch):
    tensors = torch.stack([item["input_ids"] for item in batch])
    return {
        "input_ids": tensors,
        "labels": tensors,
        "attention_mask": torch.ones_like(tensors),
    }


def check_shards(tokens_dir: str) -> list[Path]:
    """Scan all .tar files in the directory for corruption."""
    print(f"\n--- Scanning for corrupted shards in {tokens_dir} ---")
    paths = sorted(list(Path(tokens_dir).glob("*.tar")))
    corrupted = []

    for p in paths:
        try:
            with tarfile.open(p, "r") as tar:
                _ = tar.getnames()
        except Exception as e:
            print(f"[CORRUPT] {p.name}: {e}")
            corrupted.append(p)

    print(f"Scan complete. {len(corrupted)}/{len(paths)} corrupted.")
    return corrupted
