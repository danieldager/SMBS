"""Token dataset for training language models.

Token Dataset uses WebDataset to stream tokenized audio data from .tar files.
It yields fixed-size blocks of tokens for training. Token sequences are packed.

"""

import random
import tarfile
from pathlib import Path
import webdataset as wds

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# CONSTANTS
SEED = 101
MAX_TOKENS = 2048
SHUFFLE_BUFFER = 1000


class TokenDataset(IterableDataset):
    def __init__(self, tokens_dir: str, bos_token_id: int, eos_token_id: int):

        # self.urls is all the paths strings to *.tar files in tokens_dir
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
                shardshuffle=False,  # TODO: check this
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
                print("Warning: 'tokens.npy' not found in sample, skipping.")
                continue  # Skip samples without tokens

            # token_tensor = torch.from_numpy(tokens).long()

            token_list = [self.bos_token_id] + tokens.tolist() + [self.eos_token_id]
            buffer.extend(token_list)

            # when we have enough tokens, cut and yield a block
            while len(buffer) >= self.block_size:
                block = buffer[: self.block_size]
                buffer = buffer[self.block_size :]

                yield {"input_ids": torch.tensor(block, dtype=torch.long)}


class EvalDataset(Dataset):
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
            shardshuffle=False,  # TODO: check this
            nodesplitter=wds.shardlists.split_by_node,
            handler=wds.warn_and_continue,  # type: ignore
        ).decode()
        # dataset = dataset.compose(wds.shardlists.split_by_worker)

        buffer = []
        self.blocks = []

        for _, sample in enumerate(dataset):

            if len(self.blocks) < self.num_blocks:
                tokens = sample.get("tokens.npy")
                if tokens is None:
                    print("Warning: 'tokens.npy' not found in sample, skipping.")
                    continue  # Skip samples without tokens

                token_list = [self.bos_token_id] + tokens.tolist() + [self.eos_token_id]
                buffer.extend(token_list)

                # when we have enough tokens, cut and yield a block
                while len(buffer) >= self.block_size:
                    block = buffer[: self.block_size]
                    buffer = buffer[self.block_size :]

                    block_tensor = torch.tensor(block, dtype=torch.long)
                    self.blocks.append(block_tensor)

            else:
                break  # Stop if we've collected enough blocks

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        return {"input_ids": self.blocks[idx]}


def collate_fn(batch):
    tensors = torch.stack([item["input_ids"] for item in batch])  # [B, MAX_TOKENS]
    attention_mask = torch.ones_like(tensors)

    return {
        "input_ids": tensors,
        "labels": tensors,
        "attention_mask": attention_mask,
    }


def check_shards(tokens_dir):
    """Scans all .tar files in the directory for corruption."""
    print(f"\n--- Scanning for corrupted shards in {tokens_dir} ---")
    paths = sorted(list(Path(tokens_dir).glob("*.tar")))
    corrupted = []

    for p in paths:
        try:
            with tarfile.open(p, "r") as tar:
                # We don't need to extract everything, just try to list members
                # This catches 'empty header' or 'truncated file' errors
                _ = tar.getnames()
        except Exception as e:
            print(f"[CORRUPT] {p.name}: {e}")
            corrupted.append(p)

    print(f"Scan complete. {len(corrupted)}/{len(paths)} corrupted.")

    return corrupted


if __name__ == "__main__":
    """Quick smoke test: load one batch and print shapes."""
    import argparse
    from scripts.encode.encoders import get_encoder_config

    parser = argparse.ArgumentParser(description="Smoke-test token dataset")
    parser.add_argument("tokens_dir", type=str)
    parser.add_argument("--encoder", type=str, default="spidr_base")
    args = parser.parse_args()

    enc = get_encoder_config(args.encoder)

    bad_files = check_shards(args.tokens_dir)
    if bad_files:
        print("Proceeding with caution...\n")

    dataset = TokenDataset(args.tokens_dir, enc.bos_token_id, enc.eos_token_id)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=2)

    for batch in loader:
        print(f"input_ids: {batch['input_ids'].shape}")
        print(f"labels:    {batch['labels'].shape}")
        print(f"mask:      {batch['attention_mask'].shape}")
        break
