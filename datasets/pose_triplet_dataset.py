import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import random
from collections import defaultdict

class PoseTripletDataset(Dataset):
    def __init__(self, parquet_path, sequence_length=150):
        super().__init__()
        self.sequence_length = sequence_length
        self.track_sequences = defaultdict(list)  # { (video, track_id): [frame_dict, ...] }

        # --- Load and organize parquet data ---
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Sort by video, track_id, frame_idx
        df = df.sort_values(by=["video", "track_id", "frame_idx"])

        # Group poses by (video, track_id)
        for _, row in df.iterrows():
            key = (row["video"], row["track_id"])
            self.track_sequences[key].append(row)

        # Keep only keys with enough frames
        self.valid_keys = [
            key for key, frames in self.track_sequences.items()
            if len(frames) >= self.sequence_length
        ]

    def __len__(self):
        return len(self.valid_keys)

    def get_last_sequence(self, frames):
        # Get the last 150 frames
        sliced = frames[-self.sequence_length:]
        pose_seq = [np.array(p["pose_xy"]).flatten() for p in sliced]
        return torch.tensor(pose_seq, dtype=torch.float32)  # [seq_len, 34]

    def __getitem__(self, idx):
        anchor_key = self.valid_keys[idx]
        anchor_frames = self.track_sequences[anchor_key]
        anchor_seq = self.get_last_sequence(anchor_frames)

        # Positive: same fighter, slightly earlier sequence (if long enough)
        if len(anchor_frames) >= 2 * self.sequence_length:
            positive_frames = anchor_frames[-2*self.sequence_length:-self.sequence_length]
        else:
            positive_frames = anchor_frames[-self.sequence_length:]
        positive_seq = self.get_last_sequence(positive_frames)

        # Negative: random fighter
        neg_key = random.choice([k for k in self.valid_keys if k[1] != anchor_key[1]])
        neg_frames = self.track_sequences[neg_key]
        negative_seq = self.get_last_sequence(neg_frames)

        return anchor_seq, positive_seq, negative_seq
