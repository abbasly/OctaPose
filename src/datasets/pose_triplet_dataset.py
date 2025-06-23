import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import ast
import random
import re


def safe_parse(val):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val

def extract_fighter_name(video):
            return re.match(r"([a-zA-Z]+)", video).group(1)  # e.g., conor1.mp4 → conor


def preprocess_sequence(df):
    df["pose_xy"] = df["pose_xy"].apply(safe_parse)
    df["kp_conf"] = df["kp_conf"].apply(safe_parse)

    poses = []
    for i, row in df.iterrows():
        pose = np.vstack(row.pose_xy).astype(np.float32)
        conf = np.array(row.kp_conf)  # (17,)

        # Mask low-confidence keypoints
        pose[conf < 0.3] = 0.0

        # Normalize: center & scale
        midhip = (pose[11] + pose[12]) / 2 if np.all(pose[11]) and np.all(pose[12]) else pose.mean(axis=0)
        centered = pose - midhip
        scale = np.linalg.norm(centered.max(axis=0) - centered.min(axis=0)) + 1e-6
        normalized = centered / scale

        poses.append(normalized)

    return torch.tensor(poses, dtype=torch.float32)  # (T, 17, 2)


class PoseTripletDataset(Dataset):
    def __init__(self, table):
        self.data = []
        df = table.to_pandas()
        df["fighter_name"] = df["video"].apply(extract_fighter_name)

        # Group by fighter + sequence
        grouped = df.groupby(["fighter_name", "video"])
        for (fighter, seq_id), group in grouped:
            # print(group)
            tensor = preprocess_sequence(group)
            self.data.append({
                "fighter": fighter,
                "seq_id": seq_id,
                "pose": tensor,
                "length": tensor.shape[0]
            })

        # Index by fighter
        self.by_fighter = {}
        for idx, entry in enumerate(self.data):
            self.by_fighter.setdefault(entry["fighter"], []).append(idx)
        self.fighters = list(self.by_fighter.keys())
        self.label_map = {name: i for i, name in enumerate(self.fighters)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor_idx = random.randint(0, len(self.data) - 1)
        anchor_idx = index
        anchor = self.data[anchor_idx]
        fighter = anchor["fighter"]

        # Sample positive (different sequence of same fighter)
        pos_pool = [i for i in self.by_fighter[fighter] if i != anchor_idx]
        positive_idx = random.choice(pos_pool) if pos_pool else anchor_idx

        # Sample negative (different fighter)
        neg_fighter = random.choice([f for f in self.fighters if f != fighter])
        negative_idx = random.choice(self.by_fighter[neg_fighter])
        label = self.label_map[anchor["fighter"]]
        # return anchor["pose"], positive["pose"], negative["pose"], label

        return anchor["pose"], self.data[positive_idx]["pose"], self.data[negative_idx]["pose"], label


def collate_triplets(batch):
    def pad(seq, max_len):
        pad_len = max_len - seq.shape[0]
        if pad_len <= 0:
            return seq[:max_len]
        pad_tensor = torch.zeros((pad_len, 17, 2))
        return torch.cat([seq, pad_tensor], dim=0)

    anchor, positive, negative, label = zip(*batch)
    max_len = max(max(a.shape[0] for a in anchor), max(p.shape[0] for p in positive), max(n.shape[0] for n in negative))

    anchor = torch.stack([pad(a, max_len) for a in anchor])
    positive = torch.stack([pad(p, max_len) for p in positive])
    negative = torch.stack([pad(n, max_len) for n in negative])

    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "lengths": torch.tensor([a.shape[0] for a in anchor]),
        "labels": torch.tensor(label)
    }


# Usage:
# table = pq.read_table("poses/clean/all_sequences.parquet")
# dataset = PoseTripletDataset(table)
# loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_triplets)
