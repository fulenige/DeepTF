"""
TFSequenceDataset 及数据读取工具
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TFSequenceDataset(Dataset):
    """PyTorch Dataset 封装"""
    def __init__(self, sequences, labels):
        self.sequences = sequences  # shape: (N, 21, 15)
        self.labels    = labels     # shape: (N,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_data(file_path: str):
    """
    从 csv 读取 Encoded Sequences 和 Expression_Label
    返回 (tensor_seq, tensor_label)
    """
    seqs, labels = [], []
    df = pd.read_csv(file_path, sep='\t')

    for _, row in df.iterrows():
        try:
            enc = json.loads(row['Encoded Sequences'])
        except json.JSONDecodeError:
            continue

        if len(enc) != 21:
            continue
        arr = np.array(enc, dtype=np.float32)
        if arr.shape != (21, 15):
            continue

        seqs.append(arr)
        labels.append(int(row['Expression_Level']))

    seqs   = np.array(seqs)
    labels = np.array(labels, dtype=np.int64)
    return torch.tensor(seqs), torch.tensor(labels, dtype=torch.long)