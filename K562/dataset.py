import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TFSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def load_data(file_path: str):
    sequences, labels = [], []
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
        sequences.append(arr)
        labels.append(int(row['Expression_Level']))
    sequences = np.array(sequences)
    return torch.tensor(sequences), torch.tensor(labels, dtype=torch.long)