import torch
import numpy as np
from typing import List, Tuple, Dict


LETTERS = "TSYBEFPCNXZARO"


def one_hot_to_letter(one_hot: np.ndarray) -> str:
    seq = []
    for vec in one_hot:
        if vec.sum() == 0:
            seq.append("n")
        else:
            seq.append(LETTERS[np.argmax(vec)])
    return "".join(seq)


def extract_activation(model, sequences: torch.Tensor):
    """返回 (activations, first_conv_layer)"""
    first_conv = model.conv_block1[0]
    with torch.no_grad():
        x = sequences.permute(0, 2, 1)
        acts = first_conv(x)
    return acts, first_conv


def extract_motifs(
    activations: torch.Tensor,
    sequences: torch.Tensor,
    kernel_idx: int,
    threshold: float = 0.0,
    subseq_len: int = 6,
) -> Tuple[List[Tuple[int, str]], List[int]]:
    """返回 [(sample_idx, letter_seq), ...] 和被跳过的索引"""
    acts = activations[:, kernel_idx]                # (B, L)
    motifs, skipped = [], []

    for idx in range(acts.size(0)):
        act = acts[idx]
        if (act <= threshold).all():
            skipped.append(idx)
            continue

        max_pos = int(act.argmax())
        half = subseq_len // 2
        start = max(0, max_pos - half)
        end = min(sequences.size(1), max_pos + half)
        sub = sequences[idx, start:end].numpy()
        motifs.append((idx, one_hot_to_letter(sub)))

    return motifs, skipped


def compute_pfm(motifs: List[str]) -> List[List[int]]:
    """返回 PFM 矩阵（每行对应字母，每列对应位置）"""
    if not motifs:
        return []
    seq_len = len(motifs[0])
    counts = {c: [0] * seq_len for c in LETTERS}
    for seq in motifs:
        for pos, c in enumerate(seq):
            if c in counts:
                counts[c][pos] += 1
    return [counts[c] for c in LETTERS]


def print_pfm(pfm: List[List[int]]):
    seq_len = len(pfm[0])
    header = ["pos"] + [str(i + 1) for i in range(seq_len)]
    print(",".join(header))
    for row, counts in zip(LETTERS, pfm):
        line = row + "," + ",".join(map(str, counts))
        print(line)