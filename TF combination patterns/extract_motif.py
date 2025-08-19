#!/usr/bin/env python3
"""
命令行入口：python extract_motifs.py
可在同级目录放置 .env 或 argparse 控制路径与超参。
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from dataset import load_data
from model import ConvNet
from utils import extract_activation, extract_motifs, compute_pfm, print_pfm

# 默认相对路径，便于 clone 即可跑
DEFAULT_MODEL = Path(__file__).with_suffix('').parent.parent / "model" / "best_model14.pth"
DEFAULT_DATA = Path(__file__).with_suffix('').parent.parent / "data" / "test_data.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract motifs from CNN kernels.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to .pth model")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Path to test_data.csv")
    parser.add_argument("--kernel", type=int, default=0, help="Index of conv kernel to inspect")
    parser.add_argument("--threshold", type=float, default=0.0, help="Activation threshold")
    parser.add_argument("--subseq", type=int, default=6, help="Length of extracted subsequence")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"[ERROR] Data not found: {args.data}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = ConvNet()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 2. 加载数据（仅保留 label==1）
    sequences, labels = load_data(args.data)
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    pos_seq = sequences[pos_idx]

    # 3. 提取激活
    acts, _ = extract_activation(model, pos_seq)

    # 4. 提取 motif
    motifs, skipped = extract_motifs(
        acts, pos_seq, args.kernel, args.threshold, args.subseq
    )
    print(f"[INFO] Skipped {len(skipped)} samples (all activations <= threshold).")

    # 5. 计算并打印 PFM
    pfm = compute_pfm([m[1] for m in motifs])
    print_pfm(pfm)

    # 6. 打印子序列
    print("\nExtracted subsequences:")
    for idx, seq in motifs:
        print(f"sample_{idx}: {seq}")


if __name__ == "__main__":
    main()