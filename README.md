## TF Regulatory Code

 Deciphering TF Regulatory Code for Gene Expression

## 1. Overview
This repository provides an end-to-end workflow that encodes transcription factor (TFs) binding patterns inside promoter regions into interpretable letter sequences and mines the regulatory code governing gene expression.
Gene-expression prediction is not the focus—it is merely a downstream sanity check for the encoding.

## 2. Repository Layout
TF-Regulatory-Code/

├── overlaps.py                 # 1️⃣ TF-promoter overlap detection

├── coding.py                   # 2️⃣ Encode overlaps → raw_data.txt

├── GM12878/                    # Example cell line (K562 mirrored)

│   ├── raw_data.txt            # Encoded table: Gene  Sequence  TPM …

│   ├── use_data.csv            # One-hot + TF-strength model input

│   ├── dataset.py              # PyTorch Dataset

│   ├── model.py                # CNN architecture

│   ├── train.py                # Training / 5-fold-CV / ROC plots

│   └── utils.py                # Shared utilities

├── TF_combination_patterns/    # 3️⃣ Decoding: motif extraction

│   ├── extract_motif.py        # Main script

│   ├── logo.R                  # Optional sequence-logo plotting

│   └── utils.py

└── README.md                   # This file

## 3. Quick Start
3.1 Data Preparation

(1) Provide three BED files:
promoters.bed  (promoter coordinates per gene)
TSS.bed        (TSS positions with TF names)
storm.bed/<CELL>/ directory with individual TF ChIP-seq peaks named <TF>.bed

(2) Run the encoding pipeline:
python overlaps.py      # → overlaps(GM12878).csv
python coding.py        # → raw_data.txt & tss_and_centers_sorted.txt

(3) Generate model inputs:
cd GM12878
python utils.py         # raw_data.txt → one-hot → use_data.csv

3.2 Train Model
python train.py

3.3 Extract TF-Combination Motifs

cd ../TF_combination_patterns
python utils.py \
python extract_motif.py \
    --model  /path/to/best_model.pth \
    --test   /path/to/test_data.csv \
    --outdir ./R-GM12878

## 4. Encoding Alphabet
| TF      | Letter | TF          | Letter |
| ------- | ------ | ----------- | ------ |
| TBP     | T      | CTCF        | C      |
| SP1     | S      | NRF1        | N      |
| YY1     | Y      | YBX1        | X      |
| BHLHE40 | B      | CEBPZ       | Z      |
| ETS1    | E      | NFYA        | A      |
| NFYB    | F      | NR2C2       | R      |
| POLR2A  | P      | TSS         | O      |
Strand handling: sequences on the – strand are reverse-complemented.
Fixed length: 21 bp length.





