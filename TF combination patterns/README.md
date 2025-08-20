## TF Combination Patterns - Motif Extraction and Visualization
This repository contains tools for extracting transcription factor binding motifs from a trained CNN model and visualizing them as sequence logos.

## Project Structure
TF combination patterns/
├── README.md
├── GM12878/results/
│   ├── test_data.csv            # Test dataset with encoded sequences and labels
│   ├── model.py                 # CNN model definition
│   ├── dataset.py               # Dataset loading and processing
│   └── best_model.pth           # Trained CNN model weights
├── extract_motifs.py            # Main script for motif extraction
├── utils.py                     # Utility functions
└── logo.R                       # R script for generating sequence logos

## Workflow
1. Extract Motifs
Run the Python script: python extract_motifs.py
2. Generate Sequence Logo
Rscript logo.R

## Requirements
Python 3.7+
PyTorch 1.0+
R 4.0+
ggplot2
ggseqlogo
