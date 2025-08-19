import os
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset import TFSequenceDataset, load_data
from model import ConvNet
from utils import plot_metrics, plot_roc

# ---------------- 配置 ----------------
RAW_DATA    = "use_data.csv"
PROC_DIR    = "processed"
RES_DIR     = "results"
SEED        = 42
BATCH_SIZE  = 64
LR          = 0.0003
EPOCHS      = 50
KFOLD       = 3         
TRAIN_RATIO = 0.8
THRESHOLD   = 0.5
# ------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


def prepare_data():
    df = pd.read_csv(RAW_DATA, sep='\t')
    df['TPM_log'] = 10 * np.log10(df['TPM'] + 1)
    q1, q3 = df['TPM_log'].quantile(0.25), df['TPM_log'].quantile(0.75)
    df = df[(df['TPM_log'] <= q1) | (df['TPM_log'] >= q3)].copy()
    df['Expression_Level'] = (df['TPM_log'] >= q3).astype(int)
    df.to_csv(os.path.join(PROC_DIR, "low_high_data.csv"), index=False)
    return df


def split_data(df):
    tr_te = df.sample(frac=1, random_state=SEED)
    split = int(TRAIN_RATIO * len(tr_te))
    train_df, test_df = tr_te.iloc[:split], tr_te.iloc[split:]
    test_df.to_csv(os.path.join(PROC_DIR, "test_data.csv"), index=False)

    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    train_paths, val_paths = [], []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        tr_path = os.path.join(PROC_DIR, f"train_fold{fold}.csv")
        val_path = os.path.join(PROC_DIR, f"val_fold{fold}.csv")
        tr.to_csv(tr_path, index=False)
        val.to_csv(val_path, index=False)
        train_paths.append(tr_path)
        val_paths.append(val_path)
    return train_paths, val_paths, os.path.join(PROC_DIR, "test_data.csv")


def run_fold(train_csv, val_csv, fold):
    train_x, train_y = load_data(train_csv)
    val_x, val_y = load_data(val_csv)

    train_loader = DataLoader(TFSequenceDataset(train_x, train_y),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TFSequenceDataset(val_x, val_y),
                            batch_size=BATCH_SIZE, shuffle=False)

    model = ConvNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_auc, best_state = 0, None
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # train
        model.train()
        epoch_loss, preds, labels = 0, [], []
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            y = y.view(-1, 1).float()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds.extend(torch.sigmoid(out).cpu().detach().numpy().flatten())
            labels.extend(y.cpu().numpy().flatten())
        train_losses.append(epoch_loss / len(train_loader))

        # val
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                y = y.view(-1, 1).float()
                val_loss += criterion(out, y).item()
                val_preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
                val_labels.extend(y.cpu().numpy().flatten())
        val_losses.append(val_loss / len(val_loader))
        val_auc = roc_auc_score(val_labels, val_preds)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
        scheduler.step()

    torch.save(best_state, os.path.join(RES_DIR, f"best_fold{fold}.pth"))
    plot_metrics(train_losses, val_losses, 'Loss',
                 os.path.join(RES_DIR, f"loss_fold{fold}.png"))
    return best_auc


def main():
    df = prepare_data()
    train_paths, val_paths, test_csv = split_data(df)

    aucs = []
    for fold, (tr, val) in enumerate(zip(train_paths, val_paths)):
        auc = run_fold(tr, val, fold)
        print(f"Fold{fold} AUC: {auc:.4f}")
        aucs.append(auc)
    print("Mean AUC:", np.mean(aucs))


if __name__ == "__main__":
    main()