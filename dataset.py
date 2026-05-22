"""
dataset.py - PTB-XL ECG veri seti ön işleme modülü.
Kaynak: Wagner et al., Scientific Data 2020 (Research Paper)
"""

import os
import numpy as np
import pandas as pd
import wfdb
import ast
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

DATA_PATH  = "./ptb-xl/"
SEQ_LEN    = 1000
N_CHANNELS = 12
BATCH_SIZE = 32


def load_labels(df):
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    agg_df = pd.read_csv(DATA_PATH + "scp_statements.csv", index_col=0)
    # diagnostic=1 olan kodları kullan
    agg_df = agg_df[agg_df["diagnostic"] == 1.0]
    valid_codes = set(agg_df.index)

    def get_label(scp_dict):
        for code in scp_dict.keys():
            if code in valid_codes:
                if "diagnostic_class" in agg_df.columns:
                    val = agg_df.loc[code, "diagnostic_class"]
                    if pd.notna(val) and str(val).strip() != "":
                        return str(val).strip()
                return code
        return None

    df["label"] = df["scp_codes"].apply(get_label)
    df = df.dropna(subset=["label"])
    return df


def load_raw_data(df):
    data = []
    print(f"  {len(df)} kayit okunuyor (3-5 dakika surebilir)...")
    for i, f in enumerate(df["filename_lr"]):
        try:
            record = wfdb.rdrecord(DATA_PATH + f)
            data.append(record.p_signal)
        except Exception:
            data.append(np.zeros((SEQ_LEN, N_CHANNELS)))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(df)} kayit okundu...")
    return np.array(data)


def prepare_data():
    print("  Metadata yukleniyor...")
    df = pd.read_csv(DATA_PATH + "ptbxl_database.csv", index_col="ecg_id")
    df = load_labels(df)

    print(f"Toplam kayit: {len(df)}")
    print(f"Sinif dagilimi:\n{df['label'].value_counts()}\n")

    le = LabelEncoder()
    y  = le.fit_transform(df["label"].values)
    n_classes = len(le.classes_)
    print(f"Siniflar ({n_classes}): {le.classes_}")

    print("EKG sinyalleri yukleniyor...")
    X = load_raw_data(df)
    print(f"Veri sekli: {X.shape}")

    X = np.nan_to_num(X)
    N, T, C = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, C)).reshape(N, T, C)

    folds      = df["strat_fold"].values
    X_train, y_train = X[folds <= 8], y[folds <= 8]
    X_val,   y_val   = X[folds == 9], y[folds == 9]
    X_test,  y_test  = X[folds == 10], y[folds == 10]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return (make_loader(X_train, y_train, True),
            make_loader(X_val,   y_val,   False),
            make_loader(X_test,  y_test,  False),
            n_classes, le.classes_)


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loader(X, y, shuffle):
    return DataLoader(ECGDataset(X, y), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0)
