from pathlib import Path
from datetime import datetime
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / RUN_ID
RUN_DIR.mkdir(exist_ok=True)

LOG_PATH = RUN_DIR / "train.log"
METRICS_PATH = RUN_DIR / "metrics.txt"

FULL_REDCAP_PATH = DATA_DIR / "Full REDCap Data Intervention for Dexcom FINAL.xlsx"
MASTER_CLARITY_PATH = DATA_DIR / "Master Clarity log 1-101 for Dexcom FINAL.xlsx"


# Utilities
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_message(message):
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def yes_no_to_flag(series):
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["yes", "y", "true", "1"]).astype(np.float32)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.float32)


def make_unique_columns(columns):
    seen = {}
    new_cols = []
    for col in map(str, columns):
        col = col.strip()
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}__dup{seen[col]}")
    return new_cols


# Load data
def load_excel_files():
    log_message("Loading Excel files...")

    full_redcap = pd.read_excel(FULL_REDCAP_PATH)
    master_clarity = pd.read_excel(MASTER_CLARITY_PATH)

    full_redcap.columns = make_unique_columns(full_redcap.columns)
    master_clarity.columns = make_unique_columns(master_clarity.columns)

    log_message("Loaded successfully.")
    return full_redcap, master_clarity


# Cleaning
def clean_master_clarity(master_clarity):
    df = master_clarity.copy()

    timestamp_col = next((col for col in df.columns if "imes" in str(col)), None)
    if timestamp_col is None:
        raise ValueError("Could not find timestamp column in Master Clarity.")

    df = df.rename(columns={timestamp_col: "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df[df["Sub ID"].notna() & df["Timestamp"].notna()].copy()
    df["Sub ID"] = df["Sub ID"].astype(str).str.strip()

    df["Glucose_raw"] = df["Glucose Value (mg/dL)"].astype(str).str.strip()
    df["Glucose_text_upper"] = df["Glucose_raw"].str.upper()
    df["Glucose_numeric"] = pd.to_numeric(df["Glucose_raw"], errors="coerce")

    df["Is_High"] = (
        (df["Glucose_text_upper"] == "HIGH") |
        (df["Glucose_numeric"] > 400)
    ).astype(int)

    df = df.sort_values(["Sub ID", "Timestamp"]).reset_index(drop=True)

    log_message(f"Cleaned Master Clarity shape: {df.shape}")
    log_message(f"Rows with numeric glucose: {df['Glucose_numeric'].notna().sum()}")
    log_message(f"High count: {df['Is_High'].sum()}")
    return df


def clean_daily_insulin(full_redcap):
    df = full_redcap[full_redcap["Repeat Instrument"] == "Daily Insulin Dosing"].copy()

    if df.empty:
        log_message("No Daily Insulin Dosing rows found.")
        return pd.DataFrame(columns=[
            "Study_ID", "Daily_Date",
            "iv_insulin_flag", "sc_insulin_flag",
            "subq_bolus_units", "basal_insulin_units", "nph_units"
        ])

    df["Study_ID"] = df["Unique Study ID"].astype(str).str.strip()
    df["Daily_Date"] = pd.to_datetime(df["Date .1"], errors="coerce").dt.date

    df["iv_insulin_flag"] = yes_no_to_flag(df["Is the patient on IV insulin"])
    df["sc_insulin_flag"] = yes_no_to_flag(df["Is the patient on subQ insulin"])
    df["subq_bolus_units"] = safe_numeric(df["Total number of units of SubQ bolus insulin received"])
    df["basal_insulin_units"] = safe_numeric(df["Total number of units in basal insulin dose"])
    df["nph_units"] = safe_numeric(df["How many daily units of NPH insulin is patient receiving"])

    df = df[[
        "Study_ID", "Daily_Date",
        "iv_insulin_flag", "sc_insulin_flag",
        "subq_bolus_units", "basal_insulin_units", "nph_units"
    ]].dropna(subset=["Study_ID", "Daily_Date"])

    df = df.groupby(["Study_ID", "Daily_Date"], as_index=False).agg({
        "iv_insulin_flag": "max",
        "sc_insulin_flag": "max",
        "subq_bolus_units": "sum",
        "basal_insulin_units": "sum",
        "nph_units": "sum",
    })

    log_message(f"Cleaned Daily Insulin Dosing shape: {df.shape}")
    return df


def clean_daily_nutrition(full_redcap):
    df = full_redcap[full_redcap["Repeat Instrument"] == "Daily Clinical Condition and Use"].copy()

    if df.empty:
        log_message("No Daily Clinical Condition and Use rows found.")
        return pd.DataFrame(columns=[
            "Study_ID", "Daily_Date",
            "enteral_nutrition_flag", "enteral_feed_duration",
            "tpn_flag", "tpn_duration"
        ])

    if "Date" not in df.columns:
        raise ValueError("Could not find nutrition date column 'Date'.")

    df["Study_ID"] = df["Unique Study ID"].astype(str).str.strip()
    df["Daily_Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    df["enteral_nutrition_flag"] = yes_no_to_flag(df["Was the patient receiving enteral nutrition"])
    df["enteral_feed_duration"] = safe_numeric(df["Duration of enteral feed"])
    df["tpn_flag"] = yes_no_to_flag(df["Was the patient receiving parenteral nutrition (TPN)?"])
    df["tpn_duration"] = safe_numeric(df["Duration of TPN"])

    df = df[[
        "Study_ID", "Daily_Date",
        "enteral_nutrition_flag", "enteral_feed_duration",
        "tpn_flag", "tpn_duration"
    ]].dropna(subset=["Study_ID", "Daily_Date"])

    df = df.groupby(["Study_ID", "Daily_Date"], as_index=False).agg({
        "enteral_nutrition_flag": "max",
        "enteral_feed_duration": "max",
        "tpn_flag": "max",
        "tpn_duration": "max",
    })

    log_message(f"Cleaned Daily Nutrition shape: {df.shape}")
    return df


# Feature lookup
def get_insulin_features(daily_insulin_clean, subject_id, current_time):
    df = daily_insulin_clean[
        (daily_insulin_clean["Study_ID"] == str(subject_id).strip()) &
        (daily_insulin_clean["Daily_Date"] == current_time.date())
    ]
    if df.empty:
        return [0.0] * 5

    row = df.iloc[0]
    return [
        float(row["iv_insulin_flag"]),
        float(row["sc_insulin_flag"]),
        float(row["subq_bolus_units"]),
        float(row["basal_insulin_units"]),
        float(row["nph_units"]),
    ]


def get_nutrition_features(daily_nutrition_clean, subject_id, current_time):
    df = daily_nutrition_clean[
        (daily_nutrition_clean["Study_ID"] == str(subject_id).strip()) &
        (daily_nutrition_clean["Daily_Date"] == current_time.date())
    ]
    if df.empty:
        return [0.0] * 4

    row = df.iloc[0]
    return [
        float(row["enteral_nutrition_flag"]),
        float(row["enteral_feed_duration"]),
        float(row["tpn_flag"]),
        float(row["tpn_duration"]),
    ]


# Sequence building for regression
def build_transformer_sequences(
    clarity_clean,
    daily_insulin_clean,
    daily_nutrition_clean,
    seq_len=36,
):
    log_message("Building transformer sequences for regression...")

    df = clarity_clean[clarity_clean["Event Type"] == "EGV"].copy()
    df = df.sort_values(["Sub ID", "Timestamp"]).reset_index(drop=True)

    sequences, labels, subject_ids = [], [], []

    history_window = pd.Timedelta(hours=3)
    future_window = pd.Timedelta(hours=1)

    for subject_id, group in df.groupby("Sub ID"):
        group = group.sort_values("Timestamp").reset_index(drop=True)

        for i in range(len(group)):
            if pd.isna(group.loc[i, "Glucose_numeric"]):
                continue

            current_time = group.loc[i, "Timestamp"]

            history_df = group[
                (group["Timestamp"] >= current_time - history_window) &
                (group["Timestamp"] <= current_time) &
                (group["Glucose_numeric"].notna())
            ].copy()

            future_df = group[
                (group["Timestamp"] > current_time) &
                (group["Timestamp"] <= current_time + future_window) &
                (group["Glucose_numeric"].notna())
            ].copy()

            if len(history_df) < 6 or len(future_df) < 1:
                continue

            target_glucose = float(future_df["Glucose_numeric"].iloc[-1])

            glucose_seq = history_df["Glucose_numeric"].values.astype(np.float32)
            time_seq = history_df["Timestamp"]

            if len(glucose_seq) > seq_len:
                glucose_seq = glucose_seq[-seq_len:]
                time_seq = time_seq.iloc[-seq_len:]

            padded_seq = np.zeros(seq_len, dtype=np.float32)
            time_gap = np.zeros(seq_len, dtype=np.float32)
            glucose_delta = np.zeros(seq_len, dtype=np.float32)
            glucose_slope = np.zeros(seq_len, dtype=np.float32)
            rolling_mean = np.zeros(seq_len, dtype=np.float32)
            rolling_std = np.zeros(seq_len, dtype=np.float32)

            padded_seq[-len(glucose_seq):] = glucose_seq

            if len(time_seq) > 1:
                deltas = time_seq.diff().dt.total_seconds().fillna(0).values.astype(np.float32) / 60.0
            else:
                deltas = np.zeros(len(glucose_seq), dtype=np.float32)

            time_gap[-len(glucose_seq):] = deltas

            g_delta = np.diff(glucose_seq, prepend=glucose_seq[0]).astype(np.float32)
            glucose_delta[-len(glucose_seq):] = g_delta

            safe_gap = deltas.copy()
            safe_gap[safe_gap == 0] = 1.0
            glucose_slope[-len(glucose_seq):] = (g_delta / safe_gap).astype(np.float32)

            g_series = pd.Series(glucose_seq)
            rolling_mean[-len(glucose_seq):] = g_series.rolling(3, min_periods=1).mean().values.astype(np.float32)
            rolling_std[-len(glucose_seq):] = (
                g_series.rolling(3, min_periods=1).std().fillna(0).values.astype(np.float32)
            )

            insulin = get_insulin_features(daily_insulin_clean, subject_id, current_time)
            nutrition = get_nutrition_features(daily_nutrition_clean, subject_id, current_time)

            fixed_features = [
                np.full(seq_len, v, dtype=np.float32)
                for v in insulin + nutrition
            ]

            seq_features = np.stack(
                [
                    padded_seq,
                    time_gap,
                    glucose_delta,
                    glucose_slope,
                    rolling_mean,
                    rolling_std,
                ] + fixed_features,
                axis=1
            )

            sequences.append(seq_features)
            labels.append(target_glucose)
            subject_ids.append(str(subject_id).strip())

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    subject_ids = np.array(subject_ids)

    log_message(f"Transformer sequence shape: {X.shape}")
    log_message(f"Transformer label shape: {y.shape}")
    log_message(f"Number of unique subjects: {len(np.unique(subject_ids))}")
    if len(y) > 0:
        log_message(f"Target glucose range: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

    return X, y, subject_ids


# Dataset
class GlucoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Model
class GlucoseTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.regressor(x).squeeze(-1)


# Train / Evaluate
def train_transformer_model(X, y, subject_ids, epochs=5, batch_size=64, lr=1e-3):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    subject_train, subject_test = subject_ids[train_idx], subject_ids[test_idx]

    log_message(f"Train samples: {len(train_idx)}")
    log_message(f"Test samples: {len(test_idx)}")
    log_message(f"Train subjects: {len(np.unique(subject_train))}")
    log_message(f"Test subjects: {len(np.unique(subject_test))}")

    feature_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feature_std = X_train.std(axis=(0, 1), keepdims=True)
    feature_std[feature_std == 0] = 1.0

    X_train = (X_train - feature_mean) / feature_std
    X_test = (X_test - feature_mean) / feature_std

    train_loader = DataLoader(GlucoseDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(GlucoseDataset(X_test, y_test), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlucoseTransformer(input_dim=X.shape[2]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_message(f"Training on device: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        log_message(f"Epoch {epoch + 1}/{epochs}, Train MSE Loss: {avg_loss:.4f}")

    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy()

            all_pred.extend(preds)
            all_true.extend(batch_y.numpy())

    all_true = np.array(all_true, dtype=np.float32)
    all_pred = np.array(all_pred, dtype=np.float32)

    mae = mean_absolute_error(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)

    log_message("\n===== FINAL RESULTS =====")
    log_message(f"MAE: {mae:.4f}")
    log_message(f"RMSE: {rmse:.4f}")
    log_message(f"R2: {r2:.4f}")

    preview_df = pd.DataFrame({
        "true_glucose": all_true[:20],
        "pred_glucose": all_pred[:20],
        "abs_error": np.abs(all_true[:20] - all_pred[:20]),
    })
    log_message("\nPrediction preview (first 20 rows):")
    log_message(preview_df)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("===== FINAL RESULTS =====\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2: {r2:.4f}\n\n")
        f.write("Prediction preview (first 20 rows):\n")
        f.write(preview_df.to_string(index=False))
        f.write("\n")

    return model, {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# Main
if __name__ == "__main__":
    set_seed(42)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("Training log\n")

    full_redcap_df, master_clarity_df = load_excel_files()

    clarity_clean_df = clean_master_clarity(master_clarity_df)
    daily_insulin_clean_df = clean_daily_insulin(full_redcap_df)
    daily_nutrition_clean_df = clean_daily_nutrition(full_redcap_df)

    X_seq, y_seq, subject_ids = build_transformer_sequences(
        clarity_clean=clarity_clean_df,
        daily_insulin_clean=daily_insulin_clean_df,
        daily_nutrition_clean=daily_nutrition_clean_df,
        seq_len=36,
    )

    if len(X_seq) == 0:
        raise ValueError("No valid regression sequences were built. Please check the data filtering logic.")

    model, metrics = train_transformer_model(
        X_seq,
        y_seq,
        subject_ids,
        epochs=5,
        batch_size=64,
        lr=1e-3,
    )

    log_message("\nDone.")