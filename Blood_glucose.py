from pathlib import Path
from datetime import datetime
import logging
import random
import re
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MASTER_CLARITY_PATH = DATA_DIR / "Master Clarity log 1-101 for Dexcom FINAL.xlsx"
FULL_REDCAP_PATH = DATA_DIR / "Full REDCap Data Intervention for Dexcom FINAL.xlsx"
SECONDARY_COHORT_PATH = DATA_DIR / "Final Seconary CGM cohort pull_uncleaned.xlsx"

OUTPUT_DIR = BASE_DIR / "output_lightgbm_30min_subgroups"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_PATH = RUN_DIR / "subgroup_metrics.csv"
OVERALL_TRAJ_PLOT_PATH = RUN_DIR / "overall_glucose_trajectory_30min.png"
SUBGROUP_TRAJ_PLOT_PATH = RUN_DIR / "subgroup_glucose_trajectory_30min.png"
CURRENT_VS_30M_PLOT_PATH = RUN_DIR / "current_vs_30min_glucose_by_group.png"
LOG_PATH = RUN_DIR / "run.log"


# =========================================================
# Logging
# =========================================================
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("glucose_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    out = df.copy()
    out.columns = new_cols
    return out


def first_existing_prefix(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    lower_cols = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        cand_lower = str(cand).strip().lower()
        for lc, orig in lower_cols.items():
            if lc == cand_lower or lc.startswith(cand_lower + "__dup"):
                return orig
    return None


def safe_date_col(df: pd.DataFrame, candidates):
    col = first_existing_prefix(df, candidates)
    if col is not None:
        return col
    for col in df.columns:
        cl = str(col).strip().lower()
        if "date" in cl or "time" in cl:
            return col
    return None


def get_first_column(df: pd.DataFrame, col_name):
    if col_name is None:
        return pd.Series(np.nan, index=df.index)
    if col_name in df.columns:
        data = df[col_name]
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0]
        return data
    matched = first_existing_prefix(df, [col_name])
    if matched is None:
        return pd.Series(np.nan, index=df.index)
    data = df[matched]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def get_numeric_column(df: pd.DataFrame, col_name):
    return pd.to_numeric(get_first_column(df, col_name), errors="coerce")


def safe_to_datetime_series(series: pd.Series) -> pd.Series:
    s = series.copy()

    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean() if len(numeric) > 0 else 0

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    if numeric_ratio > 0.8:
        out = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
        return out

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M",
        "%m/%d/%y %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
    ]

    remaining = s.astype(str).str.strip()
    remaining = remaining.replace({"": np.nan, "nan": np.nan, "NaT": np.nan, "None": np.nan})

    for fmt in formats:
        mask = out.isna() & remaining.notna()
        if not mask.any():
            break
        parsed = pd.to_datetime(remaining[mask], format=fmt, errors="coerce")
        out.loc[mask] = parsed

    mask = out.isna() & remaining.notna()
    if mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                parsed = pd.to_datetime(remaining[mask], format="mixed", errors="coerce")
            except Exception:
                parsed = pd.to_datetime(remaining[mask], errors="coerce")
        out.loc[mask] = parsed

    return out


def get_datetime_column(df: pd.DataFrame, col_name):
    return safe_to_datetime_series(get_first_column(df, col_name))


def find_best_date_column(df: pd.DataFrame, candidates):
    best_col = None
    best_non_null = -1

    checked = []

    for cand in candidates:
        col = first_existing_prefix(df, [cand])
        if col is not None and col not in checked:
            checked.append(col)

    for col in df.columns:
        cl = str(col).strip().lower()
        if ("date" in cl or "time" in cl) and col not in checked:
            checked.append(col)

    for col in checked:
        try:
            parsed = get_datetime_column(df, col)
            non_null = int(parsed.notna().sum())
            if non_null > best_non_null:
                best_non_null = non_null
                best_col = col
        except Exception:
            continue

    return best_col


def yes_no_to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"yes", "checked", "1", "true", "y"}:
        return 1.0
    if s in {"no", "unchecked", "0", "false", "n"}:
        return 0.0
    return np.nan


def duration_to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    mapping = {
        "continuous": 24.0,
        "cyclic": 12.0,
        "nocturnal": 8.0,
        "bolus": 1.0,
    }
    if s in mapping:
        return mapping[s]
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return np.nan


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nrmse_std(y_true, y_pred):
    s = float(np.std(y_true, ddof=0))
    if s == 0:
        return np.nan
    return rmse(y_true, y_pred) / s


def metric_row(name, y_true, y_pred):
    if len(y_true) == 0:
        return {
            "subset": name,
            "n_samples": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "nrmse_std": np.nan,
            "r2": np.nan,
        }
    return {
        "subset": name,
        "n_samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "nrmse_std": nrmse_std(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


# =========================================================
# Load Master Clarity
# =========================================================
def load_master_clarity() -> pd.DataFrame:
    df = pd.read_excel(MASTER_CLARITY_PATH)
    df = ensure_unique_columns(df)

    time_col = first_existing_prefix(df, [
        "imes amp (YYYY-MM-DD hh:mm:ss)",
        "Timestamp",
        "Time Stamp",
    ])
    if time_col is None:
        for c in df.columns:
            lc = str(c).lower()
            if "yyyy-mm-dd" in lc or "imes amp" in lc:
                time_col = c
                break
    if time_col is None:
        raise ValueError("Could not find Master Clarity timestamp column.")

    sub_col = first_existing_prefix(df, ["Sub ID", "Unique Study ID", "subject_id"])
    if sub_col is None:
        raise ValueError("Could not find study ID column in Master Clarity.")

    event_col = first_existing_prefix(df, ["Event Type"])
    glucose_col = first_existing_prefix(df, ["Glucose Value (mg/dL)"])
    if glucose_col is None:
        raise ValueError("Could not find glucose column in Master Clarity.")

    df["subject_id"] = pd.to_numeric(get_first_column(df, sub_col), errors="coerce")
    df["timestamp"] = get_datetime_column(df, time_col)
    df["glucose"] = pd.to_numeric(get_first_column(df, glucose_col), errors="coerce")

    if event_col is not None:
        event_series = get_first_column(df, event_col).astype(str).str.strip()
        df = df[event_series.eq("EGV")].copy()

    df = df[df["subject_id"].notna() & df["timestamp"].notna() & df["glucose"].notna()].copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["date"] = df["timestamp"].dt.normalize()
    df = df.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)
    return df


# =========================================================
# Load REDCap
# =========================================================
def load_redcap(logger):
    df = pd.read_excel(FULL_REDCAP_PATH)
    df = ensure_unique_columns(df)

    id_col = first_existing_prefix(df, ["Unique Study ID"])
    instr_col = first_existing_prefix(df, ["Repeat Instrument"])

    if id_col is None or instr_col is None:
        raise ValueError("Could not find required REDCap columns.")

    baseline = df[get_first_column(df, instr_col).isna()].copy()
    baseline["subject_id"] = pd.to_numeric(get_first_column(baseline, id_col), errors="coerce").astype("Int64")

    static = pd.DataFrame({
        "subject_id": baseline["subject_id"],
        "age": get_numeric_column(baseline, first_existing_prefix(baseline, ["Age", "Age "])),
        "sex": get_first_column(baseline, first_existing_prefix(baseline, ["Sex", "Sex "])),
        "bmi": get_numeric_column(baseline, first_existing_prefix(baseline, ["BMI"])),
        "dm_history": get_first_column(baseline, first_existing_prefix(baseline, ["History of DM?"])).map(yes_no_to_num),
        "admission_glucose": get_numeric_column(baseline, first_existing_prefix(baseline, ["Admission glucose", "Admission glucose "])),
        "a1c": get_numeric_column(baseline, first_existing_prefix(baseline, [
            "Admission HbA1C (if no PRBC transfusion in last 3 months)",
            "Admission HbA1C (if no PRBC transfusion in last 3 months) "
        ])),
        "egfr_admit": get_numeric_column(baseline, first_existing_prefix(baseline, [
            "Admission eGFR (mL/min/1.73M^2)",
            "Admission eGFR (mL/min/1.73M^2) "
        ])),
    })

    sensor_time_cols = [
        c for c in baseline.columns
        if "sensor" in c.lower() and ("date" in c.lower() or "time" in c.lower())
    ]
    sensor_times = baseline[[id_col] + sensor_time_cols].copy() if sensor_time_cols else pd.DataFrame({id_col: get_first_column(baseline, id_col)})
    sensor_times["subject_id"] = pd.to_numeric(get_first_column(sensor_times, id_col), errors="coerce").astype("Int64")

    def pull_daily(instr_name, date_candidates):
        mask = get_first_column(df, instr_col).astype(str).str.strip().eq(instr_name)
        x = df[mask].copy()
        x["subject_id"] = pd.to_numeric(get_first_column(x, id_col), errors="coerce").astype("Int64")

        dt_col = find_best_date_column(x, date_candidates)
        if dt_col is None:
            x["date"] = pd.NaT
        else:
            x["date"] = get_datetime_column(x, dt_col).dt.normalize()
            logger.info(f"{instr_name} uses date column: {dt_col} | non-null dates: {int(x['date'].notna().sum())}")

        return x

    dccu = pull_daily("Daily Clinical Condition and Use", ["Date"])
    meds = pull_daily("Daily Medications", ["Date", "Date "])
    ins = pull_daily("Daily Insulin Dosing", ["Date", "Date .1"])
    labs = pull_daily("Daily Hospital Labs", ["Date", "Date .2"])

    daily_clin = pd.DataFrame({
        "subject_id": dccu["subject_id"],
        "date": dccu["date"],
        "dialysis_flag": get_first_column(dccu, first_existing_prefix(dccu, ["Was the patient receiving dialysis"])).map(yes_no_to_num),
        "ecmo_flag": get_first_column(dccu, first_existing_prefix(dccu, ["Was the patient on ECMO?", "Was the patient on ECMO? "])).map(yes_no_to_num),
        "vent_flag": get_first_column(dccu, first_existing_prefix(dccu, ["Was the patient mechanically ventilated", "Was the patient mechanically ventilated "])).map(yes_no_to_num),
        "supp_o2_flag": get_first_column(dccu, first_existing_prefix(dccu, ["Receiving supplemental O2"])).map(yes_no_to_num),
        "enteral_flag": get_first_column(dccu, first_existing_prefix(dccu, ["Was the patient receiving enteral nutrition"])).map(yes_no_to_num),
        "enteral_duration_hr": get_first_column(dccu, first_existing_prefix(dccu, ["Duration of enteral feed"])).map(duration_to_num),
        "tpn_flag": get_first_column(dccu, first_existing_prefix(dccu, ["Was the patient receiving parenteral nutrition (TPN)?"])).map(yes_no_to_num),
        "tpn_duration_hr": get_first_column(dccu, first_existing_prefix(dccu, ["Duration of TPN"])).map(duration_to_num),
    })

    daily_meds = pd.DataFrame({
        "subject_id": meds["subject_id"],
        "date": meds["date"],
        "pressor_flag": get_first_column(meds, first_existing_prefix(meds, ["Was the patient on pressor support?", "Was the patient on pressor support? "])).map(yes_no_to_num),
        "steroid_flag": get_first_column(meds, first_existing_prefix(meds, ["Did the patient receive steroids?", "Did the patient receive steroids? "])).map(yes_no_to_num),
        "n_pressors": get_numeric_column(meds, first_existing_prefix(meds, ["Number of pressors", "Number of pressors "])),
        "dexamethasone_dose": get_numeric_column(meds, first_existing_prefix(meds, ["Total daily dexamethasone dose"])),
        "prednisone_dose": get_numeric_column(meds, first_existing_prefix(meds, ["Total daily prednisone dose", "Total daily prednisone dose "])),
        "norepi_highest": get_numeric_column(meds, first_existing_prefix(meds, ["Select HIGHEST Norepinephrine dose in this calendar day"])),
        "vasopressin_highest": get_numeric_column(meds, first_existing_prefix(meds, ["Select HIGHEST Vasopressin dose (U/min) in this calendar day"])),
    })

    daily_ins = pd.DataFrame({
        "subject_id": ins["subject_id"],
        "date": ins["date"],
        "iv_insulin_flag": get_first_column(ins, first_existing_prefix(ins, ["Is the patient on IV insulin", "Is the patient on IV insulin "])).map(yes_no_to_num),
        "iv_insulin_units": get_numeric_column(ins, first_existing_prefix(ins, ["Total daily units of IV insulin", "Total daily units of IV insulin "])),
        "subq_insulin_flag": get_first_column(ins, first_existing_prefix(ins, ["Is the patient on subQ insulin", "Is the patient on subQ insulin "])).map(yes_no_to_num),
        "subq_bolus_units": get_numeric_column(ins, first_existing_prefix(ins, ["Total number of units of SubQ bolus insulin received", "Total number of units of SubQ bolus insulin received "])),
        "basal_units": get_numeric_column(ins, first_existing_prefix(ins, ["Total number of units in basal insulin dose", "Total number of units in basal insulin dose "])),
        "nph_flag": get_first_column(ins, first_existing_prefix(ins, ["Is the patient taking NPH or Mixed insulin containing NPH insulin?", "Is the patient taking NPH or Mixed insulin containing NPH insulin? "])).map(yes_no_to_num),
        "nph_units": get_numeric_column(ins, first_existing_prefix(ins, ["How many daily units of NPH insulin is patient receiving", "How many daily units of NPH insulin is patient receiving "])),
    })

    logger.info("===== DEBUG DAILY_INS =====")
    logger.info(f"daily_ins rows: {len(daily_ins)}")
    logger.info(f"daily_ins date non-null: {int(daily_ins['date'].notna().sum())}")
    for col in ["iv_insulin_flag", "subq_insulin_flag", "nph_flag", "basal_units"]:
        logger.info(f"{col} non-null={int(daily_ins[col].notna().sum())}, mean={pd.to_numeric(daily_ins[col], errors='coerce').mean()}")

    daily_labs = pd.DataFrame({
        "subject_id": labs["subject_id"],
        "date": labs["date"],
        "creatinine": get_numeric_column(labs, first_existing_prefix(labs, ["Creatinine (mg/dl)"])),
        "egfr": get_numeric_column(labs, first_existing_prefix(labs, ["eGFR (mL/min/1.73M^2)"])),
        "wbc": get_numeric_column(labs, first_existing_prefix(labs, ["WBC (K/uL)"])),
    })

    daily = (
        daily_clin
        .merge(daily_meds, on=["subject_id", "date"], how="outer")
        .merge(daily_ins, on=["subject_id", "date"], how="outer")
        .merge(daily_labs, on=["subject_id", "date"], how="outer")
        .sort_values(["subject_id", "date"])
        .reset_index(drop=True)
    )

    logger.info("===== DEBUG DAILY AFTER REDCAP MERGE =====")
    for col in ["iv_insulin_flag", "subq_insulin_flag", "nph_flag", "basal_units", "enteral_flag", "tpn_flag"]:
        if col in daily.columns:
            logger.info(f"{col} non-null={int(daily[col].notna().sum())}, mean={pd.to_numeric(daily[col], errors='coerce').mean()}")

    return static, daily, sensor_times


# =========================================================
# Load Secondary workbook
# =========================================================
def load_secondary_daily():
    xls = pd.ExcelFile(SECONDARY_COHORT_PATH)
    daily_frames = []

    if "Steriod" in xls.sheet_names:
        st = pd.read_excel(SECONDARY_COHORT_PATH, sheet_name="Steriod")
        st = ensure_unique_columns(st)
        mrn_col = first_existing_prefix(st, ["MRN"])
        dt_col = safe_date_col(st, ["TAKEN_TIME", "taken_time", "ADM_DATE_TIME"])
        st["mrn"] = get_first_column(st, mrn_col).astype(str).str.strip()
        st["date"] = get_datetime_column(st, dt_col).dt.normalize()
        tmp = pd.DataFrame({"mrn": st["mrn"], "date": st["date"], "sec_steroid_flag": 1.0})
        daily_frames.append(tmp.groupby(["mrn", "date"], as_index=False).max())

    insulin_sheet = "Subcutaneous insulin doses  "
    if insulin_sheet in xls.sheet_names:
        ins = pd.read_excel(SECONDARY_COHORT_PATH, sheet_name=insulin_sheet)
        ins = ensure_unique_columns(ins)
        mrn_col = first_existing_prefix(ins, ["MRN"])
        dt_col = safe_date_col(ins, ["taken_time", "TAKEN_TIME", "ADM_DATE_TIME"])
        med_col = first_existing_prefix(ins, ["MED_NAME", "display_name"])
        action_col = first_existing_prefix(ins, ["Action"])
        dose_col = first_existing_prefix(ins, ["DOSE"])
        route_col = first_existing_prefix(ins, ["route", "ROUTE"])

        ins["mrn"] = get_first_column(ins, mrn_col).astype(str).str.strip()
        ins["date"] = get_datetime_column(ins, dt_col).dt.normalize()
        med = get_first_column(ins, med_col).astype(str).str.upper().fillna("") if med_col else pd.Series("", index=ins.index)
        route = get_first_column(ins, route_col).astype(str).str.upper().fillna("") if route_col else pd.Series("", index=ins.index)
        act = get_first_column(ins, action_col).astype(str).str.upper().fillna("") if action_col else pd.Series("", index=ins.index)
        dose = get_numeric_column(ins, dose_col) if dose_col else pd.Series(np.nan, index=ins.index)

        tmp = pd.DataFrame({
            "mrn": ins["mrn"],
            "date": ins["date"],
            "sec_any_insulin_flag": 1.0,
            "sec_iv_insulin_flag": (
                route.str.contains("INTRAVENOUS|IV", na=False)
                | med.str.contains("INFUSION", na=False)
                | act.isin(["RATE VERIFY", "RATE CHANGE", "NEW BAG", "RESTARTED", "PUMP ASSOCIATION"])
            ).astype(float),
            "sec_subq_insulin_flag": (
                (~route.str.contains("INTRAVENOUS|IV", na=False))
                & med.str.contains("INSULIN", na=False)
            ).astype(float),
            "sec_insulin_total_dose": dose,
        })
        agg = tmp.groupby(["mrn", "date"], as_index=False).agg({
            "sec_any_insulin_flag": "max",
            "sec_iv_insulin_flag": "max",
            "sec_subq_insulin_flag": "max",
            "sec_insulin_total_dose": "sum",
        })
        daily_frames.append(agg)

    if "DIETARY PLACE A TUBE FOOD " in xls.sheet_names:
        d = pd.read_excel(SECONDARY_COHORT_PATH, sheet_name="DIETARY PLACE A TUBE FOOD ")
        d = ensure_unique_columns(d)
        mrn_col = first_existing_prefix(d, ["MRN"])
        dt_col = safe_date_col(d, ["ORDERING_DATE", "ADM_DATE_TIME"])
        d["mrn"] = get_first_column(d, mrn_col).astype(str).str.strip()
        d["date"] = get_datetime_column(d, dt_col).dt.normalize()
        tmp = pd.DataFrame({"mrn": d["mrn"], "date": d["date"], "sec_tube_feed_flag": 1.0})
        daily_frames.append(tmp.groupby(["mrn", "date"], as_index=False).max())

    if not daily_frames:
        return pd.DataFrame(columns=["mrn", "date"])

    sec_daily = daily_frames[0]
    for x in daily_frames[1:]:
        sec_daily = sec_daily.merge(x, on=["mrn", "date"], how="outer")

    sec_daily = sec_daily.sort_values(["mrn", "date"]).reset_index(drop=True)
    return sec_daily


# =========================================================
# Matching
# =========================================================
def build_redcap_fingerprint(daily: pd.DataFrame, clarity: pd.DataFrame, sensor_times: pd.DataFrame):
    keep_cols = {
        "subject_id", "date", "steroid_flag", "iv_insulin_flag", "subq_insulin_flag",
        "enteral_flag", "tpn_flag"
    }
    fp = daily[[c for c in daily.columns if c in keep_cols]].copy()
    fp = fp.rename(columns={
        "steroid_flag": "red_steroid_flag",
        "iv_insulin_flag": "red_iv_insulin_flag",
        "subq_insulin_flag": "red_subq_insulin_flag",
        "enteral_flag": "red_enteral_flag",
        "tpn_flag": "red_tpn_flag",
    })

    clarity_span = clarity.groupby("subject_id").agg(
        cgm_start=("date", "min"),
        cgm_end=("date", "max"),
    ).reset_index()

    sensor_cols = [c for c in sensor_times.columns if c != "subject_id"]
    sensor_long = []
    for c in sensor_cols:
        s = get_datetime_column(sensor_times, c)
        sensor_long.append(pd.DataFrame({"subject_id": sensor_times["subject_id"], "sensor_time": s}))

    if sensor_long:
        sensor_long = pd.concat(sensor_long, ignore_index=True)
        sensor_long = sensor_long[sensor_long["sensor_time"].notna()]
        sensor_span = sensor_long.groupby("subject_id").agg(
            sensor_start=("sensor_time", "min"),
            sensor_end=("sensor_time", "max"),
        ).reset_index()
        sensor_span["sensor_start"] = pd.to_datetime(sensor_span["sensor_start"]).dt.normalize()
        sensor_span["sensor_end"] = pd.to_datetime(sensor_span["sensor_end"]).dt.normalize()
    else:
        sensor_span = pd.DataFrame(columns=["subject_id", "sensor_start", "sensor_end"])

    subj_meta = clarity_span.merge(sensor_span, on="subject_id", how="left")
    return fp, subj_meta


def build_secondary_fingerprint(sec_daily: pd.DataFrame):
    fp = sec_daily.copy()
    mrn_meta = fp.groupby("mrn").agg(
        sec_start=("date", "min"),
        sec_end=("date", "max"),
    ).reset_index()
    return fp, mrn_meta


def compute_pair_score(red_fp_one: pd.DataFrame, sec_fp_one: pd.DataFrame, red_meta_one: pd.Series, sec_meta_one: pd.Series):
    red_start = pd.to_datetime(red_meta_one.get("sensor_start") if pd.notna(red_meta_one.get("sensor_start")) else red_meta_one.get("cgm_start"))
    red_end = pd.to_datetime(red_meta_one.get("sensor_end") if pd.notna(red_meta_one.get("sensor_end")) else red_meta_one.get("cgm_end"))
    sec_start = pd.to_datetime(sec_meta_one.get("sec_start"))
    sec_end = pd.to_datetime(sec_meta_one.get("sec_end"))

    if pd.isna(red_start) or pd.isna(red_end) or pd.isna(sec_start) or pd.isna(sec_end):
        date_overlap_score = 0.0
    else:
        overlap_days = max(0, (min(red_end, sec_end) - max(red_start, sec_start)).days + 1)
        union_days = max(1, (max(red_end, sec_end) - min(red_start, sec_start)).days + 1)
        date_overlap_score = overlap_days / union_days

    merged = red_fp_one.merge(sec_fp_one, on="date", how="inner")
    if len(merged) == 0:
        return {
            "score": 0.15 * date_overlap_score,
            "overlap_days": 0,
            "date_overlap_score": date_overlap_score,
            "feature_agreement": np.nan,
        }

    feature_pairs = [
        ("red_steroid_flag", "sec_steroid_flag"),
        ("red_iv_insulin_flag", "sec_iv_insulin_flag"),
        ("red_subq_insulin_flag", "sec_subq_insulin_flag"),
        ("red_enteral_flag", "sec_tube_feed_flag"),
    ]

    agreements = []
    for a, b in feature_pairs:
        if a in merged.columns and b in merged.columns:
            x = merged[a]
            y = merged[b]
            ok = x.notna() & y.notna()
            if ok.sum() > 0:
                agreements.append((x[ok].round(0) == y[ok].round(0)).mean())

    feature_agreement = float(np.mean(agreements)) if agreements else np.nan
    overlap_days = int(merged["date"].nunique())
    feature_component = 0.0 if np.isnan(feature_agreement) else feature_agreement
    support_bonus = min(1.0, overlap_days / 5.0)
    score = 0.60 * feature_component + 0.25 * date_overlap_score + 0.15 * support_bonus

    return {
        "score": float(score),
        "overlap_days": overlap_days,
        "date_overlap_score": float(date_overlap_score),
        "feature_agreement": feature_agreement,
    }


def probabilistic_match_subjects(red_daily: pd.DataFrame, sec_daily: pd.DataFrame, clarity: pd.DataFrame, sensor_times: pd.DataFrame):
    if sec_daily.empty:
        return pd.DataFrame(columns=["subject_id", "mrn", "match_score", "match_confidence"]), pd.DataFrame()

    red_fp, red_meta = build_redcap_fingerprint(red_daily, clarity, sensor_times)
    sec_fp, sec_meta = build_secondary_fingerprint(sec_daily)

    red_ids = list(red_meta["subject_id"].dropna().astype(int).unique())
    mrns = list(sec_meta["mrn"].dropna().astype(str).unique())

    red_meta_idx = red_meta.set_index("subject_id")
    sec_meta_idx = sec_meta.set_index("mrn")

    diagnostics = []
    for sid in red_ids:
        red_one = red_fp[red_fp["subject_id"] == sid].copy()
        red_meta_one = red_meta_idx.loc[sid]
        for mrn in mrns:
            sec_one = sec_fp[sec_fp["mrn"] == mrn].copy()
            sec_meta_one = sec_meta_idx.loc[mrn]
            score_obj = compute_pair_score(red_one, sec_one, red_meta_one, sec_meta_one)
            diagnostics.append({"subject_id": sid, "mrn": mrn, **score_obj})

    diag = pd.DataFrame(diagnostics).sort_values(["score", "overlap_days"], ascending=False)

    matches = []
    used_sids = set()
    used_mrns = set()

    for sid, g in diag.groupby("subject_id", sort=False):
        g = g.sort_values(["score", "overlap_days"], ascending=False).reset_index(drop=True)
        if len(g) == 0:
            continue
        row = g.iloc[0]
        sid = int(row["subject_id"])
        mrn = str(row["mrn"])
        score = float(row["score"])
        overlap_days = int(row["overlap_days"])

        if sid not in used_sids and mrn not in used_mrns and score >= 0.65 and overlap_days >= 2:
            used_sids.add(sid)
            used_mrns.add(mrn)
            confidence = "high" if score >= 0.80 else "medium"
            matches.append({
                "subject_id": sid,
                "mrn": mrn,
                "match_score": score,
                "match_confidence": confidence,
            })

    matches = pd.DataFrame(matches).sort_values("match_score", ascending=False).reset_index(drop=True)
    return matches, diag


# =========================================================
# Time-series features
# =========================================================
def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["subject_id", "timestamp"]).copy()

    lag_steps = [1, 2, 3, 6, 12, 18, 24, 30, 36]
    for lag in lag_steps:
        df[f"glucose_lag_{lag}"] = df.groupby("subject_id")["glucose"].shift(lag)
        df[f"dt_lag_{lag}_min"] = (
            df["timestamp"] - df.groupby("subject_id")["timestamp"].shift(lag)
        ).dt.total_seconds() / 60.0

    prev = df.groupby("subject_id")["glucose"].shift(1)
    for w in [3, 6, 12, 36]:
        g = prev.groupby(df["subject_id"])
        df[f"roll_mean_{w}"] = g.rolling(window=w, min_periods=max(2, w // 3)).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = g.rolling(window=w, min_periods=max(2, w // 3)).std().reset_index(level=0, drop=True)

    df["delta_1"] = df["glucose"] - df["glucose_lag_1"]
    df["delta_6"] = df["glucose"] - df["glucose_lag_6"]
    df["delta_12"] = df["glucose"] - df["glucose_lag_12"]

    df["slope_1"] = df["delta_1"] / df["dt_lag_1_min"]
    df["slope_6"] = df["delta_6"] / df["dt_lag_6_min"]
    df["slope_12"] = df["delta_12"] / df["dt_lag_12_min"]

    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["target_glucose_30m"] = df.groupby("subject_id")["glucose"].shift(-6)
    future_ts_30 = df.groupby("subject_id")["timestamp"].shift(-6)
    df["target_gap_min"] = (future_ts_30 - df["timestamp"]).dt.total_seconds() / 60.0

    return df


# =========================================================
# Dataset summary logging
# =========================================================
def log_dataset_summary(logger, clarity, static, daily, sec_daily):
    logger.info("===== DATASET SUMMARY =====")
    logger.info(f"Master Clarity rows: {len(clarity)}")
    logger.info(f"Master Clarity subjects: {clarity['subject_id'].nunique()}")
    logger.info(f"Master Clarity time span: {clarity['timestamp'].min()} -> {clarity['timestamp'].max()}")
    logger.info(f"Glucose mean/std: {clarity['glucose'].mean():.2f} / {clarity['glucose'].std():.2f}")
    logger.info(f"Glucose min/max: {clarity['glucose'].min():.2f} / {clarity['glucose'].max():.2f}")

    logger.info(f"Baseline rows: {len(static)}")
    logger.info(f"Age mean: {static['age'].mean():.2f}")
    logger.info(f"BMI mean: {static['bmi'].mean():.2f}")
    logger.info(f"DM history rate: {static['dm_history'].mean(skipna=True):.4f}")
    logger.info(f"Admission eGFR mean: {static['egfr_admit'].mean(skipna=True):.2f}")

    logger.info(f"Daily REDCap rows: {len(daily)}")
    for col in ["enteral_flag", "tpn_flag", "steroid_flag", "pressor_flag", "iv_insulin_flag", "subq_insulin_flag", "dialysis_flag"]:
        if col in daily.columns:
            logger.info(f"{col} rate: {daily[col].mean(skipna=True):.4f}")

    logger.info(f"Secondary daily rows: {len(sec_daily)}")
    if not sec_daily.empty and "mrn" in sec_daily.columns:
        logger.info(f"Secondary unique MRNs: {sec_daily['mrn'].nunique()}")


# =========================================================
# Plot helpers
# =========================================================
def add_intervention_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    nutrition_any = (
        out.get("enteral_flag", pd.Series(index=out.index)).fillna(0).eq(1)
        | out.get("tpn_flag", pd.Series(index=out.index)).fillna(0).eq(1)
    )
    insulin_any = (
        out.get("iv_insulin_flag", pd.Series(index=out.index)).fillna(0).eq(1)
        | out.get("subq_insulin_flag", pd.Series(index=out.index)).fillna(0).eq(1)
        | out.get("nph_flag", pd.Series(index=out.index)).fillna(0).eq(1)
        | out.get("basal_units", pd.Series(index=out.index)).fillna(0).gt(0)
    )

    out["group_label"] = "other"
    out.loc[(~nutrition_any) & (~insulin_any), "group_label"] = "no_intervention"
    out.loc[nutrition_any & (~insulin_any), "group_label"] = "nutrition_only"
    out.loc[(~nutrition_any) & insulin_any, "group_label"] = "insulin_only"
    out.loc[nutrition_any & insulin_any, "group_label"] = "both_nutrition_and_insulin"

    return out


def make_overall_glucose_trajectory_plot(plot_source_df: pd.DataFrame, logger):
    plot_cols = [
        ("glucose_lag_36", -180),
        ("glucose_lag_30", -150),
        ("glucose_lag_24", -120),
        ("glucose_lag_18", -90),
        ("glucose_lag_12", -60),
        ("glucose_lag_6", -30),
        ("glucose", 0),
        ("target_glucose_30m", 30),
    ]

    available = [(c, t) for c, t in plot_cols if c in plot_source_df.columns]
    if len(available) < 3:
        logger.info("Skip overall trajectory plot: not enough trajectory columns.")
        return

    plot_df = pd.DataFrame({
        "minute": [t for _, t in available],
        "mean_glucose": [plot_source_df[c].mean(skipna=True) for c, _ in available],
    }).sort_values("minute")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_df["minute"], plot_df["mean_glucose"], marker="o")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time relative to current point (min)")
    ax.set_ylabel("Average glucose (mg/dL)")
    ax.set_title("Overall Average Glucose Trajectory (All Rows, 30-min target)")
    fig.tight_layout()
    fig.savefig(OVERALL_TRAJ_PLOT_PATH, dpi=200)
    plt.close(fig)

    logger.info("===== OVERALL GLUCOSE TRAJECTORY =====")
    logger.info("\n" + plot_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    logger.info(f"Overall trajectory plot saved to: {OVERALL_TRAJ_PLOT_PATH}")


def make_subgroup_glucose_trajectory_plot(plot_source_df: pd.DataFrame, logger):
    df = add_intervention_group_columns(plot_source_df)

    plot_cols = [
        ("glucose_lag_36", -180),
        ("glucose_lag_30", -150),
        ("glucose_lag_24", -120),
        ("glucose_lag_18", -90),
        ("glucose_lag_12", -60),
        ("glucose_lag_6", -30),
        ("glucose", 0),
        ("target_glucose_30m", 30),
    ]
    available = [(c, t) for c, t in plot_cols if c in df.columns]
    if len(available) < 3:
        logger.info("Skip subgroup trajectory plot: not enough trajectory columns.")
        return

    groups = [
        "no_intervention",
        "nutrition_only",
        "insulin_only",
        "both_nutrition_and_insulin",
    ]

    logger.info("===== SUBGROUP TRAJECTORY GROUP COUNTS =====")
    for g in groups:
        logger.info(f"{g}: {(df['group_label'] == g).sum()} rows")

    fig, ax = plt.subplots(figsize=(9, 6))
    summary_tables = []

    for g in groups:
        sub = df[df["group_label"] == g].copy()
        if len(sub) == 0:
            continue

        traj = pd.DataFrame({
            "minute": [t for _, t in available],
            "mean_glucose": [sub[c].mean(skipna=True) for c, _ in available],
        }).sort_values("minute")
        traj["group"] = g
        summary_tables.append(traj)

        ax.plot(
            traj["minute"],
            traj["mean_glucose"],
            marker="o",
            linewidth=2,
            label=f"{g} (n={len(sub)})"
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time relative to current point (min)")
    ax.set_ylabel("Average glucose (mg/dL)")
    ax.set_title("Subgroup Average Glucose Trajectory (All Rows, 30-min target)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SUBGROUP_TRAJ_PLOT_PATH, dpi=200)
    plt.close(fig)

    if summary_tables:
        summary_df = pd.concat(summary_tables, ignore_index=True)
        logger.info("===== SUBGROUP GLUCOSE TRAJECTORY =====")
        logger.info("\n" + summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    logger.info(f"Subgroup trajectory plot saved to: {SUBGROUP_TRAJ_PLOT_PATH}")


def make_current_vs_30m_plot(plot_source_df: pd.DataFrame, logger, min_bin_samples: int = 5):
    df = add_intervention_group_columns(plot_source_df).copy()

    plot_df = df[df["target_glucose_30m"].notna() & df["glucose"].notna()].copy()
    plot_df = plot_df[plot_df["target_gap_min"].between(25, 35)].copy()

    if len(plot_df) == 0:
        logger.info("Skip current vs 30m plot: no valid rows after filtering.")
        return

    bins = np.arange(40, 401, 20)
    plot_df["glucose_bin"] = pd.cut(plot_df["glucose"], bins=bins, right=False)

    groups = [
        "no_intervention",
        "nutrition_only",
        "insulin_only",
        "both_nutrition_and_insulin",
    ]

    logger.info("===== CURRENT VS 30M GROUP COUNTS =====")
    for g in groups:
        logger.info(f"{g}: {(plot_df['group_label'] == g).sum()} rows")

    fig, ax = plt.subplots(figsize=(9, 6))
    summary_tables = []

    for g in groups:
        sub = plot_df[plot_df["group_label"] == g].copy()
        if len(sub) == 0:
            continue

        agg = (
            sub.groupby("glucose_bin", observed=False)
            .agg(
                current_glucose_mean=("glucose", "mean"),
                glucose_30m_mean=("target_glucose_30m", "mean"),
                n_samples=("glucose", "size"),
            )
            .reset_index()
        )

        agg = agg[agg["n_samples"] >= min_bin_samples].copy()
        agg = agg.dropna(subset=["current_glucose_mean", "glucose_30m_mean"])

        if len(agg) < 2:
            continue

        agg["group"] = g
        summary_tables.append(agg)

        ax.plot(
            agg["current_glucose_mean"],
            agg["glucose_30m_mean"],
            marker="o",
            linewidth=2,
            label=f"{g} (bin_n>={min_bin_samples})"
        )

    ax.set_xlabel("Current glucose (mg/dL)")
    ax.set_ylabel("30-min later glucose (mg/dL)")
    ax.set_title("Current Glucose vs 30-Min Later Glucose by Group (All Rows)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CURRENT_VS_30M_PLOT_PATH, dpi=200)
    plt.close(fig)

    if summary_tables:
        summary_df = pd.concat(summary_tables, ignore_index=True)
        logger.info("===== CURRENT GLUCOSE VS 30-MIN LATER GLUCOSE =====")
        logger.info("\n" + summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    logger.info(f"Current vs 30m plot saved to: {CURRENT_VS_30M_PLOT_PATH}")


# =========================================================
# Build model table
# =========================================================
def build_model_table(logger):
    clarity = load_master_clarity()
    static, daily, sensor_times = load_redcap(logger)
    sec_daily = load_secondary_daily()

    log_dataset_summary(logger, clarity, static, daily, sec_daily)

    matches, diag = probabilistic_match_subjects(daily, sec_daily, clarity, sensor_times)

    logger.info("===== MATCHING SUMMARY =====")
    logger.info(f"Total candidate pair rows: {len(diag)}")
    logger.info(f"Accepted matched pairs: {len(matches)}")
    if not matches.empty:
        logger.info(f"High-confidence matches: {(matches['match_confidence'] == 'high').sum()}")
        logger.info(f"Medium-confidence matches: {(matches['match_confidence'] == 'medium').sum()}")
    total_subjects = int(clarity['subject_id'].nunique())
    matched_subjects = int(matches['subject_id'].nunique()) if not matches.empty else 0
    logger.info(f"Matched subjects: {matched_subjects}")
    logger.info(f"Unmatched subjects: {total_subjects - matched_subjects}")

    if not matches.empty:
        sec_mapped = matches[["subject_id", "mrn", "match_score", "match_confidence"]].merge(sec_daily, on="mrn", how="left")
        daily_all = daily.merge(sec_mapped.drop(columns=["mrn"]), on=["subject_id", "date"], how="left")
    else:
        daily_all = daily.copy()
        daily_all["match_score"] = np.nan
        daily_all["match_confidence"] = np.nan

    merged = (
        clarity
        .merge(daily_all, on=["subject_id", "date"], how="left")
        .merge(static, on="subject_id", how="left")
    )

    logger.info("===== DEBUG MERGED BEFORE FEATURE ENGINEERING =====")
    for col in ["iv_insulin_flag", "subq_insulin_flag", "nph_flag", "basal_units", "enteral_flag", "tpn_flag"]:
        if col in merged.columns:
            logger.info(f"{col} non-null={int(merged[col].notna().sum())}, mean={pd.to_numeric(merged[col], errors='coerce').mean()}")

    merged = add_time_series_features(merged)

    merged = merged[
        merged["target_gap_min"].between(25, 35)
        & merged["glucose_lag_6"].notna()
        & merged["dt_lag_6_min"].between(20, 40)
        & merged["target_glucose_30m"].notna()
    ].copy()

    logger.info("===== MODEL TABLE SUMMARY =====")
    logger.info(f"Model table rows: {len(merged)}")
    logger.info(f"Model table subjects: {merged['subject_id'].nunique()}")
    logger.info(f"Target mean/std: {merged['target_glucose_30m'].mean():.2f} / {merged['target_glucose_30m'].std():.2f}")

    return merged


# =========================================================
# Train and evaluate
# =========================================================
def train_and_evaluate(df: pd.DataFrame, logger):
    feature_cols = [
        "glucose",
        "glucose_lag_1", "glucose_lag_2", "glucose_lag_3", "glucose_lag_6", "glucose_lag_12", "glucose_lag_24", "glucose_lag_36",
        "dt_lag_1_min", "dt_lag_6_min", "dt_lag_12_min",
        "delta_1", "delta_6", "delta_12",
        "slope_1", "slope_6", "slope_12",
        "roll_mean_3", "roll_std_3", "roll_mean_6", "roll_std_6", "roll_mean_12", "roll_std_12", "roll_mean_36", "roll_std_36",
        "hour_sin", "hour_cos", "day_of_week", "is_weekend",

        "age", "sex", "bmi", "dm_history", "admission_glucose", "a1c", "egfr_admit",

        "dialysis_flag", "ecmo_flag", "vent_flag", "supp_o2_flag",
        "enteral_flag", "enteral_duration_hr", "tpn_flag", "tpn_duration_hr",
        "pressor_flag", "n_pressors", "steroid_flag",
        "dexamethasone_dose", "prednisone_dose",
        "iv_insulin_flag", "iv_insulin_units",
        "subq_insulin_flag", "subq_bolus_units", "basal_units", "nph_flag", "nph_units",
        "creatinine", "egfr", "wbc",

        "match_score",
        "sec_steroid_flag", "sec_any_insulin_flag", "sec_iv_insulin_flag", "sec_subq_insulin_flag", "sec_insulin_total_dose", "sec_tube_feed_flag",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    work = df.copy()
    if "sex" in work.columns:
        work["sex"] = work["sex"].astype("category")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss1.split(work, groups=work["subject_id"]))
    train_df = work.iloc[train_idx].copy()
    test_df = work.iloc[test_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    tr_idx, val_idx = next(gss2.split(train_df, groups=train_df["subject_id"]))
    tr_df = train_df.iloc[tr_idx].copy()
    val_df = train_df.iloc[val_idx].copy()

    X_tr = tr_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_tr = tr_df["target_glucose_30m"].astype(float)
    y_val = val_df["target_glucose_30m"].astype(float)
    y_test = test_df["target_glucose_30m"].astype(float)

    for X in [X_tr, X_val, X_test]:
        if "sex" in X.columns:
            X["sex"] = X["sex"].astype("category")

    logger.info("===== TRAIN / VAL / TEST SPLIT =====")
    logger.info(f"Total subjects: {work['subject_id'].nunique()}")
    logger.info(f"Train subjects: {tr_df['subject_id'].nunique()}, rows: {len(tr_df)}")
    logger.info(f"Val subjects: {val_df['subject_id'].nunique()}, rows: {len(val_df)}")
    logger.info(f"Test subjects: {test_df['subject_id'].nunique()}, rows: {len(test_df)}")
    logger.info(f"Number of features: {len(feature_cols)}")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(50)],
        categorical_feature=[c for c in ["sex"] if c in feature_cols],
    )

    pred = model.predict(X_test, num_iteration=model.best_iteration_)

    make_overall_glucose_trajectory_plot(df, logger)
    make_subgroup_glucose_trajectory_plot(df, logger)
    make_current_vs_30m_plot(df, logger, min_bin_samples=5)

    rows = []

    def add_subset(name, mask, min_n=None):
        idx = np.where(mask.fillna(False).values)[0]
        if min_n is not None and len(idx) < min_n:
            rows.append(metric_row(name, np.array([]), np.array([])))
        else:
            rows.append(metric_row(name, y_test.iloc[idx].values, pred[idx]))

    nutrition_any = (
        test_df.get("enteral_flag", pd.Series(index=test_df.index)).fillna(0).eq(1)
        | test_df.get("tpn_flag", pd.Series(index=test_df.index)).fillna(0).eq(1)
    )
    insulin_any = (
        test_df.get("iv_insulin_flag", pd.Series(index=test_df.index)).fillna(0).eq(1)
        | test_df.get("subq_insulin_flag", pd.Series(index=test_df.index)).fillna(0).eq(1)
        | test_df.get("nph_flag", pd.Series(index=test_df.index)).fillna(0).eq(1)
        | test_df.get("basal_units", pd.Series(index=test_df.index)).fillna(0).gt(0)
    )

    logger.info("===== DEBUG INSULIN / NUTRITION COUNTS (TEST) =====")
    for col in ["iv_insulin_flag", "subq_insulin_flag", "nph_flag", "basal_units", "enteral_flag", "tpn_flag"]:
        if col in test_df.columns:
            logger.info(f"{col} non-null={test_df[col].notna().sum()}, mean={pd.to_numeric(test_df[col], errors='coerce').mean()}")
        else:
            logger.info(f"{col} NOT FOUND")

    logger.info(f"insulin_any true count: {insulin_any.sum()}")
    logger.info(f"nutrition_any true count: {nutrition_any.sum()}")
    logger.info(f"both true count: {(insulin_any & nutrition_any).sum()}")
    logger.info(f"insulin only count: {(insulin_any & (~nutrition_any)).sum()}")

    add_subset("overall", pd.Series(True, index=test_df.index))
    add_subset("no_intervention", (~nutrition_any) & (~insulin_any))
    add_subset("nutrition_only", nutrition_any & (~insulin_any))
    add_subset("insulin_only", insulin_any & (~nutrition_any))
    add_subset("both_nutrition_and_insulin", insulin_any & nutrition_any)

    add_subset("enteral", test_df.get("enteral_flag", pd.Series(index=test_df.index)).fillna(0).eq(1))
    add_subset("tpn", test_df.get("tpn_flag", pd.Series(index=test_df.index)).fillna(0).eq(1))
    add_subset("iv_insulin", test_df.get("iv_insulin_flag", pd.Series(index=test_df.index)).fillna(0).eq(1))
    add_subset("subq_insulin", test_df.get("subq_insulin_flag", pd.Series(index=test_df.index)).fillna(0).eq(1))
    add_subset("basal_insulin", test_df.get("basal_units", pd.Series(index=test_df.index)).fillna(0).gt(0))
    add_subset(
        "nph_insulin",
        test_df.get("nph_flag", pd.Series(index=test_df.index)).fillna(0).eq(1)
        | test_df.get("nph_units", pd.Series(index=test_df.index)).fillna(0).gt(0),
    )

    add_subset("steroid_yes", test_df.get("steroid_flag", pd.Series(index=test_df.index)).fillna(0).eq(1))
    add_subset("steroid_no", test_df.get("steroid_flag", pd.Series(index=test_df.index)).fillna(0).eq(0))
    add_subset("vasopressor_yes", test_df.get("pressor_flag", pd.Series(index=test_df.index)).fillna(0).eq(1))
    add_subset("vasopressor_no", test_df.get("pressor_flag", pd.Series(index=test_df.index)).fillna(0).eq(0))

    add_subset("reduced_renal_function_egfr_lt_60", test_df.get("egfr_admit", pd.Series(index=test_df.index)).lt(60))
    add_subset("high_creatinine_ge_2", test_df.get("creatinine", pd.Series(index=test_df.index)).ge(2.0))
    add_subset("dialysis", test_df.get("dialysis_flag", pd.Series(index=test_df.index)).fillna(0).eq(1))

    add_subset("hypoglycemia_lt_70", test_df["target_glucose_30m"].lt(70))
    add_subset("in_range_70_180", test_df["target_glucose_30m"].between(70, 180, inclusive="both"))
    add_subset("hyperglycemia_gt_180", test_df["target_glucose_30m"].gt(180))

    if "roll_std_12" in test_df.columns:
        q75_std12 = test_df["roll_std_12"].quantile(0.75)
        add_subset("high_variability", test_df["roll_std_12"].ge(q75_std12))
    else:
        rows.append(metric_row("high_variability", np.array([]), np.array([])))

    if "slope_6" in test_df.columns:
        add_subset("rapid_rise", test_df["slope_6"].ge(2.0))
        add_subset("rapid_fall", test_df["slope_6"].le(-2.0))
    else:
        rows.append(metric_row("rapid_rise", np.array([]), np.array([])))
        rows.append(metric_row("rapid_fall", np.array([]), np.array([])))

    seg_df = pd.DataFrame(rows)
    seg_df = seg_df[(seg_df["subset"] == "overall") | (seg_df["n_samples"] > 0)].copy()
    seg_df = seg_df[["subset", "n_samples", "mae", "rmse", "nrmse_std", "r2"]]

    logger.info("===== SUBSET METRICS TABLE =====")
    logger.info("\n" + seg_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    return seg_df


# =========================================================
# Main
# =========================================================
def main():
    logger = setup_logger(LOG_PATH)
    set_seed(42)

    logger.info("Starting 30-minute glucose regression pipeline")
    model_table = build_model_table(logger)
    seg_df = train_and_evaluate(model_table, logger)
    seg_df.to_csv(OUTPUT_CSV_PATH, index=False)

    logger.info(f"Output file: {OUTPUT_CSV_PATH}")
    logger.info(f"Overall trajectory plot: {OVERALL_TRAJ_PLOT_PATH}")
    logger.info(f"Subgroup trajectory plot: {SUBGROUP_TRAJ_PLOT_PATH}")
    logger.info(f"Current vs 30m plot: {CURRENT_VS_30M_PLOT_PATH}")
    logger.info(f"Log file: {LOG_PATH}")
    logger.info("Finished")


if __name__ == "__main__":
    main()