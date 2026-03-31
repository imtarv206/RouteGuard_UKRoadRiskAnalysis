"""
preprocess.py — Data Preprocessing for RouteGuard (UK Road Safety Dataset)
===========================================================================
Dataset: https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles
Files expected:
  - Accident_Information.csv
  - Vehicle_Information.csv   (optional, for enrichment)

Output:
  - Accident_Information_clean.csv  (ready for RiskAnalyzer)
  - preprocessing_report.txt        (quality summary)

Usage:
  python preprocess.py
  python preprocess.py --accident path/to/Accident_Information.csv
  python preprocess.py --accident Accident_Information.csv --vehicle Vehicle_Information.csv
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# Columns that RouteGuard REQUIRES
REQUIRED_COLS = ["Latitude", "Longitude", "Accident_Severity"]

# Columns that RouteGuard USES (if present)
USED_COLS = [
    "Latitude", "Longitude", "Accident_Severity",
    "Number_of_Casualties", "Number_of_Vehicles",
    "Time", "Date", "Day_of_Week",
    "Weather_Conditions", "Road_Surface_Conditions",
    "Road_Type", "Speed_limit", "Light_Conditions",
    "Urban_or_Rural_Area", "Local_Authority_(District)",
]

# UK bounding box (generous)
LAT_MIN, LAT_MAX = 49.5, 61.0
LON_MIN, LON_MAX = -8.5, 2.0

# Day-of-week normalisation: raw value → standard English name
DAY_MAP = {
    # Numeric codes found in older releases
    "1": "Sunday", "2": "Monday", "3": "Tuesday", "4": "Wednesday",
    "5": "Thursday", "6": "Friday", "7": "Saturday",
    # Abbreviated
    "sun": "Sunday", "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
    "thu": "Thursday", "fri": "Friday", "sat": "Saturday",
    # Already full names – kept as-is by the normaliser
}

SEVERITY_VALID = {1, 2, 3}   # 1=Fatal, 2=Serious, 3=Slight


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)


def normalise_day(val: str) -> str:
    """Return a canonical day name, or the original value if unrecognised."""
    v = str(val).strip().lower()
    return DAY_MAP.get(v, val)


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

def load_accidents(path: str) -> pd.DataFrame:
    section("1. LOADING Accident_Information.csv")
    print(f"   Path : {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    print(f"   Rows : {len(df):,}")
    print(f"   Cols : {len(df.columns)}")
    print(f"   Columns:\n{textwrap.fill(', '.join(df.columns), width=70, initial_indent='     ')}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP-BY-STEP CLEANING
# ─────────────────────────────────────────────────────────────

def step_required_cols(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """Abort early if core columns are missing."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        msg = f"[FATAL] Missing required columns: {missing}"
        report.append(msg)
        print(f"\n{msg}")
        sys.exit(1)
    report.append(f"Required columns present: {REQUIRED_COLS}")
    return df


def step_drop_duplicates(df: pd.DataFrame, report: list) -> pd.DataFrame:
    before = len(df)
    # Use Accident_Index as unique key if available
    if "Accident_Index" in df.columns:
        df = df.drop_duplicates(subset=["Accident_Index"])
    else:
        df = df.drop_duplicates()
    removed = before - len(df)
    report.append(f"Duplicates removed      : {removed:,}")
    print(f"   Duplicates removed : {removed:,}  →  {len(df):,} rows remain")
    return df


def step_clean_coordinates(df: pd.DataFrame, report: list) -> pd.DataFrame:
    before = len(df)

    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # Drop null coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Drop coordinates outside UK bounding box
    in_uk = (
        df["Latitude"].between(LAT_MIN, LAT_MAX) &
        df["Longitude"].between(LON_MIN, LON_MAX)
    )
    df = df[in_uk]

    removed = before - len(df)
    report.append(f"Invalid/OOB coords removed : {removed:,}")
    print(f"   Invalid coordinates removed : {removed:,}  →  {len(df):,} rows remain")
    return df


def step_clean_severity(df: pd.DataFrame, report: list) -> pd.DataFrame:
    # Map text labels to numeric if present (e.g. "Fatal" → 1)
    text_to_num = {
        "fatal": 1, "serious": 2, "slight": 3,
    }
    raw = df["Accident_Severity"].astype(str).str.strip().str.lower()
    mapped = raw.map(text_to_num)

    # If mapping worked for most rows, use it; otherwise try numeric parse
    if mapped.notna().sum() > len(df) * 0.5:
        df["Accident_Severity"] = mapped
        print(f"   Severity: mapped text labels (Fatal/Serious/Slight) → 1/2/3")
        report.append("Severity: mapped text labels → numeric")
    else:
        df["Accident_Severity"] = pd.to_numeric(df["Accident_Severity"], errors="coerce")

    invalid_mask = ~df["Accident_Severity"].isin(SEVERITY_VALID)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        df.loc[invalid_mask, "Accident_Severity"] = 3   # default to Slight
        report.append(f"Severity out-of-range (→3) : {invalid_count:,}")
        print(f"   Severity out-of-range (defaulted to 3): {invalid_count:,}")

    df["Accident_Severity"] = df["Accident_Severity"].astype(int)
    report.append(f"Severity distribution:\n{df['Accident_Severity'].value_counts().to_string()}")
    return df


def step_clean_casualties(df: pd.DataFrame, report: list) -> pd.DataFrame:
    if "Number_of_Casualties" not in df.columns:
        df["Number_of_Casualties"] = 1
        report.append("Number_of_Casualties: column absent — filled with 1")
        print("   Number_of_Casualties: missing → filled with 1")
    else:
        df["Number_of_Casualties"] = pd.to_numeric(
            df["Number_of_Casualties"], errors="coerce"
        ).fillna(1).clip(lower=1).astype(int)
        report.append(f"Number_of_Casualties range: {df['Number_of_Casualties'].min()}–{df['Number_of_Casualties'].max()}")
    return df


def step_clean_time(df: pd.DataFrame, report: list) -> pd.DataFrame:
    if "Time" not in df.columns:
        report.append("Time: column absent — skipped")
        return df

    # Standardise to HH:MM
    # The dataset has values like "13:30" — ensure consistent format
    df["Time"] = df["Time"].astype(str).str.strip()

    # Try to parse and reformat
    parsed = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    # Some rows might be "1:30" (single digit hour)
    parsed_single = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    fallback = pd.to_datetime(df["Time"], errors="coerce")
    parsed = parsed.fillna(fallback)

    null_time = int(parsed.isna().sum())
    df["Time"] = parsed.dt.strftime("%H:%M")
    # Derive Hour column used by RiskAnalyzer
    df["Hour"] = parsed.dt.hour

    report.append(f"Time nulls after parse : {null_time:,}")
    print(f"   Time nulls after parse : {null_time:,}")
    return df


def step_clean_date(df: pd.DataFrame, report: list) -> pd.DataFrame:
    if "Date" not in df.columns:
        return df

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    null_dates = int(df["Date"].isna().sum())

    # Derive Year / Month for optional downstream use
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    report.append(f"Date nulls : {null_dates:,}")
    print(f"   Date nulls : {null_dates:,}")

    if df["Date"].notna().any():
        year_range = f"{int(df['Year'].min())}–{int(df['Year'].max())}"
        report.append(f"Date range : {year_range}")
        print(f"   Year range : {year_range}")
    return df


def step_clean_day_of_week(df: pd.DataFrame, report: list) -> pd.DataFrame:
    if "Day_of_Week" not in df.columns:
        return df

    before = df["Day_of_Week"].unique()
    df["Day_of_Week"] = df["Day_of_Week"].astype(str).str.strip()
    df["Day_of_Week"] = df["Day_of_Week"].apply(normalise_day)

    after = df["Day_of_Week"].unique()
    report.append(f"Day_of_Week values after norm: {sorted(after)}")
    print(f"   Day_of_Week normalised: {sorted(before)} → {sorted(after)}")
    return df


def step_clean_weather(df: pd.DataFrame, report: list) -> pd.DataFrame:
    if "Weather_Conditions" not in df.columns:
        return df

    df["Weather_Conditions"] = (
        df["Weather_Conditions"].astype(str).str.strip()
    )

    # Replace obviously bad / placeholder values
    bad_values = ["nan", "unknown", "unknown (self reported)", "data missing or out of range"]
    mask = df["Weather_Conditions"].str.lower().isin(bad_values)
    replaced = int(mask.sum())
    df.loc[mask, "Weather_Conditions"] = "Unknown"

    top = df["Weather_Conditions"].value_counts().head(8)
    report.append(f"Weather_Conditions top values:\n{top.to_string()}")
    report.append(f"Weather_Conditions unknown/bad replaced: {replaced:,}")
    print(f"   Weather unknowns replaced: {replaced:,}")
    print(f"   Top weather values:\n{top.to_string()}")
    return df


def step_clean_road_surface(df: pd.DataFrame, report: list) -> pd.DataFrame:
    if "Road_Surface_Conditions" not in df.columns:
        return df

    df["Road_Surface_Conditions"] = (
        df["Road_Surface_Conditions"].astype(str).str.strip()
    )
    bad = ["nan", "unknown", "data missing or out of range"]
    mask = df["Road_Surface_Conditions"].str.lower().isin(bad)
    df.loc[mask, "Road_Surface_Conditions"] = "Unknown"
    return df


def step_drop_unused_cols(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """Keep only columns relevant to RouteGuard to reduce file size."""
    keep = [c for c in USED_COLS if c in df.columns]
    # Also keep Hour / Year / Month if derived
    for extra in ["Hour", "Year", "Month"]:
        if extra in df.columns and extra not in keep:
            keep.append(extra)
    dropped = [c for c in df.columns if c not in keep]
    df = df[keep]
    report.append(f"Columns dropped (unused) : {dropped}")
    print(f"   Columns kept  : {len(keep)}")
    print(f"   Columns dropped (unused) : {len(dropped)}")
    return df


def step_final_nulls(df: pd.DataFrame, report: list) -> pd.DataFrame:
    null_summary = df.isnull().sum()
    null_summary = null_summary[null_summary > 0]
    if len(null_summary):
        report.append(f"Remaining nulls per column:\n{null_summary.to_string()}")
        print(f"\n   Remaining nulls per column:\n{null_summary.to_string()}")
    else:
        report.append("No remaining nulls in retained columns.")
        print("   No remaining nulls ✓")
    return df


# ─────────────────────────────────────────────────────────────
# OPTIONAL: VEHICLE ENRICHMENT
# ─────────────────────────────────────────────────────────────

def enrich_with_vehicles(acc_df: pd.DataFrame, veh_path: str, report: list) -> pd.DataFrame:
    section("OPTIONAL: Enriching with Vehicle_Information.csv")
    veh_df = pd.read_csv(veh_path, low_memory=False)
    veh_df.columns = veh_df.columns.str.strip()

    if "Accident_Index" not in veh_df.columns or "Accident_Index" not in acc_df.columns:
        print("   Accident_Index missing in one file — skipping enrichment")
        report.append("Vehicle enrichment: skipped (no Accident_Index)")
        return acc_df

    # Count vehicles per accident
    veh_counts = veh_df.groupby("Accident_Index").size().rename("Vehicle_Count")
    acc_df = acc_df.merge(veh_counts, on="Accident_Index", how="left")
    acc_df["Vehicle_Count"] = acc_df["Vehicle_Count"].fillna(0).astype(int)

    # Most common vehicle type per accident
    if "Vehicle_Type" in veh_df.columns:
        dominant_veh = (
            veh_df.groupby("Accident_Index")["Vehicle_Type"]
            .agg(lambda x: x.mode().iloc[0] if len(x) else "Unknown")
            .rename("Dominant_Vehicle_Type")
        )
        acc_df = acc_df.merge(dominant_veh, on="Accident_Index", how="left")
        acc_df["Dominant_Vehicle_Type"] = acc_df["Dominant_Vehicle_Type"].fillna("Unknown")

    report.append("Vehicle enrichment: Vehicle_Count and Dominant_Vehicle_Type added")
    print(f"   Vehicle enrichment complete. Columns added: Vehicle_Count, Dominant_Vehicle_Type")
    return acc_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RouteGuard Data Preprocessing")
    parser.add_argument(
        "--accident", default="Accident_Information.csv",
        help="Path to Accident_Information.csv (default: ./Accident_Information.csv)"
    )
    parser.add_argument(
        "--vehicle", default=None,
        help="Path to Vehicle_Information.csv (optional enrichment)"
    )
    parser.add_argument(
        "--output", default="Accident_Information_clean.csv",
        help="Output CSV path (default: Accident_Information_clean.csv)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.accident):
        print(f"[ERROR] File not found: {args.accident}")
        print("  Download from: https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles")
        sys.exit(1)

    report = []
    report.append("RouteGuard Preprocessing Report")
    report.append("=" * 60)

    # ── Load ──────────────────────────────────────────────────
    df = load_accidents(args.accident)
    rows_original = len(df)
    report.append(f"Original rows : {rows_original:,}")

    # ── Clean ─────────────────────────────────────────────────
    section("2. CLEANING")

    df = step_required_cols(df, report)
    df = step_drop_duplicates(df, report)
    df = step_clean_coordinates(df, report)
    df = step_clean_severity(df, report)
    df = step_clean_casualties(df, report)
    df = step_clean_time(df, report)
    df = step_clean_date(df, report)
    df = step_clean_day_of_week(df, report)
    df = step_clean_weather(df, report)
    df = step_clean_road_surface(df, report)
    df = step_drop_unused_cols(df, report)
    df = step_final_nulls(df, report)

    # ── Optional Vehicle Enrichment ───────────────────────────
    if args.vehicle and os.path.exists(args.vehicle):
        df = enrich_with_vehicles(df, args.vehicle, report)

    # ── Final Stats ───────────────────────────────────────────
    section("3. FINAL STATS")
    rows_final = len(df)
    pct_kept = 100 * rows_final / rows_original if rows_original else 0
    print(f"   Original rows : {rows_original:,}")
    print(f"   Clean rows    : {rows_final:,}  ({pct_kept:.1f}% retained)")
    print(f"   Columns       : {list(df.columns)}")
    report.append(f"\nFinal rows  : {rows_final:,}  ({pct_kept:.1f}% retained)")
    report.append(f"Final cols  : {list(df.columns)}")

    # Severity summary
    sev_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
    sev_counts = df["Accident_Severity"].map(sev_map).value_counts()
    print(f"\n   Severity breakdown:\n{sev_counts.to_string()}")
    report.append(f"\nSeverity breakdown:\n{sev_counts.to_string()}")

    # Year range
    if "Year" in df.columns:
        yr = f"{int(df['Year'].min())}–{int(df['Year'].max())}"
        print(f"   Year range : {yr}")

    # ── Save ──────────────────────────────────────────────────
    section("4. SAVING")
    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"   Saved : {out_path}  ({size_mb:.1f} MB)")
    report.append(f"\nOutput file : {out_path}  ({size_mb:.1f} MB)")

    # ── Save report ───────────────────────────────────────────
    report_path = out_path.parent / "preprocessing_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"   Report: {report_path}")

    section("DONE ✓")
    print(f"   → Use '{out_path}' as your Accident_Information.csv in RouteGuard.\n")


if __name__ == "__main__":
    main()