"""
train_model.py — Train an ML model to predict accident severity/probability
===============================================================================
Uses the cleaned UK Road Safety dataset to train a Gradient Boosting classifier
that predicts the probability of a SEVERE accident (Fatal or Serious)
given contextual features:
  - Hour of day
  - Day of week
  - Weather conditions
  - Road type
  - Speed limit
  - Light conditions
  - Urban or Rural area

The model outputs P(severe) which is used by RiskAnalyzer to adjust route
risk scores beyond simple historical counting.

Usage:
  python train_model.py
  python train_model.py --data path/to/Accident_Information_clean.csv
  python train_model.py --data data.csv --output model.joblib

Output:
  - accident_model.joblib        (trained model pipeline)
  - model_report.txt             (training metrics)
"""

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

# Features the model uses
NUMERIC_FEATURES = ["Hour", "Speed_limit"]
CATEGORICAL_FEATURES = ["Day_of_Week", "Weather_Group", "Road_Type", "Light_Group", "Urban_or_Rural_Area"]

# Weather grouping: simplify many categories into a few
WEATHER_MAP = {
    "fine no high winds": "Fine",
    "fine + high winds": "Fine",
    "raining no high winds": "Rain",
    "raining + high winds": "Rain",
    "snowing no high winds": "Snow",
    "snowing + high winds": "Snow",
    "fog or mist": "Fog",
}

# Light conditions grouping
LIGHT_MAP = {
    "daylight": "Day",
    "darkness - lights lit": "Dark_Lit",
    "darkness - lights unlit": "Dark_Unlit",
    "darkness - no lighting": "Dark_None",
    "darkness - lighting unknown": "Dark_Unknown",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw columns."""
    out = pd.DataFrame()

    # Hour (numeric)
    if "Hour" in df.columns:
        out["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").fillna(12).astype(int)
    else:
        out["Hour"] = 12

    # Speed limit (numeric)
    if "Speed_limit" in df.columns:
        out["Speed_limit"] = pd.to_numeric(df["Speed_limit"], errors="coerce").fillna(30).astype(int)
    else:
        out["Speed_limit"] = 30

    # Day of week (categorical)
    if "Day_of_Week" in df.columns:
        out["Day_of_Week"] = df["Day_of_Week"].astype(str).str.strip()
    else:
        out["Day_of_Week"] = "Unknown"

    # Weather group (categorical)
    if "Weather_Conditions" in df.columns:
        out["Weather_Group"] = (
            df["Weather_Conditions"]
            .astype(str).str.strip().str.lower()
            .map(WEATHER_MAP)
            .fillna("Other")
        )
    else:
        out["Weather_Group"] = "Other"

    # Road type (categorical)
    if "Road_Type" in df.columns:
        out["Road_Type"] = df["Road_Type"].astype(str).str.strip()
    else:
        out["Road_Type"] = "Unknown"

    # Light conditions group (categorical)
    if "Light_Conditions" in df.columns:
        out["Light_Group"] = (
            df["Light_Conditions"]
            .astype(str).str.strip().str.lower()
            .map(LIGHT_MAP)
            .fillna("Unknown")
        )
    else:
        out["Light_Group"] = "Unknown"

    # Urban or Rural (categorical)
    if "Urban_or_Rural_Area" in df.columns:
        out["Urban_or_Rural_Area"] = df["Urban_or_Rural_Area"].astype(str).str.strip()
    else:
        out["Urban_or_Rural_Area"] = "Unknown"

    return out


def prepare_target(df: pd.DataFrame) -> np.ndarray:
    """
    Binary target: 1 = Severe (Fatal or Serious, severity 1 or 2)
                   0 = Slight (severity 3)
    """
    sev = pd.to_numeric(df["Accident_Severity"], errors="coerce").fillna(3).astype(int)
    return (sev <= 2).astype(int).values


# ─────────────────────────────────────────────────────────────
# BUILD PIPELINE
# ─────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """Build sklearn pipeline with preprocessing + classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=42,
        )),
    ])

    return pipeline


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RouteGuard ML Model")
    parser.add_argument(
        "--data", default="Accident_Information_clean.csv",
        help="Path to cleaned accident CSV"
    )
    parser.add_argument(
        "--output", default="accident_model.joblib",
        help="Output model path"
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"[ERROR] File not found: {args.data}")
        print("  Run preprocess.py first, or specify --data path")
        sys.exit(1)

    report = []

    # ── Load ──────────────────────────────────────────────────
    print("=" * 60)
    print("  RouteGuard ML Model Training")
    print("=" * 60)

    df = pd.read_csv(args.data, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"\n  Dataset: {len(df):,} records")
    report.append(f"Dataset: {len(df):,} records")

    # ── Prepare features and target ───────────────────────────
    X = prepare_features(df)
    y = prepare_target(df)

    severe_pct = y.mean() * 100
    print(f"  Severe accidents: {y.sum():,} ({severe_pct:.1f}%)")
    print(f"  Slight accidents: {(y == 0).sum():,} ({100 - severe_pct:.1f}%)")
    report.append(f"Severe: {y.sum():,} ({severe_pct:.1f}%)")
    report.append(f"Slight: {(y == 0).sum():,} ({100 - severe_pct:.1f}%)")

    print(f"\n  Features: {list(X.columns)}")
    report.append(f"Features: {list(X.columns)}")

    # ── Train/test split ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    report.append(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Build and train ───────────────────────────────────────
    print("\n  Training Gradient Boosting Classifier...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("  Training complete ✓")

    # ── Evaluate ──────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n  ROC AUC: {auc:.4f}")
    report.append(f"\nROC AUC: {auc:.4f}")

    cr = classification_report(y_test, y_pred, target_names=["Slight", "Severe"])
    print(f"\n{cr}")
    report.append(f"\nClassification Report:\n{cr}")

    # ── Feature importance ────────────────────────────────────
    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get feature names after transformation
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_features = list(NUMERIC_FEATURES) + list(cat_features)

    importances = clf.feature_importances_
    feat_imp = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)

    print("\n  Top 10 Feature Importances:")
    report.append("\nTop 10 Feature Importances:")
    for name, imp in feat_imp[:10]:
        line = f"    {name:40s} {imp:.4f}"
        print(line)
        report.append(line)

    # ── Probability distribution ──────────────────────────────
    print(f"\n  P(severe) distribution on test set:")
    for pct in [10, 25, 50, 75, 90]:
        val = np.percentile(y_prob, pct)
        print(f"    P{pct}: {val:.4f}")
    report.append(f"\nP(severe) median: {np.median(y_prob):.4f}")
    report.append(f"P(severe) mean: {np.mean(y_prob):.4f}")

    # ── Save model ────────────────────────────────────────────
    out_path = Path(args.output)
    joblib.dump(pipeline, out_path)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\n  Model saved: {out_path} ({size_mb:.1f} MB)")
    report.append(f"\nModel saved: {out_path} ({size_mb:.1f} MB)")

    # ── Save report ───────────────────────────────────────────
    report_path = out_path.parent / "model_report.txt"
    report_path.write_text("\n".join(report))
    print(f"  Report: {report_path}")

    print("\n" + "=" * 60)
    print("  DONE ✓")
    print(f"  → Use '{out_path}' in RiskAnalyzer for ML-enhanced scoring.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()