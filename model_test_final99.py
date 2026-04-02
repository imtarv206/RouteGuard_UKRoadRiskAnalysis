"""
train_model.py — Train an ML model to predict accident severity/probability
===============================================================================
Uses the cleaned UK Road Safety dataset to train a Hybrid HistGradientBoosting
classifier that predicts the probability of a SEVERE accident (Fatal or Serious)
given contextual features.

Output:
  - accident_model.joblib   (trained model pipeline + threshold)
  - model_report.txt        (training metrics)
"""

import argparse
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

try:
	import joblib
except ImportError:
	from sklearn.externals import joblib


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

WEATHER_MAP = {
	"fine no high winds": "Fine",   "fine + high winds": "Fine",
	"raining no high winds": "Rain", "raining + high winds": "Rain",
	"snowing no high winds": "Snow", "snowing + high winds": "Snow",
	"fog or mist": "Fog",
}

LIGHT_MAP = {
	"daylight": "Day",
	"darkness - lights lit": "Dark_Lit",
	"darkness - lights unlit": "Dark_Unlit",
	"darkness - no lighting": "Dark_None",
	"darkness - lighting unknown": "Dark_Unknown",
}

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
	out = pd.DataFrame()

	# Numeric
	out["Hour"]        = pd.to_numeric(df.get("Hour"), errors="coerce").fillna(12).astype(int)
	out["Speed_limit"] = pd.to_numeric(df.get("Speed_limit"), errors="coerce").fillna(30).astype(int)
	out["Month"]       = pd.to_numeric(df.get("Month"), errors="coerce").fillna(1).astype(int)

	# Engineered numeric
	out["Is_High_Speed"] = (out["Speed_limit"] > 40).astype(int)

	# Engineered categorical
	def time_of_day(h):
		if 7 <= h <= 9 or 16 <= h <= 19: return "RushHour"
		if 10 <= h <= 15:                 return "Day"
		return "Night"

	out["Time_of_Day"] = out["Hour"].apply(time_of_day)

	# Categorical (grouped)
	out["Weather_Group"] = (
		df.get("Weather_Conditions", pd.Series(["Other"] * len(df)))
		.astype(str).str.strip().str.lower()
		.map(WEATHER_MAP).fillna("Other")
	)
	out["Light_Group"] = (
		df.get("Light_Conditions", pd.Series(["Unknown"] * len(df)))
		.astype(str).str.strip().str.lower()
		.map(LIGHT_MAP).fillna("Unknown")
	)

	# Categorical (raw)
	for col, default in [
		("Day_of_Week",       "Unknown"),
		("Road_Type",         "Unknown"),
		("Road_Surface_Conditions", "Unknown"),
		("Urban_or_Rural_Area", "Unknown"),
		("Junction_Detail",   "Unknown"),
		("1st_Road_Class",    "Unknown"),
	]:
		out[col] = df[col].astype(str).str.strip() if col in df.columns else default

	return out


def prepare_target(df: pd.DataFrame) -> np.ndarray:
	"""Binary: 1 = Severe (Fatal/Serious), 0 = Slight"""
	sev = df["Accident_Severity"]
	
	# ถ้าเป็นตัวเลข: 1=Fatal, 2=Serious, 3=Slight
	if pd.api.types.is_numeric_dtype(sev):
		return (sev <= 2).astype(int).values
	
	# ถ้าเป็นข้อความ: Fatal, Serious, Slight
	return sev.astype(str).str.strip().isin(["Fatal", "Serious"]).astype(int).values

# ─────────────────────────────────────────────────────────────
# BUILD PIPELINE
# ─────────────────────────────────────────────────────────────

def build_pipeline() -> ImbPipeline:
	return ImbPipeline([
		("under", RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
		("clf",   HistGradientBoostingClassifier(
			max_iter=200,
			max_depth=6,
			learning_rate=0.05,
			min_samples_leaf=50,
			categorical_features="from_dtype",
			random_state=42,
		)),
	])


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
	parser = argparse.ArgumentParser(description="Train RouteGuard ML Model")
	parser.add_argument("--data",   default="accidents_clean.csv")
	parser.add_argument("--output", default="accident_model.joblib")
	args = parser.parse_args()

	if not Path(args.data).exists():
		print(f"[ERROR] File not found: {args.data}")
		sys.exit(1)

	report = []

	print("=" * 60)
	print("  RouteGuard ML Model Training")
	print("=" * 60)

	# ── Load ──────────────────────────────────────────────────
	df = pd.read_csv(args.data, low_memory=False)
	df.columns = df.columns.str.strip()
	print(f"\n  Dataset: {len(df):,} records")
	report.append(f"Dataset: {len(df):,} records")

	# ── Prepare ───────────────────────────────────────────────
	X = prepare_features(df)
	y = prepare_target(df)

	# Convert categorical columns to pandas category dtype
	cat_cols = ["Time_of_Day", "Weather_Group", "Light_Group", "Day_of_Week",
				"Road_Type", "Road_Surface_Conditions",
				"Urban_or_Rural_Area", "Junction_Detail", "1st_Road_Class"]
	for col in cat_cols:
		X[col] = X[col].astype("category")

	severe_pct = y.mean() * 100
	print(f"  Severe: {y.sum():,} ({severe_pct:.1f}%)")
	print(f"  Slight: {(y==0).sum():,} ({100-severe_pct:.1f}%)")
	print(f"  Features: {list(X.columns)}")
	report += [f"Severe: {y.sum():,} ({severe_pct:.1f}%)",
			   f"Features: {list(X.columns)}"]

	X_train_all, X_test, y_train_all, y_test = train_test_split(
		X, y,
		test_size=0.2,
		stratify=y,
		random_state=42
	)
	# ── สร้าง KFold ─────────────────────────────
	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

	# ── loop train ─────────────────────────────
	thresholds = []
	auc_scores = []

	for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all)):
		print(f"\n--- Fold {fold+1} ---")
		
		X_train = X_train_all.iloc[train_idx]
		X_val   = X_train_all.iloc[val_idx]

		y_train = y_train_all[train_idx]
		y_val   = y_train_all[val_idx]

		model = build_pipeline()
		model.fit(X_train, y_train)

		# predict prob
		y_prob = model.predict_proba(X_val)[:, 1]

		# หา threshold ดีสุด
		ths = np.arange(0.1, 0.91, 0.01)
		f1s = [
			f1_score(y_val, (y_prob > t).astype(int), average="binary")
			for t in ths
		]
		best_t = ths[np.argmax(f1s)]
		thresholds.append(best_t)

		# AUC
		auc = roc_auc_score(y_val, y_prob)
		auc_scores.append(auc)
		print(f"AUC: {auc:.4f} | Best threshold: {best_t:.2f}")

	# ── AVG ─────────────────────────────────────────────────
	final_threshold = float(np.mean(thresholds))
	final_auc = float(np.mean(auc_scores))

	print("\n=== FINAL ===")
	print(f"Avg AUC: {final_auc:.4f}")
	print(f"Avg Threshold: {final_threshold:.2f}")

	# ── FINAL Train ─────────────────────────────────────────────────
	final_model = build_pipeline()
	final_model.fit(X_train_all, y_train_all)
	print("  Training complete ✓")

	# ── Evaluate บน test set ──────────────────────────────────
	y_prob_test = final_model.predict_proba(X_test)[:, 1]
	y_pred_test = (y_prob_test > final_threshold).astype(int)

	auc_test = roc_auc_score(y_test, y_prob_test)
	cr_test  = classification_report(y_test, y_pred_test, target_names=["Slight", "Severe"])

	print(f"\n  TEST ROC AUC: {auc_test:.4f}")
	print(f"\n{cr_test}")
	report += [f"\nROC AUC: {auc_test:.4f}", f"\nClassification Report:\n{cr_test}"]

	# ── Feature importance ────────────────────────────────────
	print("  Calculating feature importance...")

	# ใช้ permutation importance แทน
	perm = permutation_importance(
		final_model,
		X_test,
		y_test,
		n_repeats=5,
		random_state=42,
		n_jobs=-1
	)

	feat_imp = sorted(
		zip(X_train_all.columns, perm.importances_mean),
		key=lambda x: x[1], reverse=True
	)

	print("  Top 10 Feature Importances:")
	report.append("\nTop 10 Feature Importances:")
	for name, imp in feat_imp[:10]:
		line = f"    {name:40s} {imp:.4f}"
		print(line)
		report.append(line)

	# ── Probability distribution ──────────────────────────────
	print(f"\n  P(severe) distribution on test set:")
	for pct in [10, 25, 50, 75, 90]:
		val = np.percentile(y_prob_test, pct)
		print(f"    P{pct}: {val:.4f}")
	report.append(f"\nP(severe) median: {np.median(y_prob_test):.4f}")

	# ── Save ──────────────────────────────────────────────────
	out_path = Path(args.output)
	joblib.dump({
		"model": final_model,
		"threshold": final_threshold
	}, out_path)
	size_mb = out_path.stat().st_size / 1_048_576
	print(f"\n  Model saved: {out_path} ({size_mb:.1f} MB)")
	report.append(f"\nModel saved: {out_path} ({size_mb:.1f} MB)")

	report_path = out_path.parent / "model_report.txt"
	report_path.write_text("\n".join(report))
	print(f"  Report: {report_path}")

	print("\n" + "=" * 60)
	print("  DONE ✓")
	print(f"  → Use '{out_path}' in RiskAnalyzer for ML-enhanced scoring.")
	print("=" * 60 + "\n")


if __name__ == "__main__":
	main()