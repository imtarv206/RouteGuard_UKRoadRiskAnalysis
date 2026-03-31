import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from typing import List, Tuple, Dict, Optional
from pathlib import Path

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# UK dataset: Accident_Severity 1=Fatal, 2=Serious, 3=Slight
SEVERITY_WEIGHT = {1: 5, 2: 3, 3: 1}
SEVERITY_LABEL  = {1: "Fatal", 2: "Serious", 3: "Slight"}

# ── Scoring parameters ──────────────────────────────────────────
# MAX_RADIUS : only accidents within this corridor (metres) affect the route
MAX_RADIUS = 200

# SIGMA : Gaussian distance-decay — accident at SIGMA metres = e^-1 ≈ 37% weight
# Keeps close accidents influential, fades out distant ones smoothly
SIGMA = 80.0

# SCALE : log-normalisation constant calibrated to UK accident density
#   raw_per_km  ≈  SCALE  →  score ≈ 63
#   Calibration (weighted accidents per effective km):
#     ~5   / km  →  score  2  (LOW, quiet rural)
#     ~30  / km  →  score 14  (LOW, suburban)
#     ~80  / km  →  score 33  (MEDIUM, normal urban)
#     ~200 / km  →  score 63  (HIGH, busy central)
#     ~500 / km  →  score 92  (HIGH, genuine hotspot)
SCALE = 200.0

RISK_THRESHOLDS = {"LOW": 25, "MEDIUM": 55}   # ≥55 → HIGH

# MIN_ROUTE_KM : minimum effective distance for per-km scoring
# Prevents short routes (< 1 km) from getting inflated scores
MIN_ROUTE_KM = 1.0


# ─────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Return distance in metres between two lat/lon points."""
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


def interpolate_route(
    route: List[Tuple[float, float]], spacing_m: int = 50
) -> List[Tuple[float, float]]:
    """Add intermediate points every ~spacing_m metres."""
    result = [route[0]]
    for i in range(1, len(route)):
        p1, p2 = route[i - 1], route[i]
        dist = haversine(*p1, *p2)
        n_steps = max(1, int(dist / spacing_m))
        for step in range(1, n_steps + 1):
            frac = step / n_steps
            result.append((
                p1[0] + frac * (p2[0] - p1[0]),
                p1[1] + frac * (p2[1] - p1[1]),
            ))
    return result


def score_to_level(score: float) -> str:
    if score < RISK_THRESHOLDS["LOW"]:
        return "LOW"
    if score < RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"


# ─────────────────────────────────────────────────────────────────
# RISK ANALYZER
# ─────────────────────────────────────────────────────────────────

class RiskAnalyzer:
    def __init__(self, csv_path: str, model_path: str = None):
        self.df = self._load(csv_path)
        self.ml_model = self._load_model(model_path or str(Path(csv_path).parent / "accident_model.joblib"))

    # ── Load ML Model ──────────────────────────────────────────

    def _load_model(self, path: str):
        """Load trained ML model if available. Gracefully skip if not found."""
        if not HAS_JOBLIB:
            print("[RiskAnalyzer] joblib not installed — ML model disabled.")
            return None
        p = Path(path)
        if not p.exists():
            print(f"[RiskAnalyzer] ML model not found at {p} — ML scoring disabled.")
            print(f"  → Run: python train_model.py --data <csv> --output {p}")
            return None
        try:
            model = joblib.load(p)
            print(f"[RiskAnalyzer] ML model loaded ✓ ({p.name})")
            return model
        except Exception as e:
            print(f"[RiskAnalyzer] Failed to load ML model: {e}")
            return None

    # ── Load ──────────────────────────────────────────────────────

    def _load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()

        required = ["Latitude", "Longitude", "Accident_Severity"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        df = df.dropna(subset=["Latitude", "Longitude"])
        df["Accident_Severity"] = pd.to_numeric(
            df["Accident_Severity"], errors="coerce"
        ).fillna(3).astype(int)

        if "Number_of_Casualties" not in df.columns:
            df["Number_of_Casualties"] = 1
        else:
            df["Number_of_Casualties"] = (
                pd.to_numeric(df["Number_of_Casualties"], errors="coerce")
                .fillna(1)
                .clip(lower=1)
            )

        if "Time" in df.columns:
            df["Hour"] = pd.to_datetime(
                df["Time"], format="%H:%M", errors="coerce"
            ).dt.hour
        else:
            df["Hour"] = np.nan

        if "Day_of_Week" in df.columns:
            df["Day_of_Week"] = df["Day_of_Week"].astype(str).str.strip()

        print(f"[RiskAnalyzer] Loaded {len(df):,} accident records.")
        return df

    # ── Analyse ───────────────────────────────────────────────────

    def analyze(
        self,
        route: List[Tuple[float, float]],
        time_from: Optional[int] = None,
        time_to: Optional[int] = None,
        day_of_week: str = "any",
        weather: str = "any",
    ) -> dict:

        dense_route = interpolate_route(route, spacing_m=50)

        df_filtered = self._filter_context(
            time_from=time_from,
            time_to=time_to,
            day_of_week=day_of_week,
            weather=weather,
        )

        route_km = sum(
            haversine(*route[i], *route[i + 1]) / 1000
            for i in range(len(route) - 1)
        )
        route_km = max(route_km, 0.1)

        # Effective distance for scoring: floor at MIN_ROUTE_KM so that
        # very short routes are not penalised disproportionately.
        # The real route_km is still shown in stats.
        effective_km = max(route_km, MIN_ROUTE_KM)

        if df_filtered.empty:
            return {
                "overall_score": 0.0,
                "risk_level": "LOW",
                "risk_points": [],
                "stats": {
                    "route_km": round(route_km, 2),
                    "route_points_checked": len(dense_route),
                    "accidents_in_filter": 0,
                    "fatal_nearby": 0,
                    "serious_nearby": 0,
                },
                "recommendations": ["🟢 ไม่พบข้อมูลอุบัติเหตุที่เข้าเงื่อนไขนี้ใกล้เส้นทาง"],
            }

        acc_lats = df_filtered["Latitude"].values
        acc_lons = df_filtered["Longitude"].values
        acc_sev  = df_filtered["Accident_Severity"].values

        # ── Step 1: Find minimum distance from each accident to the route ──
        # Build a matrix: rows = accidents, cols = route points
        # Then take row-wise minimum.
        # For very long routes we chunk to avoid memory issues.

        min_dist = self._min_dist_to_route(acc_lats, acc_lons, dense_route)

        # ── Step 2: Keep only accidents within MAX_RADIUS ─────────────────
        nearby_mask = min_dist <= MAX_RADIUS
        n_nearby   = int(nearby_mask.sum())

        if n_nearby == 0:
            return {
                "overall_score": 0.0,
                "risk_level": "LOW",
                "risk_points": [],
                "stats": {
                    "route_km": round(route_km, 2),
                    "route_points_checked": len(dense_route),
                    "accidents_in_filter": len(df_filtered),
                    "fatal_nearby": 0,
                    "serious_nearby": 0,
                },
                "recommendations": ["🟢 ไม่พบอุบัติเหตุในระยะ 200 ม. จากเส้นทาง"],
            }

        near_dist = min_dist[nearby_mask]
        near_sev  = acc_sev[nearby_mask]

        # ── Step 3: Gaussian distance decay ──────────────────────────────
        decay = np.exp(-near_dist / SIGMA)

        # ── Step 4: Severity weight (no casualties multiplier — too noisy) ─
        sev_w = np.vectorize(lambda s: SEVERITY_WEIGHT.get(int(s), 1))(near_sev)

        # ── Step 5: Per-accident weighted score ──────────────────────────
        acc_scores = sev_w * decay          # shape: (n_nearby,)

        # ── Step 6: Aggregate and normalise ──────────────────────────────
        # raw_per_km = total weighted accidents per effective km of route
        raw_per_km = float(acc_scores.sum()) / effective_km

        # Log-saturation: score = 100 * (1 - exp(-raw_per_km / SCALE))
        # With MIN_ROUTE_KM = 1.0, short routes use at least 1 km as divisor,
        # so a 0.5 km route in a dense area scores the same as a 1 km route
        # with the same accident density — preventing artificial inflation.
        base_score = 100.0 * (1.0 - np.exp(-raw_per_km / SCALE))

        # ── Step 6b: ML severity adjustment ───────────────────────────
        # If ML model is available, predict P(severe) for the given context
        # and use it as a multiplier: higher P(severe) → higher score
        ml_prob = self._predict_severity_prob(time_from, time_to, day_of_week, weather)
        if ml_prob is not None:
            # ml_multiplier ranges from ~0.85 (low severity context) to ~1.3 (high severity context)
            # Baseline P(severe) in UK data ≈ 0.15, so normalise around that
            ml_multiplier = 0.7 + 2.0 * ml_prob   # e.g. 0.15 → 1.0, 0.25 → 1.2, 0.05 → 0.8
            overall_score = round(min(100.0, base_score * ml_multiplier), 1)
        else:
            overall_score = round(min(100.0, base_score), 1)

        overall_level = score_to_level(overall_score)

        fatal_count   = int((near_sev == 1).sum())
        serious_count = int((near_sev == 2).sum())

        # ── Step 7: Build risk_points for map display ─────────────────────
        # Assign each nearby accident to the closest route point for display
        risk_points = self._build_risk_points(
            acc_lats[nearby_mask],
            acc_lons[nearby_mask],
            near_sev,
            acc_scores,
            near_dist,
            dense_route,
        )

        recs = self._recommendations(
            level=overall_level,
            time_from=time_from,
            time_to=time_to,
            fatal=fatal_count,
            serious=serious_count,
            df_filtered=df_filtered,
            ml_prob=ml_prob,
        )

        return {
            "overall_score": overall_score,
            "risk_level": overall_level,
            "risk_points": risk_points[:200],
            "stats": {
                "route_km": round(route_km, 2),
                "route_points_checked": len(dense_route),
                "accidents_in_filter": len(df_filtered),
                "total_nearby": n_nearby,
                "fatal_nearby": fatal_count,
                "serious_nearby": serious_count,
                "ml_severity_prob": round(ml_prob, 4) if ml_prob is not None else None,
            },
            "recommendations": recs,
        }

    # ── Helpers ───────────────────────────────────────────────────

    def _min_dist_to_route(
        self,
        acc_lats: np.ndarray,
        acc_lons: np.ndarray,
        route_points: List[Tuple[float, float]],
        chunk_size: int = 500,
    ) -> np.ndarray:
        """
        Return an array of length n_accidents where each value is the
        minimum distance (metres) from that accident to any route point.
        Processed in chunks to limit peak memory usage.
        """
        n_acc = len(acc_lats)
        min_dist = np.full(n_acc, np.inf)

        for i in range(0, len(route_points), chunk_size):
            chunk = route_points[i : i + chunk_size]
            for (rlat, rlon) in chunk:
                d = self._vectorised_haversine(rlat, rlon, acc_lats, acc_lons)
                np.minimum(min_dist, d, out=min_dist)

        return min_dist

    def _build_risk_points(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        sevs: np.ndarray,
        scores: np.ndarray,
        dists: np.ndarray,
        route_points: List[Tuple[float, float]],
    ) -> list:
        """
        Group nearby accidents by their closest route point and return
        one risk_point dict per route point that has accidents nearby.
        """
        n_route = len(route_points)

        # Find closest route point index for each accident
        rp_lats = np.array([p[0] for p in route_points])
        rp_lons = np.array([p[1] for p in route_points])

        # For each accident find index of nearest route point
        closest_idx = np.zeros(len(lats), dtype=int)
        best_d = np.full(len(lats), np.inf)
        for ri, (rlat, rlon) in enumerate(route_points):
            d = self._vectorised_haversine(rlat, rlon, lats, lons)
            mask = d < best_d
            best_d[mask] = d[mask]
            closest_idx[mask] = ri

        # Aggregate by route point
        result = {}
        for ai in range(len(lats)):
            ri = closest_idx[ai]
            if ri not in result:
                result[ri] = {
                    "lat": route_points[ri][0],
                    "lng": route_points[ri][1],
                    "total_score": 0.0,
                    "nearby_accidents": 0,
                }
            result[ri]["total_score"]      += float(scores[ai])
            result[ri]["nearby_accidents"] += 1

        risk_points = []
        for ri, data in result.items():
            pt_score = data["total_score"]
            # Per-point level uses the same log formula but with a shorter reference
            pt_level = score_to_level(
                100.0 * (1.0 - np.exp(-pt_score / (SCALE * 0.15)))
            )
            risk_points.append({
                "lat": data["lat"],
                "lng": data["lng"],
                "risk_level": pt_level,
                "score": round(pt_score, 2),
                "nearby_accidents": data["nearby_accidents"],
            })

        # Sort by score descending for map rendering
        risk_points.sort(key=lambda x: x["score"], reverse=True)
        return risk_points

    # ── Filter ────────────────────────────────────────────────────

    def _filter_context(
        self,
        time_from: Optional[int] = None,
        time_to: Optional[int] = None,
        day_of_week: str = "any",
        weather: str = "any",
    ) -> pd.DataFrame:
        df = self.df.copy()

        if "Hour" in df.columns:
            if time_from is not None and time_to is not None:
                if time_from <= time_to:
                    df = df[df["Hour"].between(time_from, time_to, inclusive="both")]
                else:
                    df = df[(df["Hour"] >= time_from) | (df["Hour"] <= time_to)]
            elif time_from is not None:
                df = df[df["Hour"] >= time_from]
            elif time_to is not None:
                df = df[df["Hour"] <= time_to]

        if day_of_week != "any" and "Day_of_Week" in df.columns:
            day_map = {
                "monday":    ["monday",    "mon", "2"],
                "tuesday":   ["tuesday",   "tue", "3"],
                "wednesday": ["wednesday", "wed", "4"],
                "thursday":  ["thursday",  "thu", "5"],
                "friday":    ["friday",    "fri", "6"],
                "saturday":  ["saturday",  "sat", "7"],
                "sunday":    ["sunday",    "sun", "1"],
            }
            key     = str(day_of_week).strip().lower()
            allowed = day_map.get(key, [key])
            day_series = df["Day_of_Week"].astype(str).str.strip().str.lower()
            df = df[day_series.isin(allowed)]

        if weather != "any" and "Weather_Conditions" in df.columns:
            df = df[
                df["Weather_Conditions"]
                .astype(str)
                .str.lower()
                .str.contains(weather.lower(), na=False)
            ]

        return df.reset_index(drop=True)

    @staticmethod
    def _vectorised_haversine(lat1, lon1, lats, lons) -> np.ndarray:
        R = 6_371_000
        lat1, lon1 = radians(lat1), radians(lon1)
        lats_r = np.radians(lats)
        lons_r = np.radians(lons)
        dlat   = lats_r - lat1
        dlon   = lons_r - lon1
        a = np.sin(dlat / 2) ** 2 + cos(lat1) * np.cos(lats_r) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # ── ML Prediction ────────────────────────────────────────────

    # Weather mapping (must match train_model.py)
    _WEATHER_MAP = {
        "fine": "Fine", "rain": "Rain", "snow": "Snow", "fog": "Fog",
    }

    def _predict_severity_prob(
        self,
        time_from: Optional[int],
        time_to: Optional[int],
        day_of_week: str,
        weather: str,
    ) -> Optional[float]:
        """
        Use ML model to predict P(severe accident) given context.
        Returns None if model is not available.
        """
        if self.ml_model is None:
            return None

        try:
            # Use mid-point of time range, or 12 if not specified
            if time_from is not None and time_to is not None:
                hour = (time_from + time_to) // 2 if time_from <= time_to else ((time_from + time_to + 24) // 2) % 24
            elif time_from is not None:
                hour = time_from
            elif time_to is not None:
                hour = time_to
            else:
                hour = 12

            # Map weather filter to model's weather group
            weather_group = "Other"
            if weather != "any":
                weather_group = self._WEATHER_MAP.get(weather.lower(), "Other")

            # Day of week
            day = day_of_week if day_of_week != "any" else "Friday"  # default to a common day

            # Build a single-row DataFrame matching training features
            row = pd.DataFrame([{
                "Hour": hour,
                "Speed_limit": 30,              # default urban speed
                "Day_of_Week": day,
                "Weather_Group": weather_group,
                "Road_Type": "Single carriageway",  # most common
                "Light_Group": "Day" if 6 <= hour <= 18 else "Dark_Lit",
                "Urban_or_Rural_Area": "1",      # urban (most routes)
            }])

            prob = float(self.ml_model.predict_proba(row)[:, 1][0])
            return prob

        except Exception as e:
            print(f"[RiskAnalyzer] ML prediction error: {e}")
            return None

    # ── Recommendations ───────────────────────────────────────────

    def _recommendations(
        self,
        level: str,
        time_from: Optional[int],
        time_to: Optional[int],
        fatal: int,
        serious: int,
        df_filtered: pd.DataFrame,
        ml_prob: Optional[float] = None,
    ) -> list:
        recs = []

        if level == "HIGH":
            recs.append("⚠️ เส้นทางนี้มีความเสี่ยงสูง — พิจารณาเส้นทางสำรอง")
        elif level == "MEDIUM":
            recs.append("🟡 เส้นทางนี้มีความเสี่ยงปานกลาง — ขับขี่ด้วยความระมัดระวัง")
        else:
            recs.append("🟢 เส้นทางนี้มีความเสี่ยงต่ำ — ปลอดภัยดี")

        # ML-based severity insight
        if ml_prob is not None:
            pct = round(ml_prob * 100, 1)
            if ml_prob > 0.25:
                recs.append(f"🤖 ML ประเมิน: สภาพแวดล้อมนี้มีโอกาสเกิดอุบัติเหตุรุนแรง {pct}% — สูงกว่าปกติ")
            elif ml_prob > 0.15:
                recs.append(f"🤖 ML ประเมิน: โอกาสเกิดอุบัติเหตุรุนแรง {pct}% — อยู่ในระดับเฉลี่ย")
            else:
                recs.append(f"🤖 ML ประเมิน: โอกาสเกิดอุบัติเหตุรุนแรง {pct}% — ต่ำกว่าปกติ")

        if fatal > 0:
            recs.append(f"💀 มีจุดอุบัติเหตุร้ายแรง {fatal} จุดใกล้เส้นทาง — ระวังเป็นพิเศษ")
        if serious > 5:
            recs.append(f"🚑 มีอุบัติเหตุบาดเจ็บสาหัส {serious} จุดใกล้เส้นทาง")

        if time_from is not None or time_to is not None:
            if time_from is not None and time_to is not None:
                recs.append(
                    f"🕐 กำลังประเมินตามช่วงเวลา {time_from:02d}:00 – {time_to:02d}:59 น."
                )
            elif time_from is not None:
                recs.append(f"🕐 กำลังประเมินตั้งแต่ {time_from:02d}:00 น.")
            else:
                recs.append(f"🕐 กำลังประเมินถึง {time_to:02d}:59 น.")

        if "Hour" in df_filtered.columns and df_filtered["Hour"].notna().any():
            hourly = df_filtered.groupby("Hour").size()
            if not hourly.empty:
                peak_hour = int(hourly.idxmax())
                recs.append(f"🕐 ช่วงเวลาที่มีอุบัติเหตุสูงสุดในข้อมูลนี้คือ {peak_hour:02d}:00 น.")
                if peak_hour >= 20 or peak_hour <= 5:
                    recs.append("🌙 แนะนำหลีกเลี่ยงการเดินทางช่วงกลางคืน")
                elif 7 <= peak_hour <= 9:
                    recs.append("🚗 ชั่วโมงเร่งด่วนเช้า — เผื่อเวลาและขับขี่อย่างใจเย็น")
                elif 17 <= peak_hour <= 19:
                    recs.append("🚗 ชั่วโมงเร่งด่วนเย็น — ระวังรถติดและผู้ขับขี่เหนื่อยล้า")

        if "Weather_Conditions" in df_filtered.columns and len(df_filtered) > 0:
            rain_acc = (
                df_filtered["Weather_Conditions"]
                .astype(str).str.lower()
                .str.contains("rain", na=False)
                .sum()
            )
            if rain_acc > len(df_filtered) * 0.3:
                recs.append("🌧️ พื้นที่นี้มีอุบัติเหตุสูงช่วงฝนตก — ลดความเร็วและเพิ่มระยะห่าง")

        if "Road_Surface_Conditions" in df_filtered.columns:
            wet_acc = (
                df_filtered["Road_Surface_Conditions"]
                .astype(str).str.lower()
                .str.contains("wet|damp", na=False)
                .sum()
            )
            if wet_acc > 10:
                recs.append("💧 มีประวัติอุบัติเหตุบนถนนเปียก — ระวังถนนลื่น")

        return recs

    # ── Public helpers ────────────────────────────────────────────

    def get_hotspots(self, limit: int = 100) -> list:
        top = (
            self.df.groupby(["Latitude", "Longitude"])
            .agg(
                count=("Accident_Severity", "count"),
                avg_severity=("Accident_Severity", "mean"),
            )
            .reset_index()
            .sort_values("count", ascending=False)
            .head(limit)
        )
        return top.to_dict(orient="records")

    def get_dataset_stats(self) -> dict:
        stats: Dict = {"total_accidents": len(self.df)}

        if "Accident_Severity" in self.df.columns:
            vc = self.df["Accident_Severity"].value_counts().to_dict()
            stats["by_severity"] = {
                SEVERITY_LABEL.get(k, str(k)): int(v)
                for k, v in vc.items()
            }

        if "Hour" in self.df.columns:
            non_null = self.df["Hour"].dropna()
            stats["peak_hour"] = int(non_null.mode()[0]) if not non_null.empty else None

        return stats