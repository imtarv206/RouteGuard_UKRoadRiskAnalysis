from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from risk import RiskAnalyzer

app = FastAPI(title="Route Risk Assessment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset and ML model on startup
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Accident_Information_clean.csv"
MODEL_FILE = BASE_DIR / "accident_model.joblib"

analyzer = RiskAnalyzer(str(DATA_FILE), model_path=str(MODEL_FILE))


class Coordinate(BaseModel):
    lat: float
    lng: float


class RouteRequest(BaseModel):
    route: List[Coordinate]
    time_from: Optional[int] = None   # 0-23 or None
    time_to: Optional[int] = None     # 0-23 or None
    day_of_week: str = "any"          # Monday-Sunday / Mon-Sun / any
    weather: str = "any"              # Fine / Rain / Fog / any


class RiskPoint(BaseModel):
    lat: float
    lng: float
    risk_level: str
    score: float
    nearby_accidents: int


class RouteRiskResponse(BaseModel):
    overall_score: float
    risk_level: str
    risk_points: List[RiskPoint]
    stats: dict
    recommendations: List[str]


@app.get("/")
def root():
    return {
        "message": "Route Risk Assessment API",
        "accidents_loaded": len(analyzer.df)
    }


@app.post("/analyze", response_model=RouteRiskResponse)
def analyze_route(req: RouteRequest):
    if len(req.route) < 2:
        raise HTTPException(status_code=400, detail="Route must have at least 2 points")

    if req.time_from is not None and not (0 <= req.time_from <= 23):
        raise HTTPException(status_code=400, detail="time_from must be between 0 and 23")

    if req.time_to is not None and not (0 <= req.time_to <= 23):
        raise HTTPException(status_code=400, detail="time_to must be between 0 and 23")

    route_coords = [(p.lat, p.lng) for p in req.route]

    result = analyzer.analyze(
        route=route_coords,
        time_from=req.time_from,
        time_to=req.time_to,
        day_of_week=req.day_of_week,
        weather=req.weather,
    )
    return result


@app.get("/hotspots")
def get_hotspots(limit: int = 100):
    hotspots = analyzer.get_hotspots(limit)
    return {"hotspots": hotspots}


@app.get("/stats")
def get_stats():
    return analyzer.get_dataset_stats()