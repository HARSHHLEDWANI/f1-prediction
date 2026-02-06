from pydantic import BaseModel
from typing import Optional


# ---------- TOP 5 PROBABILITY ----------

class Top5PredictionRequest(BaseModel):
    race_id: str               # e.g. "monaco_2024"
    driver_id: str             # e.g. "leclerc"
    
    grid_position: int
    driver_elo: float
    constructor_strength: float
    track_affinity: float
    recent_form: float


# ---------- EXPECTED PERFORMANCE ----------

class ExpectedPerformanceRequest(BaseModel):
    race_id: str
    driver_id: str

    grid_position: int
    driver_elo: float
    constructor_strength: float
    track_affinity: float
    recent_form: float
