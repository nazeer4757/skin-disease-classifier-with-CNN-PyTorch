    # api/schemas.py
from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    predicted_class   : str
    label             : str
    confidence        : float
    all_probabilities : Dict[str, float]
    is_unknown        : bool

class HealthResponse(BaseModel):
    status  : str
    model   : str
    device  : str
    classes : int