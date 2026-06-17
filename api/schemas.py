# api/schemas.py
from pydantic import BaseModel
from typing import Optional, Dict


class PredictionResponse(BaseModel):
    status            : str                       # NEW: "success" | "rejected_human" | "rejected_invalid"
    predicted_class   : Optional[str]  = None
    label             : Optional[str]  = None
    confidence        : Optional[float] = None
    all_probabilities : Optional[Dict[str, float]] = None
    is_unknown        : bool
    message           : Optional[str]  = None      # NEW: user-facing rejection message


class HealthResponse(BaseModel):
    status : str
    model  : str
    device : str
    classes: int