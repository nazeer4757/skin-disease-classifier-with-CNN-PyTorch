# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import io

from model import load_model, predict, CLASS_NAMES, DEVICE
from schemas import PredictionResponse, HealthResponse

# global model variable
ml_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model once at startup
    global ml_model
    ml_model = load_model()
    yield
    # cleanup on shutdown
    ml_model = None

app = FastAPI(
    title       = "Skin Disease Classifier API",
    description = "EfficientNet-B4 trained on HAM10000 — 7 skin disease classes",
    version     = "1.0.0",
    lifespan    = lifespan
)

# allow Streamlit to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status" : "ok",
        "model"  : "EfficientNet-B4",
        "device" : str(DEVICE),
        "classes": len(CLASS_NAMES)
    }


@app.get("/classes")
def get_classes():
    return {
        "classes": CLASS_NAMES,
        "total"  : len(CLASS_NAMES)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    # validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code = 400,
            detail      = "Only JPEG and PNG images are supported"
        )

    # read image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code = 400,
            detail      = "Invalid image file"
        )

    # predict
    result = predict(ml_model, image)
    return result