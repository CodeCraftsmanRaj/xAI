"""
main.py — FastAPI backend for the ResNet18 Explainable Image Classification app.

Run with:
    uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from model_utils import load_image, predict, explain

app = FastAPI(title="ResNet18 XAI API", version="1.0.0")

# Allow the Vite dev server (and any local frontend) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "ResNet18 XAI backend is running"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Returns top-5 ImageNet-1K predictions with confidence scores."""
    image_bytes = await file.read()
    img = load_image(image_bytes)
    results, _ = predict(img, topk=5)
    return {"predictions": results}


@app.post("/explain")
async def explain_endpoint(
    file: UploadFile = File(...),
    target_idx: int = Form(...),
    baseline_type: str = Form("black"),
    n_steps: int = Form(50),
):
    """
    Returns Integrated Gradients attribution visualizations for a chosen
    predicted class, computed against the requested baseline.
    """
    image_bytes = await file.read()
    img = load_image(image_bytes)
    result = explain(img, target_idx, baseline_type=baseline_type, n_steps=n_steps)
    return result
