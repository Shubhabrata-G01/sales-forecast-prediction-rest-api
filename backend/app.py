"""FastAPI backend that serves SuperKart sales-revenue predictions."""

import io

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

MODEL_PATH = "superkart_model.joblib"
model = joblib.load(MODEL_PATH)

REQUIRED_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

superkart_api = FastAPI(
    title="SuperKart Sales Prediction API",
    description="Predicts the quarterly sales revenue of a product at a SuperKart outlet.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    Product_Weight: float
    Product_Sugar_Content: str
    Product_Allocated_Area: float
    Product_MRP: float
    Store_Size: str
    Store_Location_City_Type: str
    Store_Type: str
    Product_Id_char: str
    Store_Age_Years: int
    Product_Type_Category: str


@superkart_api.get("/")
def read_root():
    return {"status": "ok", "message": "SuperKart Sales Prediction API is running."}


@superkart_api.get("/health")
def health_check():
    return {"status": "healthy"}


@superkart_api.post("/v1/predict")
def predict(payload: PredictionRequest):
    """Online inference for a single product/store combination."""
    input_df = pd.DataFrame([payload.model_dump()])
    try:
        prediction = model.predict(input_df)[0]
    except Exception as exc:  # noqa: BLE001 - surface model errors as a client-facing 400
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return {"Product_Store_Sales_Total": round(float(prediction), 2)}


@superkart_api.post("/v1/predictbatch")
def predict_batch(file: UploadFile = File(...)):
    """Batch inference for a CSV file containing one or more rows of features."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = file.file.read()
        batch_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:  # noqa: BLE001 - malformed upload is a client error
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {exc}") from exc

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in batch_df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing_columns)}",
        )

    try:
        predictions = model.predict(batch_df[REQUIRED_COLUMNS])
    except Exception as exc:  # noqa: BLE001 - surface model errors as a client-facing 400
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return {str(idx): round(float(pred), 2) for idx, pred in enumerate(predictions)}
