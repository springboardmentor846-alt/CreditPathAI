# api/main.py
import os
import json
import sqlite3
import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- CONFIG ----------
MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_baseline.pkl")
FEATURE_LIST_PATH = os.getenv("FEATURE_LIST_PATH", "models/feature_list.txt")
SQLITE_PATH = os.getenv("LOG_DB_PATH", "models/logs.sqlite")
SAVE_SHAP = os.getenv("SAVE_SHAP", "true").lower() in ("1", "true", "yes")
ALLOWED_ORIGINS = os.getenv("API_ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
# ---------- END CONFIG ----------

logger = logging.getLogger("creditpathai")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="CreditPathAI API")

# CORS
origins = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB helper ----------
def ensure_db():
    os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT DEFAULT (datetime('now')),
        pred_prob REAL,
        risk_bucket TEXT,
        recommended_action TEXT,
        top_shap TEXT,
        input_json TEXT
    );""")
    conn.commit()
    conn.close()

def save_prediction(pred_prob: float, bucket: str, action: str, top_shap: Optional[List[str]], input_json: Dict[str, Any]):
    ensure_db()
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO predictions (pred_prob, risk_bucket, recommended_action, top_shap, input_json) VALUES (?, ?, ?, ?, ?)",
                (float(pred_prob), bucket, action, json.dumps(top_shap), json.dumps(input_json)))
    conn.commit()
    conn.close()

def fetch_logs(limit:int=200, offset:int=0, min_prob:Optional[float]=None):
    ensure_db()
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if min_prob is not None:
        cur.execute("SELECT * FROM predictions WHERE pred_prob >= ? ORDER BY id DESC LIMIT ? OFFSET ?", (min_prob, limit, offset))
    else:
        cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "ts": r["ts"],
            "pred_prob": r["pred_prob"],
            "risk_bucket": r["risk_bucket"],
            "recommended_action": r["recommended_action"],
            "top_shap": json.loads(r["top_shap"]) if r["top_shap"] else None,
            "input": json.loads(r["input_json"]) if r["input_json"] else None
        })
    return {"count": len(results), "results": results}

# ---------- Model loading ----------
MODEL = None
FEATURE_COLS = None

def load_model():
    global MODEL, FEATURE_COLS
    # Load model
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model file not found at {MODEL_PATH}. Prediction will fail until model is available.")
    else:
        MODEL = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}: {type(MODEL)}")

    # Load feature list if exists
    if os.path.exists(FEATURE_LIST_PATH):
        with open(FEATURE_LIST_PATH, "r", encoding="utf8") as f:
            FEATURE_COLS = [ln.strip() for ln in f.readlines() if ln.strip()]
        logger.info(f"Loaded {len(FEATURE_COLS)} feature columns from {FEATURE_LIST_PATH}")
    else:
        FEATURE_COLS = None
        logger.warning(f"Feature list not found at {FEATURE_LIST_PATH}. Will fallback to numeric columns when predicting.")

load_model()

# ---------- Helpers ----------
def build_features_from_input(data: Dict[str, Any]):
    """
    Given the incoming data dict, produce a pandas DataFrame with columns expected by model.
    Uses FEATURE_COLS if available, otherwise returns numeric-only columns from provided dict.
    """
    if FEATURE_COLS:
        # create dataframe with all feature cols, use zeros for missing
        row = {c: data.get(c, 0) for c in FEATURE_COLS}
        return pd.DataFrame([row])
    else:
        # fallback: use numeric values from provided dict
        df = pd.DataFrame([data])
        df = df.select_dtypes(include=[np.number])
        return df.fillna(0)

def bucket_for_prob(p: float):
    # buckets: low (0-0.6), medium (0.6-0.9), high (0.9-0.98), very_high (0.98-1.0)
    if p < 0.6: return "low"
    if p < 0.9: return "medium"
    if p < 0.98: return "high"
    return "very_high"

def recommended_action_for_bucket(b: str):
    return {
        "low": "No action (monitor)",
        "medium": "Low-touch outreach",
        "high": "Review & automated email",
        "very_high": "Escalate to collections - High touch agent call"
    }.get(b, "Review")

# ---------- Pydantic schemas ----------
class PredictRequest(BaseModel):
    data: Dict[str, Any]

class PredictResponse(BaseModel):
    default_probability: float
    risk_bucket: str
    recommended_action: str
    top_shap: Optional[List[str]] = None

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "service": "CreditPathAI API"}

@app.get("/")
def root():
    return {"status": "ok", "message": "CreditPathAI API. Use /docs to test POST /predict"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, shap_top_k: int = Query(0, ge=0, le=20)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    data = payload.data
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Payload data must be a JSON object of features.")

    # Build features
    X = build_features_from_input(data)
    if X.shape[1] == 0:
        # no numeric features detected
        raise HTTPException(status_code=422, detail="No numeric features found in payload. Provide numeric features or ensure feature_list.txt exists on server.")

    try:
        # if sklearn pipeline with predict_proba:
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(X)
            # if binary classification, take class 1 probability
            if probs.ndim == 2 and probs.shape[1] >= 2:
                p = float(probs[:,1][0])
            else:
                # fallback: mean of probabilities
                p = float(np.mean(probs))
        else:
            # fallback to predict (0/1)
            pred = MODEL.predict(X)
            p = float(pred[0])
    except Exception as e:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    bucket = bucket_for_prob(p)
    action = recommended_action_for_bucket(bucket)

    # Compute SHAP if requested and supported
    top_shap = None
    if shap_top_k and SAVE_SHAP:
        try:
            import shap
            # prepare background using zeros or small sample
            if FEATURE_COLS:
                X_for_shap = X.fillna(0)
            else:
                X_for_shap = X.fillna(0)
            # choose explainer type
            if hasattr(MODEL, "feature_importances_") or type(MODEL).__name__.lower().startswith("xgb") or "lgb" in type(MODEL).__name__.lower():
                explainer = shap.TreeExplainer(MODEL)
                shap_values = explainer.shap_values(X_for_shap)
                shap_arr = shap_values[0] if isinstance(shap_values, list) else shap_values
            else:
                explainer = shap.Explainer(MODEL.predict, X_for_shap)
                shap_values = explainer(X_for_shap)
                shap_arr = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)

            importances = np.abs(shap_arr).mean(axis=0)
            feat_imp = sorted(zip(X_for_shap.columns.tolist(), importances.tolist()), key=lambda x: x[1], reverse=True)
            top_shap = [f"{f}:{imp:.4f}" for f,imp in feat_imp[:shap_top_k]]
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            top_shap = None

    # Save to logs DB (non-blocking pattern isn't necessary here)
    try:
        save_prediction(p, bucket, action, top_shap, data)
    except Exception as e:
        logger.warning(f"Saving prediction to DB failed: {e}")

    return {
        "default_probability": p,
        "risk_bucket": bucket,
        "recommended_action": action,
        "top_shap": top_shap
    }

@app.get("/logs")
def logs(limit: int = 200, offset: int = 0, min_prob: Optional[float] = None):
    return fetch_logs(limit=limit, offset=offset, min_prob=min_prob)