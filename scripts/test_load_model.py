# scripts/test_load_model.py
import os, joblib, sys

MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_baseline.pkl")
FEAT_PATH  = os.getenv("FEATURE_LIST_PATH", "models/feature_list.txt")

print("Testing model load")
print("Model path:", MODEL_PATH)
print("Feature list path:", FEAT_PATH)

if not os.path.exists(MODEL_PATH):
    print("ERROR: model file not found at", MODEL_PATH)
    sys.exit(2)

try:
    m = joblib.load(MODEL_PATH)
    print("Model loaded OK. Type:", type(m))
    # print whether it has predict_proba or feature_importances_
    print("Has predict_proba:", hasattr(m, "predict_proba"))
    print("Has feature_importances_:", hasattr(m, "feature_importances_"))
except Exception as e:
    print("ERROR loading model:", repr(e))
    raise