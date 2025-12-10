"""
Deployment helpers: load saved model and make predictions.
"""

from typing import Any, Dict

import joblib
import pandas as pd

from .config import BEST_MODEL_PATH

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(BEST_MODEL_PATH)
    return _model


def predict_single(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    input_data: dict of feature_name -> value
    returns: { "prediction": label, "churn_probability": float }
    """
    model = load_model()
    df = pd.DataFrame([input_data])
    proba = model.predict_proba(df)[0, 1]
    pred = model.predict(df)[0]
    return {"prediction": int(pred) if isinstance(pred, (int, bool)) else pred, "churn_probability": float(proba)}


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1]
    out = df.copy()
    out["churn_prediction"] = preds
    out["churn_probability"] = probas
    return out
