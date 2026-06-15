import math

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    absolute_error = np.abs(actual - prediction)
    nonzero = actual != 0
    mape = (
        float(np.mean(absolute_error[nonzero] / actual[nonzero]) * 100)
        if nonzero.any()
        else None
    )
    return {
        "mae_rub": float(mean_absolute_error(actual, prediction)),
        "median_ae_rub": float(median_absolute_error(actual, prediction)),
        "rmse_rub": float(math.sqrt(mean_squared_error(actual, prediction))),
        "mape_pct": mape,
        "r2": float(r2_score(actual, prediction)) if len(actual) >= 2 else None,
        "bias_rub": float(np.mean(prediction - actual)),
        "p90_abs_error_rub": float(np.quantile(absolute_error, 0.90)),
        "p95_abs_error_rub": float(np.quantile(absolute_error, 0.95)),
    }


def apply_geo_blend_policy(
    policy: dict,
    total_prediction: np.ndarray,
    unit_prediction: np.ndarray,
) -> np.ndarray:
    reference = (total_prediction + unit_prediction) / 2
    prediction = np.zeros_like(reference, dtype=float)
    for band in policy["bands"]:
        lower = float(band["lower"])
        upper = (
            float("inf")
            if band["upper"] is None
            else float(band["upper"])
        )
        mask = (reference >= lower) & (reference < upper)
        total_weight = float(band["total_weight"])
        prediction[mask] = (
            total_weight * total_prediction[mask]
            + (1 - total_weight) * unit_prediction[mask]
        )
    return prediction


def apply_comparable_policy(
    policy: dict,
    model_prediction: np.ndarray,
    comparable_prediction: np.ndarray,
) -> np.ndarray:
    reference = (model_prediction + comparable_prediction) / 2
    prediction = np.zeros_like(reference, dtype=float)
    for band in policy["bands"]:
        lower = float(band["lower"])
        upper = (
            float("inf")
            if band["upper"] is None
            else float(band["upper"])
        )
        mask = (reference >= lower) & (reference < upper)
        weight = float(band["comparable_weight"])
        prediction[mask] = (
            (1 - weight) * model_prediction[mask]
            + weight * comparable_prediction[mask]
        )
    return prediction
