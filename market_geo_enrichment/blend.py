import numpy as np
from sklearn.metrics import mean_absolute_error

from .config import (
    FINAL_BLEND_EDGES_RUB,
    FINAL_BLEND_SHRINK_ROWS,
)


def choose_weight(
    actual: np.ndarray,
    model_prediction: np.ndarray,
    comparable_prediction: np.ndarray,
) -> tuple[float, float]:
    weights = np.linspace(0, 0.70, 71)
    scores = [
        mean_absolute_error(
            actual,
            (1 - weight) * model_prediction
            + weight * comparable_prediction,
        )
        for weight in weights
    ]
    position = int(np.argmin(scores))
    return float(weights[position]), float(scores[position])


def build_policy(
    actual: np.ndarray,
    model_prediction: np.ndarray,
    comparable_prediction: np.ndarray,
) -> dict:
    global_weight, global_mae = choose_weight(
        actual,
        model_prediction,
        comparable_prediction,
    )
    global_policy = {
        "name": "global",
        "validation_mae_rub": global_mae,
        "bands": [
            {
                "lower": 0.0,
                "upper": None,
                "comparable_weight": global_weight,
                "validation_rows": int(len(actual)),
            }
        ],
    }

    reference = (model_prediction + comparable_prediction) / 2
    prediction = np.zeros_like(reference, dtype=float)
    bands = []
    for lower, upper in zip(
        FINAL_BLEND_EDGES_RUB[:-1],
        FINAL_BLEND_EDGES_RUB[1:],
    ):
        mask = (reference >= lower) & (reference < upper)
        rows = int(mask.sum())
        if rows >= 30:
            raw_weight, _ = choose_weight(
                actual[mask],
                model_prediction[mask],
                comparable_prediction[mask],
            )
            comparable_weight = (
                raw_weight * rows
                + global_weight * FINAL_BLEND_SHRINK_ROWS
            ) / (rows + FINAL_BLEND_SHRINK_ROWS)
        else:
            comparable_weight = global_weight

        prediction[mask] = (
            (1 - comparable_weight) * model_prediction[mask]
            + comparable_weight * comparable_prediction[mask]
        )
        bands.append(
            {
                "lower": float(lower),
                "upper": None if np.isinf(upper) else float(upper),
                "comparable_weight": float(comparable_weight),
                "validation_rows": rows,
            }
        )

    band_mae = float(mean_absolute_error(actual, prediction))
    if band_mae < global_mae:
        return {
            "name": "price_bands",
            "validation_mae_rub": band_mae,
            "bands": bands,
        }
    return global_policy


def apply_policy(
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
