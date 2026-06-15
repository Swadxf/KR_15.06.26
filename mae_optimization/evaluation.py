import math

import numpy as np
import pandas as pd
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


def segmented_metrics(predictions: pd.DataFrame) -> dict:
    report = {
        "overall": regression_metrics(
            predictions["price_rub"],
            predictions["prediction_price_rub"],
        )
    }

    for column in ["source", "housing_market", "property_format"]:
        if column not in predictions.columns:
            continue
        report[column] = {}
        for value, group in predictions.groupby(column, dropna=False):
            report[column][str(value)] = {
                "rows": int(len(group)),
                **regression_metrics(
                    group["price_rub"],
                    group["prediction_price_rub"],
                ),
            }

    return report
