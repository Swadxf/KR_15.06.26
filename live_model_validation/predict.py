import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .comparables import predict_comparables
from .config import (
    BLEND_POLICIES_PATH,
    COMPARABLES_REFERENCE_PATH,
    FEATURE_CONFIG_PATH,
    MANIFEST_PATH,
    MAX_TRAINING_PRICE_RUB,
    POIS_PATH,
    TOTAL_MODEL_PATH,
    UNIT_MODEL_PATH,
)
from .feature_pipeline import (
    make_model_matrix,
    prepare_fresh_features,
    read_raw,
)
from .runtime_utils import (
    apply_comparable_policy,
    apply_geo_blend_policy,
    regression_metrics,
)


def _load_model(path: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def verify_bundle() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        path = MANIFEST_PATH.parent / filename
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise RuntimeError(f"Model bundle checksum mismatch: {filename}")


def _predict_model(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
    categorical_features: set[str],
    target_mode: str,
) -> np.ndarray:
    matrix = make_model_matrix(
        frame,
        model.feature_names_,
        categorical_features,
    )
    forbidden = {"price_rub", "price_log_rub", "price_m2_rub"}
    leaked = forbidden.intersection(matrix.columns)
    if leaked:
        raise RuntimeError(
            f"Target leakage columns reached inference: {sorted(leaked)}"
        )
    prediction = np.expm1(model.predict(matrix))
    if target_mode == "unit":
        prediction *= frame["total_area_m2"].to_numpy(dtype=float)
    return np.maximum(prediction, 0)


def predict_raw_files(
    avito_raw_path: Path,
    cian_raw_path: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_bundle()
    avito = read_raw(avito_raw_path, "avito")
    cian = read_raw(cian_raw_path, "cian")
    pois = pd.read_csv(POIS_PATH, encoding="utf-8-sig", low_memory=False)
    reference = pd.read_csv(
        COMPARABLES_REFERENCE_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )
    frame = prepare_fresh_features(avito, cian, pois, reference)
    if frame.empty:
        raise RuntimeError("Parsers did not produce any raw listing rows.")

    feature_config = json.loads(
        FEATURE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    categorical_features = set(feature_config["cat_features"])
    policies = json.loads(
        BLEND_POLICIES_PATH.read_text(encoding="utf-8")
    )
    total_model = _load_model(TOTAL_MODEL_PATH)
    unit_model = _load_model(UNIT_MODEL_PATH)

    required = frame[["lat", "lon", "total_area_m2"]].notna().all(axis=1)
    valid_coordinates = (
        frame["lat"].between(54.0, 57.0)
        & frame["lon"].between(35.0, 41.5)
    )
    score_mask = required & valid_coordinates

    output = frame[
        [
            "source",
            "source_listing_id",
            "url",
            "price_rub",
            "lat",
            "lon",
            "rooms",
            "total_area_m2",
            "address_exact",
            "residential_complex",
        ]
    ].copy()
    output["prediction_total_rub"] = np.nan
    output["prediction_unit_rub"] = np.nan
    output["prediction_geo_model_rub"] = np.nan
    output["prediction_comparable_rub"] = np.nan
    output["prediction_price_rub"] = np.nan
    output["status"] = "skipped_missing_required_features"

    if score_mask.any():
        scored = frame.loc[score_mask].reset_index(drop=True)
        total_prediction = _predict_model(
            total_model,
            scored,
            categorical_features,
            "total",
        )
        unit_prediction = _predict_model(
            unit_model,
            scored,
            categorical_features,
            "unit",
        )
        geo_prediction = apply_geo_blend_policy(
            policies["geo_model_policy"],
            total_prediction,
            unit_prediction,
        )
        comparable_prediction, comparable_diagnostics = predict_comparables(
            reference,
            scored,
        )
        final_prediction = apply_comparable_policy(
            policies["final_comparable_policy"],
            geo_prediction,
            comparable_prediction,
        )
        indices = output.index[score_mask]
        output.loc[indices, "prediction_total_rub"] = total_prediction
        output.loc[indices, "prediction_unit_rub"] = unit_prediction
        output.loc[indices, "prediction_geo_model_rub"] = geo_prediction
        output.loc[indices, "prediction_comparable_rub"] = (
            comparable_prediction
        )
        output.loc[indices, "prediction_price_rub"] = final_prediction
        output.loc[indices, "comparable_nearest_distance_km"] = (
            comparable_diagnostics[
                "comparable_nearest_distance_km"
            ].to_numpy()
        )
        output.loc[indices, "status"] = "predicted"

    output["within_training_price_range"] = (
        output["price_rub"].le(MAX_TRAINING_PRICE_RUB)
    )
    output["outside_training_price_range"] = (
        output["price_rub"].gt(MAX_TRAINING_PRICE_RUB)
    )
    output["abs_error_rub"] = np.abs(
        output["price_rub"] - output["prediction_price_rub"]
    )
    output["absolute_percentage_error_pct"] = (
        output["abs_error_rub"] / output["price_rub"] * 100
    )
    output_path = output_dir / "fresh_predictions.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    predicted = output.loc[output["status"].eq("predicted")]
    report = {
        "parsed_rows": int(len(output)),
        "predicted_rows": int(len(predicted)),
        "skipped_rows": int(len(output) - len(predicted)),
        "model_training_max_price_rub": MAX_TRAINING_PRICE_RUB,
        "all_predicted_rows": None,
        "within_training_range": None,
        "outside_training_range_rows": int(
            predicted["outside_training_price_range"].sum()
        ),
        "predictions_csv": str(output_path),
    }
    evaluable = predicted.dropna(
        subset=["price_rub", "prediction_price_rub"]
    )
    if not evaluable.empty:
        report["all_predicted_rows"] = regression_metrics(
            evaluable["price_rub"],
            evaluable["prediction_price_rub"],
        )
    in_range = evaluable.loc[evaluable["within_training_price_range"]]
    if not in_range.empty:
        report["within_training_range"] = regression_metrics(
            in_range["price_rub"],
            in_range["prediction_price_rub"],
        )

    report_path = output_dir / "fresh_metrics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
