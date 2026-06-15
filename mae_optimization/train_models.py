import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from .config import (
    ARTIFACTS_DIR,
    BLEND_CONFIG_PATH,
    DEFAULT_SEEDS,
    EXCLUDED_PATH,
    FEATURE_CONFIG_PATH,
    MODELS_DIR,
    PREPARED_PATH,
    RANDOM_STATE,
    SPLIT_PATH,
    TEST_PREDICTIONS_PATH,
    TEST_SIZE,
    TRAIN_READY_PATH,
    VALID_SIZE,
    VALIDATION_PREDICTIONS_PATH,
)
from .modeling import (
    TOTAL_SPEC,
    UNIT_SPEC,
    predict_ensemble,
    prepare_model_frame,
    save_models,
    train_ensemble,
)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def choose_blend_weight(
    actual: np.ndarray,
    total_prediction: np.ndarray,
    unit_prediction: np.ndarray,
) -> tuple[float, float]:
    weights = np.linspace(0, 1, 101)
    scores = [
        mean_absolute_error(
            actual,
            weight * total_prediction + (1 - weight) * unit_prediction,
        )
        for weight in weights
    ]
    best_position = int(np.argmin(scores))
    return float(weights[best_position]), float(scores[best_position])


def build_blend_policy(
    actual: np.ndarray,
    total_prediction: np.ndarray,
    unit_prediction: np.ndarray,
) -> dict:
    global_weight, global_mae = choose_blend_weight(
        actual,
        total_prediction,
        unit_prediction,
    )
    candidates = [
        {
            "name": "global",
            "edges": [0.0, float("inf")],
        },
        {
            "name": "price_bands",
            "edges": [
                0.0,
                5_000_000.0,
                8_000_000.0,
                12_000_000.0,
                20_000_000.0,
                40_000_000.0,
                float("inf"),
            ],
        },
    ]

    reference = (total_prediction + unit_prediction) / 2
    best_policy = {
        "name": "global",
        "validation_mae_rub": global_mae,
        "bands": [
            {
                "lower": 0.0,
                "upper": None,
                "total_weight": global_weight,
                "validation_rows": int(len(actual)),
            }
        ],
    }

    for candidate in candidates[1:]:
        prediction = np.zeros_like(actual, dtype=float)
        bands = []
        for lower, upper in zip(candidate["edges"][:-1], candidate["edges"][1:]):
            mask = (reference >= lower) & (reference < upper)
            rows = int(mask.sum())
            if rows >= 30:
                raw_weight, _ = choose_blend_weight(
                    actual[mask],
                    total_prediction[mask],
                    unit_prediction[mask],
                )

                total_weight = (
                    raw_weight * rows + global_weight * 20
                ) / (rows + 20)
            else:
                total_weight = global_weight

            prediction[mask] = (
                total_weight * total_prediction[mask]
                + (1 - total_weight) * unit_prediction[mask]
            )
            bands.append(
                {
                    "lower": float(lower),
                    "upper": None if np.isinf(upper) else float(upper),
                    "total_weight": float(total_weight),
                    "validation_rows": rows,
                }
            )

        validation_mae = float(mean_absolute_error(actual, prediction))
        if validation_mae < best_policy["validation_mae_rub"]:
            best_policy = {
                "name": candidate["name"],
                "validation_mae_rub": validation_mae,
                "bands": bands,
            }

    return best_policy


def apply_blend_policy(
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


def split_indices(
    row_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(row_count)
    train_valid, test = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    relative_valid_size = VALID_SIZE / (1 - TEST_SIZE)
    train, valid = train_test_split(
        train_valid,
        test_size=relative_valid_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return train, valid, test, train_valid


def train_pipeline(seeds: list[int], train_production: bool = True) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prepared = pd.read_csv(PREPARED_PATH, encoding="utf-8-sig", low_memory=False)
    ready, config, excluded = prepare_model_frame(prepared)
    ready.to_csv(TRAIN_READY_PATH, index=False, encoding="utf-8-sig")
    excluded.to_csv(EXCLUDED_PATH, index=False, encoding="utf-8-sig")
    FEATURE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train, valid, test, train_valid = split_indices(len(ready))
    split = pd.DataFrame({"row_index": np.arange(len(ready)), "split": "train"})
    split.loc[valid, "split"] = "valid"
    split.loc[test, "split"] = "test"
    split.to_csv(SPLIT_PATH, index=False, encoding="utf-8-sig")

    validation_predictions = {}
    for spec in [TOTAL_SPEC, UNIT_SPEC]:
        models = train_ensemble(ready, train, config, spec, seeds)
        validation_predictions[spec.name] = predict_ensemble(
            models, ready, valid, config, spec
        )

    actual_valid = ready.iloc[valid]["price_rub"].to_numpy(dtype=float)
    blend_policy = build_blend_policy(
        actual_valid,
        validation_predictions[TOTAL_SPEC.name],
        validation_predictions[UNIT_SPEC.name],
    )
    validation_blend = apply_blend_policy(
        blend_policy,
        validation_predictions[TOTAL_SPEC.name],
        validation_predictions[UNIT_SPEC.name],
    )
    pd.DataFrame(
        {
            "row_index": valid,
            "price_rub": actual_valid,
            "prediction_total_rub": validation_predictions[TOTAL_SPEC.name],
            "prediction_unit_rub": validation_predictions[UNIT_SPEC.name],
            "prediction_price_rub": validation_blend,
        }
    ).to_csv(VALIDATION_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    blend_config = {
        "policy": blend_policy,
        "seeds": seeds,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "valid_size": VALID_SIZE,
    }
    BLEND_CONFIG_PATH.write_text(
        json.dumps(blend_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    test_predictions = {}
    for spec in [TOTAL_SPEC, UNIT_SPEC]:
        models = train_ensemble(ready, train_valid, config, spec, seeds)
        save_models(models, MODELS_DIR / "evaluation", spec, seeds)
        test_predictions[spec.name] = predict_ensemble(
            models, ready, test, config, spec
        )

    blended_test = apply_blend_policy(
        blend_policy,
        test_predictions[TOTAL_SPEC.name],
        test_predictions[UNIT_SPEC.name],
    )
    output_columns = [
        column
        for column in ["source", "housing_market", "property_format"]
        if column in ready.columns
    ]
    test_output = ready.iloc[test][output_columns + ["price_rub"]].copy()
    test_output.insert(0, "row_index", test)
    test_output["prediction_total_rub"] = test_predictions[TOTAL_SPEC.name]
    test_output["prediction_unit_rub"] = test_predictions[UNIT_SPEC.name]
    test_output["prediction_price_rub"] = blended_test
    test_output["abs_error_rub"] = np.abs(
        test_output["price_rub"] - test_output["prediction_price_rub"]
    )
    test_output.to_csv(TEST_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    if train_production:
        all_indices = np.arange(len(ready))
        for spec in [TOTAL_SPEC, UNIT_SPEC]:
            models = train_ensemble(ready, all_indices, config, spec, seeds)
            save_models(models, MODELS_DIR / "production", spec, seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated CatBoost seeds.",
    )
    parser.add_argument(
        "--skip-production",
        action="store_true",
        help="Do not train final models on all cleaned rows.",
    )
    args = parser.parse_args()
    train_pipeline(
        seeds=parse_seeds(args.seeds),
        train_production=not args.skip_production,
    )
    print(f"Models and predictions saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
