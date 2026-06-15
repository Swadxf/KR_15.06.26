import argparse
import json

import numpy as np
import pandas as pd

from mae_optimization.evaluation import segmented_metrics
from mae_optimization.modeling import (
    TOTAL_SPEC,
    UNIT_SPEC,
    predict_ensemble,
    prepare_model_frame,
    save_models,
    train_ensemble,
)
from mae_optimization.train_models import (
    apply_blend_policy,
    build_blend_policy,
    parse_seeds,
    split_indices,
)

from .blend import apply_policy, build_policy
from .comparables import REFERENCE_COLUMNS, predict_comparables
from .config import (
    ARTIFACTS_DIR,
    BASE_TEST_PREDICTIONS_PATH,
    COMPARABLES_REFERENCE_PATH,
    ENRICHED_PREPARED_PATH,
    FEATURE_CONFIG_PATH,
    METRICS_PATH,
    MODELS_DIR,
    POLICIES_PATH,
    TEST_PREDICTIONS_PATH,
    TRAIN_READY_PATH,
    VALIDATION_PREDICTIONS_PATH,
)
from .geo_features import build_geo_dataset


def train_pipeline(
    seeds: list[int],
    train_production: bool = True,
) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prepared = build_geo_dataset()
    ready, feature_config, _ = prepare_model_frame(prepared)
    ready.to_csv(TRAIN_READY_PATH, index=False, encoding="utf-8-sig")
    FEATURE_CONFIG_PATH.write_text(
        json.dumps(feature_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train, valid, test, train_valid = split_indices(len(ready))
    validation_model_parts = {}
    for spec in [TOTAL_SPEC, UNIT_SPEC]:
        models = train_ensemble(
            ready,
            train,
            feature_config,
            spec,
            seeds,
        )
        validation_model_parts[spec.name] = predict_ensemble(
            models,
            ready,
            valid,
            feature_config,
            spec,
        )

    actual_valid = ready.iloc[valid]["price_rub"].to_numpy(dtype=float)
    geo_model_policy = build_blend_policy(
        actual_valid,
        validation_model_parts[TOTAL_SPEC.name],
        validation_model_parts[UNIT_SPEC.name],
    )
    geo_validation = apply_blend_policy(
        geo_model_policy,
        validation_model_parts[TOTAL_SPEC.name],
        validation_model_parts[UNIT_SPEC.name],
    )
    comparable_validation, valid_diagnostics = predict_comparables(
        ready.iloc[train],
        ready.iloc[valid],
    )
    final_policy = build_policy(
        actual_valid,
        geo_validation,
        comparable_validation,
    )
    final_validation = apply_policy(
        final_policy,
        geo_validation,
        comparable_validation,
    )

    validation_output = pd.DataFrame(
        {
            "row_index": valid,
            "price_rub": actual_valid,
            "geo_model_prediction_rub": geo_validation,
            "comparable_prediction_rub": comparable_validation,
            "prediction_price_rub": final_validation,
        }
    )
    validation_output = pd.concat(
        [validation_output, valid_diagnostics],
        axis=1,
    )
    validation_output.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    test_model_parts = {}
    for spec in [TOTAL_SPEC, UNIT_SPEC]:
        models = train_ensemble(
            ready,
            train_valid,
            feature_config,
            spec,
            seeds,
        )
        save_models(
            models,
            MODELS_DIR / "evaluation",
            spec,
            seeds,
        )
        test_model_parts[spec.name] = predict_ensemble(
            models,
            ready,
            test,
            feature_config,
            spec,
        )

    actual_test = ready.iloc[test]["price_rub"].to_numpy(dtype=float)
    geo_test = apply_blend_policy(
        geo_model_policy,
        test_model_parts[TOTAL_SPEC.name],
        test_model_parts[UNIT_SPEC.name],
    )
    comparable_test, test_diagnostics = predict_comparables(
        ready.iloc[train_valid],
        ready.iloc[test],
    )
    final_test = apply_policy(
        final_policy,
        geo_test,
        comparable_test,
    )

    output_columns = [
        column
        for column in ["source", "housing_market", "property_format"]
        if column in ready.columns
    ]
    test_output = ready.iloc[test][
        output_columns + ["price_rub"]
    ].reset_index(drop=True)
    test_output.insert(0, "row_index", test)
    test_output["geo_model_prediction_rub"] = geo_test
    test_output["comparable_prediction_rub"] = comparable_test
    test_output["prediction_price_rub"] = final_test
    test_output["abs_error_rub"] = np.abs(actual_test - final_test)
    test_output = pd.concat([test_output, test_diagnostics], axis=1)
    test_output.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    policies = {
        "seeds": seeds,
        "geo_model_policy": geo_model_policy,
        "final_comparable_policy": final_policy,
    }
    POLICIES_PATH.write_text(
        json.dumps(policies, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "test": segmented_metrics(test_output),
        "validation": segmented_metrics(validation_output),
    }
    if BASE_TEST_PREDICTIONS_PATH.exists():
        base_test = pd.read_csv(BASE_TEST_PREDICTIONS_PATH).set_index(
            "row_index"
        )
        aligned_base = base_test.loc[test, "prediction_price_rub"].to_numpy()
        base_output = test_output.copy()
        base_output["prediction_price_rub"] = aligned_base
        report["previous_pipeline_test"] = segmented_metrics(base_output)[
            "overall"
        ]
        report["improvement_vs_previous_mae_rub"] = float(
            report["previous_pipeline_test"]["mae_rub"]
            - report["test"]["overall"]["mae_rub"]
        )

    METRICS_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if train_production:
        all_indices = np.arange(len(ready))
        for spec in [TOTAL_SPEC, UNIT_SPEC]:
            models = train_ensemble(
                ready,
                all_indices,
                feature_config,
                spec,
                seeds,
            )
            save_models(
                models,
                MODELS_DIR / "production",
                spec,
                seeds,
            )
        ready[REFERENCE_COLUMNS].to_csv(
            COMPARABLES_REFERENCE_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--skip-production", action="store_true")
    args = parser.parse_args()
    report = train_pipeline(
        seeds=parse_seeds(args.seeds),
        train_production=not args.skip_production,
    )
    overall = report["test"]["overall"]
    print(
        f"MAE={overall['mae_rub']:,.0f} RUB, "
        f"MedAE={overall['median_ae_rub']:,.0f} RUB, "
        f"MAPE={overall['mape_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
