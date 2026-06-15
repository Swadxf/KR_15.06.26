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
from market_geo_enrichment.blend import apply_policy, build_policy
from market_geo_enrichment.comparables import (
    REFERENCE_COLUMNS,
    predict_comparables,
)
from market_geo_enrichment.geo_features import build_geo_dataset

from .config import (
    ARTIFACTS_DIR,
    COMPARABLES_REFERENCE_PATH,
    FEATURE_CONFIG_PATH,
    FULL_MARKET_METRICS_PATH,
    FULL_MARKET_TEST_PREDICTIONS_PATH,
    MAX_PRICE_RUB,
    METRICS_PATH,
    MODELS_DIR,
    POLICIES_PATH,
    REMOVED_ROWS_PATH,
    SPLIT_REPORT_PATH,
    TEST_PREDICTIONS_PATH,
    TRAIN_READY_PATH,
    VALIDATION_PREDICTIONS_PATH,
)


def filtered_splits(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, valid, test, train_valid = split_indices(len(frame))
    allowed = frame["price_rub"].le(MAX_PRICE_RUB).to_numpy()
    train = train[allowed[train]]
    valid = valid[allowed[valid]]
    test = test[allowed[test]]
    train_valid = train_valid[allowed[train_valid]]
    return train, valid, test, train_valid


def save_filter_reports(
    ready: pd.DataFrame,
    train: np.ndarray,
    valid: np.ndarray,
    test: np.ndarray,
) -> None:
    removed = ready.loc[
        ready["price_rub"].gt(MAX_PRICE_RUB),
        [
            "source",
            "housing_market",
            "property_format",
            "price_rub",
            "total_area_m2",
            "rooms",
            "address_exact",
            "residential_complex",
        ],
    ].copy()
    removed.insert(0, "row_index", removed.index)
    removed.to_csv(
        REMOVED_ROWS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "filter": f"price_rub <= {int(MAX_PRICE_RUB)}",
        "strictly_over_40m_removed": True,
        "exactly_40m_retained": True,
        "full_clean_rows": int(len(ready)),
        "removed_rows": int(len(removed)),
        "retained_rows": int(len(ready) - len(removed)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
    }
    SPLIT_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_pipeline(
    seeds: list[int],
    train_production: bool = True,
) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prepared = build_geo_dataset()
    ready, feature_config, _ = prepare_model_frame(prepared)
    train, valid, test, train_valid = filtered_splits(ready)
    save_filter_reports(ready, train, valid, test)

    retained = ready.loc[ready["price_rub"].le(MAX_PRICE_RUB)].copy()
    retained.insert(0, "original_row_index", retained.index)
    retained.to_csv(
        TRAIN_READY_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    FEATURE_CONFIG_PATH.write_text(
        json.dumps(feature_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation_parts = {}
    for spec in [TOTAL_SPEC, UNIT_SPEC]:
        models = train_ensemble(
            ready,
            train,
            feature_config,
            spec,
            seeds,
        )
        validation_parts[spec.name] = predict_ensemble(
            models,
            ready,
            valid,
            feature_config,
            spec,
        )

    actual_valid = ready.iloc[valid]["price_rub"].to_numpy(dtype=float)
    geo_policy = build_blend_policy(
        actual_valid,
        validation_parts[TOTAL_SPEC.name],
        validation_parts[UNIT_SPEC.name],
    )
    geo_validation = apply_blend_policy(
        geo_policy,
        validation_parts[TOTAL_SPEC.name],
        validation_parts[UNIT_SPEC.name],
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

    test_parts = {}
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
        test_parts[spec.name] = predict_ensemble(
            models,
            ready,
            test,
            feature_config,
            spec,
        )

    actual_test = ready.iloc[test]["price_rub"].to_numpy(dtype=float)
    geo_test = apply_blend_policy(
        geo_policy,
        test_parts[TOTAL_SPEC.name],
        test_parts[UNIT_SPEC.name],
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
        "max_price_rub": MAX_PRICE_RUB,
        "seeds": seeds,
        "geo_model_policy": geo_policy,
        "final_comparable_policy": final_policy,
    }
    POLICIES_PATH.write_text(
        json.dumps(policies, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "filter": {
            "max_price_rub": MAX_PRICE_RUB,
            "removed_rows": int(
                ready["price_rub"].gt(MAX_PRICE_RUB).sum()
            ),
            "exactly_40m_retained": int(
                ready["price_rub"].eq(MAX_PRICE_RUB).sum()
            ),
        },
        "test": segmented_metrics(test_output),
        "validation": segmented_metrics(validation_output),
    }

    if FULL_MARKET_TEST_PREDICTIONS_PATH.exists():
        previous = pd.read_csv(
            FULL_MARKET_TEST_PREDICTIONS_PATH
        ).set_index("row_index")
        previous_same_rows = previous.loc[test].copy()
        previous_metrics = segmented_metrics(
            previous_same_rows.reset_index()
        )
        report["full_market_model_on_same_test_rows"] = previous_metrics
        report["retraining_improvement_mae_rub"] = float(
            previous_metrics["overall"]["mae_rub"]
            - report["test"]["overall"]["mae_rub"]
        )

    if FULL_MARKET_METRICS_PATH.exists():
        full_report = json.loads(
            FULL_MARKET_METRICS_PATH.read_text(encoding="utf-8")
        )
        report["archived_full_market_overall"] = full_report["test"][
            "overall"
        ]

    METRICS_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if train_production:
        production_indices = np.flatnonzero(
            ready["price_rub"].le(MAX_PRICE_RUB).to_numpy()
        )
        for spec in [TOTAL_SPEC, UNIT_SPEC]:
            models = train_ensemble(
                ready,
                production_indices,
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
        ready.iloc[production_indices][REFERENCE_COLUMNS].to_csv(
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
