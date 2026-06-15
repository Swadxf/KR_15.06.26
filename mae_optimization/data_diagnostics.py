import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .config import (
    DATA_DIAGNOSTICS_PATH,
    EXCLUDED_PATH,
    PREPARED_PATH,
    TRAIN_READY_PATH,
)


def leave_one_out_group_median(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> dict:
    prediction = np.full(len(frame), np.nan)
    for _, group in frame.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        if len(group) < 2:
            continue
        indices = group.index.to_numpy()
        prices = group["price_rub"].to_numpy(dtype=float)
        for position, row_index in enumerate(indices):
            prediction[row_index] = np.median(
                np.delete(prices, position)
            )

    available = np.isfinite(prediction)
    return {
        "rows": int(available.sum()),
        "coverage_pct": float(available.mean() * 100),
        "leave_one_out_median_mae_rub": float(
            mean_absolute_error(
                frame.loc[available, "price_rub"],
                prediction[available],
            )
        ),
    }


def leakage_reconstruction_mae() -> dict:
    prepared = pd.read_csv(
        PREPARED_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )
    excluded = pd.read_csv(
        EXCLUDED_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )
    excluded_keys = set(
        zip(
            excluded["source"].astype(str),
            excluded["source_listing_id"].astype(str),
        )
    )
    included = [
        (str(source), str(listing_id)) not in excluded_keys
        for source, listing_id in zip(
            prepared["source"],
            prepared["source_listing_id"],
        )
    ]
    frame = prepared.loc[included].copy()
    valid = (
        frame["price_m2_rub"].notna()
        & frame["total_area_m2"].notna()
        & frame["price_rub"].notna()
    )
    reconstructed = (
        frame.loc[valid, "price_m2_rub"]
        * frame.loc[valid, "total_area_m2"]
    )
    return {
        "rows": int(valid.sum()),
        "mae_rub": float(
            mean_absolute_error(
                frame.loc[valid, "price_rub"],
                reconstructed,
            )
        ),
        "warning": (
            "price_m2_rub is calculated from the target and must not be used "
            "as a model feature"
        ),
    }


def build_diagnostics() -> dict:
    frame = pd.read_csv(
        TRAIN_READY_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )
    frame["area_rounded_m2"] = frame["total_area_m2"].round(0)

    report = {
        "clean_rows": int(len(frame)),
        "near_duplicate_consistency": {
            "same_address_rooms_area": leave_one_out_group_median(
                frame,
                ["address_exact", "rooms", "area_rounded_m2"],
            ),
            "same_address_rooms_area_floor": leave_one_out_group_median(
                frame,
                ["address_exact", "rooms", "area_rounded_m2", "floor"],
            ),
            "interpretation": (
                "Even nearly identical listings have substantial price "
                "dispersion; these figures are diagnostic proxies, not a "
                "formal lower bound."
            ),
        },
        "target_leakage_check": leakage_reconstruction_mae(),
    }
    DATA_DIAGNOSTICS_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(build_diagnostics(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
