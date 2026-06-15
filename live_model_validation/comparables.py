import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .config import COMPARABLE_PARAMS, EARTH_RADIUS_M


REFERENCE_COLUMNS = [
    "price_rub",
    "lat",
    "lon",
    "total_area_m2",
    "rooms",
    "floor_ratio",
    "house_year",
    "housing_market",
    "property_format",
    "source",
    "address_exact",
    "residential_complex",
]


def _string_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].fillna("unknown").astype(str).to_numpy()


def predict_comparables(
    reference: pd.DataFrame,
    query: pd.DataFrame,
    params: dict | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    settings = dict(COMPARABLE_PARAMS)
    if params:
        settings.update(params)

    reference = reference.reset_index(drop=True)
    query = query.reset_index(drop=True)
    candidate_count = min(
        int(settings["candidate_count"]),
        len(reference),
    )
    neighbor_count = min(
        int(settings["neighbor_count"]),
        candidate_count,
    )

    tree = BallTree(
        np.radians(reference[["lat", "lon"]].to_numpy()),
        metric="haversine",
    )
    distance_radians, candidate_indices = tree.query(
        np.radians(query[["lat", "lon"]].to_numpy()),
        k=candidate_count,
    )
    distance_km = distance_radians * EARTH_RADIUS_M / 1000

    predictions = np.zeros(len(query), dtype=float)
    nearest_distance = np.zeros(len(query), dtype=float)
    exact_match_count = np.zeros(len(query), dtype=int)

    reference_area = reference["total_area_m2"].to_numpy(dtype=float)
    reference_price_per_m2 = (
        reference["price_rub"].to_numpy(dtype=float) / reference_area
    )
    reference_rooms = reference["rooms"].to_numpy(dtype=float)
    reference_year = reference["house_year"].to_numpy(dtype=float)
    reference_market = _string_array(reference, "housing_market")
    reference_address = _string_array(reference, "address_exact")
    reference_complex = _string_array(reference, "residential_complex")

    for row_number, candidate_index in enumerate(candidate_indices):
        query_row = query.iloc[row_number]
        area = max(float(query_row["total_area_m2"]), 1.0)
        score = (
            distance_km[row_number]
            / float(settings["distance_scale_km"])
        )
        score += (
            np.abs(np.log(reference_area[candidate_index] / area))
            * float(settings["area_penalty"])
        )

        rooms = query_row["rooms"]
        if pd.notna(rooms):
            candidate_rooms = reference_rooms[candidate_index]
            candidate_rooms = np.where(
                np.isnan(candidate_rooms),
                float(rooms),
                candidate_rooms,
            )
            score += (
                np.abs(candidate_rooms - float(rooms))
                * float(settings["room_penalty"])
            )

        house_year = query_row["house_year"]
        if pd.notna(house_year):
            candidate_year = reference_year[candidate_index]
            candidate_year = np.where(
                np.isnan(candidate_year),
                float(house_year),
                candidate_year,
            )
            score += (
                np.abs(candidate_year - float(house_year))
                / 20
                * float(settings["house_age_penalty"])
            )

        score += (
            reference_market[candidate_index]
            != str(query_row["housing_market"])
        ) * float(settings["market_penalty"])

        exact_match = (
            reference_address[candidate_index]
            == str(query_row["address_exact"])
        )
        complex_name = str(query_row["residential_complex"])
        complex_match = (
            reference_complex[candidate_index] == complex_name
        ) & (complex_name != "unknown")
        score -= exact_match * float(settings["exact_address_bonus"])
        score -= complex_match * float(settings["complex_bonus"])

        selected = np.argsort(score)[:neighbor_count]
        selected_score = score[selected]
        selected_price_per_m2 = reference_price_per_m2[
            candidate_index[selected]
        ]
        weights = 1 / np.maximum(
            selected_score - selected_score.min() + 0.3,
            0.3,
        )
        estimated_price_per_m2 = np.exp(
            np.average(
                np.log(selected_price_per_m2),
                weights=weights,
            )
        )
        predictions[row_number] = estimated_price_per_m2 * area
        nearest_distance[row_number] = distance_km[row_number, 0]
        exact_match_count[row_number] = int(exact_match.sum())

    diagnostics = pd.DataFrame(
        {
            "comparable_nearest_distance_km": nearest_distance,
            "comparable_exact_address_candidates": exact_match_count,
        }
    )
    return predictions, diagnostics
