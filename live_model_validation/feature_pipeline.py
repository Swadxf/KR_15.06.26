import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .config import (
    APPROX_MKAD_RADIUS_KM,
    EARTH_RADIUS_M,
    LISTING_DENSITY_RADII_M,
    MOSCOW_CENTER,
    OSM_CATEGORIES,
    OSM_NOT_FOUND_DISTANCE_M,
    OSM_RADIUS_M,
    POI_RADII_M,
)
from .enhanced_features import add_enhanced_features
from .feature_builder import build_ml_dataset, read_csv_safe

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


AVITO_EXTRA_COLUMNS = (
    "source_listing_id",
    "Полный адрес",
    "Метро",
    "О жилом комплексе: Название новостройки",
    "О жилом комплексе: Корпус, строение",
)
CIAN_EXTRA_COLUMNS = (
    "source_listing_id",
    "Полный адрес",
    "Метро",
    "Дом",
)


def read_raw(path: Path, source: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_csv_safe(str(path), source)


def _ensure_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    source: str,
) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = np.nan
    if "source" not in output.columns:
        output["source"] = source
    output["source"] = output["source"].fillna(source)
    return output


def _valid_coordinate_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["lat"].between(54.0, 57.0)
        & frame["lon"].between(35.0, 41.5)
    )


def _write_for_valid(
    frame: pd.DataFrame,
    valid: pd.Series,
    column: str,
    values: np.ndarray,
    fill_value: float = np.nan,
) -> None:
    output = np.full(len(frame), fill_value, dtype=float)
    output[valid.to_numpy()] = values
    frame[column] = output


def add_osm_1000m_features(
    frame: pd.DataFrame,
    pois: pd.DataFrame,
) -> None:
    valid = _valid_coordinate_mask(frame)
    coordinates = np.radians(frame.loc[valid, ["lat", "lon"]].to_numpy())
    pois = pois.drop_duplicates(["category", "osm_uid"])

    for category in OSM_CATEGORIES:
        group = pois.loc[pois["category"].eq(category)]
        count_column = f"osm_{category}_count_1000m"
        nearest_column = f"osm_{category}_nearest_1000m"
        has_column = f"has_{category}_1000m"
        if group.empty or not len(coordinates):
            frame[count_column] = 0
            frame[nearest_column] = OSM_NOT_FOUND_DISTANCE_M
            frame[has_column] = 0
            continue

        tree = BallTree(
            np.radians(group[["lat", "lon"]].to_numpy()),
            metric="haversine",
        )
        counts = tree.query_radius(
            coordinates,
            r=OSM_RADIUS_M / EARTH_RADIUS_M,
            count_only=True,
        )
        distance_radians, _ = tree.query(coordinates, k=1)
        nearest = distance_radians[:, 0] * EARTH_RADIUS_M
        nearest = np.where(
            nearest <= OSM_RADIUS_M,
            nearest,
            OSM_NOT_FOUND_DISTANCE_M,
        )
        _write_for_valid(frame, valid, count_column, counts, fill_value=0)
        _write_for_valid(
            frame,
            valid,
            nearest_column,
            nearest,
            fill_value=OSM_NOT_FOUND_DISTANCE_M,
        )
        frame[has_column] = frame[count_column].gt(0).astype(int)

    unique_pois = pois.drop_duplicates("osm_uid")
    if len(coordinates) and not unique_pois.empty:
        tree = BallTree(
            np.radians(unique_pois[["lat", "lon"]].to_numpy()),
            metric="haversine",
        )
        total_counts = tree.query_radius(
            coordinates,
            r=OSM_RADIUS_M / EARTH_RADIUS_M,
            count_only=True,
        )
        _write_for_valid(
            frame,
            valid,
            "osm_total_poi_count_1000m",
            total_counts,
            fill_value=0,
        )
    else:
        frame["osm_total_poi_count_1000m"] = 0

    frame["infrastructure_score_1000m"] = (
        frame["osm_metro_count_1000m"] * 4
        + frame["osm_park_count_1000m"] * 2
        + frame["osm_school_count_1000m"]
        + frame["osm_kindergarten_count_1000m"]
        + frame["osm_supermarket_count_1000m"]
        + frame["osm_public_transport_stop_count_1000m"]
        + frame["osm_pharmacy_count_1000m"] * 0.5
        + frame["osm_hospital_clinic_count_1000m"]
    )
    for column in [
        column
        for column in frame.columns
        if column.startswith("osm_") and column.endswith("_count_1000m")
    ]:
        frame[f"log1p_{column}"] = np.log1p(frame[column].fillna(0))
    frame["log1p_osm_total_poi_count_1000m"] = np.log1p(
        frame["osm_total_poi_count_1000m"].fillna(0)
    )
    frame["osm_features_available"] = valid.astype(int)
    frame["osm_radius_m"] = np.where(valid, OSM_RADIUS_M, np.nan)


def add_multiscale_poi_features(
    frame: pd.DataFrame,
    pois: pd.DataFrame,
) -> None:
    valid = _valid_coordinate_mask(frame)
    coordinates = np.radians(frame.loc[valid, ["lat", "lon"]].to_numpy())
    pois = pois.drop_duplicates(["category", "osm_uid"])

    added_columns = []
    for category, group in pois.groupby("category", sort=True):
        tree = BallTree(
            np.radians(group[["lat", "lon"]].to_numpy()),
            metric="haversine",
        )
        for radius_m in POI_RADII_M:
            counts = tree.query_radius(
                coordinates,
                r=radius_m / EARTH_RADIUS_M,
                count_only=True,
            )
            column = f"ext_{category}_count_{radius_m}m"
            _write_for_valid(frame, valid, column, counts, fill_value=0)
            added_columns.append(column)

    for radius_m in POI_RADII_M:
        columns = [
            column
            for column in added_columns
            if column.endswith(f"_count_{radius_m}m")
        ]
        frame[f"ext_total_poi_count_{radius_m}m"] = frame[columns].sum(
            axis=1
        )


def add_listing_density_features(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> None:
    valid = _valid_coordinate_mask(frame)
    query_coordinates = np.radians(
        frame.loc[valid, ["lat", "lon"]].to_numpy()
    )
    reference = reference.dropna(subset=["lat", "lon"])
    reference_coordinates = np.radians(
        reference[["lat", "lon"]].to_numpy()
    )
    tree = BallTree(reference_coordinates, metric="haversine")

    for radius_m in LISTING_DENSITY_RADII_M:
        counts = tree.query_radius(
            query_coordinates,
            r=radius_m / EARTH_RADIUS_M,
            count_only=True,
        )
        _write_for_valid(
            frame,
            valid,
            f"listing_density_{radius_m}m",
            counts,
        )

    nearest_radians, _ = tree.query(query_coordinates, k=1)
    _write_for_valid(
        frame,
        valid,
        "nearest_listing_distance_m",
        nearest_radians[:, 0] * EARTH_RADIUS_M,
    )


def add_moscow_position_features(frame: pd.DataFrame) -> None:
    valid = _valid_coordinate_mask(frame)
    coordinates = np.radians(frame.loc[valid, ["lat", "lon"]].to_numpy())
    center = np.radians(np.array([MOSCOW_CENTER]))
    tree = BallTree(center, metric="haversine")
    distance_radians, _ = tree.query(coordinates, k=1)
    radial_km = distance_radians[:, 0] * EARTH_RADIUS_M / 1000

    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]
    center_latitude, center_longitude = center[0]
    longitude_delta = longitude - center_longitude
    bearing = np.arctan2(
        np.sin(longitude_delta) * np.cos(latitude),
        np.cos(center_latitude) * np.sin(latitude)
        - np.sin(center_latitude)
        * np.cos(latitude)
        * np.cos(longitude_delta),
    )
    values = {
        "distance_to_moscow_center_km": radial_km,
        "approx_mkad_distance_km": np.abs(
            radial_km - APPROX_MKAD_RADIUS_KM
        ),
        "inside_approx_mkad": (
            radial_km < APPROX_MKAD_RADIUS_KM
        ).astype(int),
        "center_bearing_sin": np.sin(bearing),
        "center_bearing_cos": np.cos(bearing),
    }
    for column, column_values in values.items():
        _write_for_valid(frame, valid, column, column_values)


def prepare_fresh_features(
    avito_raw: pd.DataFrame,
    cian_raw: pd.DataFrame,
    pois: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    avito = _ensure_columns(avito_raw, AVITO_EXTRA_COLUMNS, "avito")
    cian = _ensure_columns(cian_raw, CIAN_EXTRA_COLUMNS, "cian")
    raw = pd.concat([avito, cian], ignore_index=True, sort=False)
    if raw.empty:
        return pd.DataFrame()

    prepared = build_ml_dataset(raw)
    add_osm_1000m_features(prepared, pois)
    prepared = add_enhanced_features(prepared, avito, cian)
    add_multiscale_poi_features(prepared, pois)
    add_listing_density_features(prepared, reference)
    add_moscow_position_features(prepared)

    rooms = pd.to_numeric(prepared["rooms"], errors="coerce").fillna(1)
    rooms = rooms.clip(lower=1)
    prepared["area_per_room_m2"] = (
        pd.to_numeric(prepared["total_area_m2"], errors="coerce") / rooms
    )
    return prepared


def make_model_matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
    categorical_features: set[str],
) -> pd.DataFrame:
    columns = {}
    for column in feature_names:
        if column in frame.columns:
            columns[column] = frame[column]
        else:
            columns[column] = pd.Series(
                "unknown" if column in categorical_features else np.nan,
                index=frame.index,
            )
    output = pd.DataFrame(columns, index=frame.index)

    for column in feature_names:
        if column in categorical_features:
            output[column] = output[column].fillna("unknown").astype(str)
        else:
            output[column] = pd.to_numeric(
                output[column],
                errors="coerce",
            )
    return output.replace([np.inf, -np.inf], np.nan)
