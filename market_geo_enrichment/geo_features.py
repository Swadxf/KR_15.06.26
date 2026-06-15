import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .config import (
    APPROX_MKAD_RADIUS_KM,
    ARTIFACTS_DIR,
    AVITO_RAW_PATH,
    BASE_PREPARED_PATH,
    CIAN_RAW_PATH,
    EARTH_RADIUS_M,
    ENRICHED_PREPARED_PATH,
    FEATURE_REPORT_PATH,
    LISTING_DENSITY_RADII_M,
    MOSCOW_CENTER,
    POI_RADII_M,
    POIS_PATH,
)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_feature(
    frame: pd.DataFrame,
    valid_coordinates: pd.Series,
    name: str,
    values: np.ndarray,
) -> None:
    output = np.full(len(frame), np.nan)
    output[valid_coordinates.to_numpy()] = values
    frame[name] = output


def add_listing_density_features(frame: pd.DataFrame) -> list[str]:
    valid = frame[["lat", "lon"]].notna().all(axis=1)
    coordinates = np.radians(frame.loc[valid, ["lat", "lon"]].to_numpy())
    tree = BallTree(coordinates, metric="haversine")
    columns = []

    for radius_m in LISTING_DENSITY_RADII_M:
        counts = tree.query_radius(
            coordinates,
            r=radius_m / EARTH_RADIUS_M,
            count_only=True,
        ) - 1
        column = f"listing_density_{radius_m}m"
        _write_feature(frame, valid, column, counts)
        columns.append(column)

    distances, _ = tree.query(coordinates, k=2)
    nearest = distances[:, 1] * EARTH_RADIUS_M
    _write_feature(
        frame,
        valid,
        "nearest_listing_distance_m",
        nearest,
    )
    columns.append("nearest_listing_distance_m")
    return columns


def add_multiscale_poi_features(
    frame: pd.DataFrame,
    pois: pd.DataFrame,
) -> list[str]:
    valid = frame[["lat", "lon"]].notna().all(axis=1)
    coordinates = np.radians(frame.loc[valid, ["lat", "lon"]].to_numpy())
    columns = []

    pois = pois.drop_duplicates(["category", "osm_uid"])
    for category, group in pois.groupby("category", sort=True):
        poi_coordinates = np.radians(group[["lat", "lon"]].to_numpy())
        tree = BallTree(poi_coordinates, metric="haversine")
        for radius_m in POI_RADII_M:
            counts = tree.query_radius(
                coordinates,
                r=radius_m / EARTH_RADIUS_M,
                count_only=True,
            )
            column = f"ext_{category}_count_{radius_m}m"
            _write_feature(frame, valid, column, counts)
            columns.append(column)

    for radius_m in POI_RADII_M:
        radius_columns = [
            column
            for column in columns
            if column.endswith(f"_count_{radius_m}m")
        ]
        total_column = f"ext_total_poi_count_{radius_m}m"
        frame[total_column] = frame[radius_columns].sum(axis=1)
        columns.append(total_column)

    return columns


def add_moscow_position_features(frame: pd.DataFrame) -> list[str]:
    valid = frame[["lat", "lon"]].notna().all(axis=1)
    coordinates = np.radians(frame.loc[valid, ["lat", "lon"]].to_numpy())
    center = np.radians(np.array([MOSCOW_CENTER]))
    tree = BallTree(center, metric="haversine")
    distance_radians, _ = tree.query(coordinates, k=1)
    radial_distance_km = distance_radians[:, 0] * EARTH_RADIUS_M / 1000

    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]
    center_latitude, center_longitude = center[0]
    longitude_delta = longitude - center_longitude
    bearing_y = np.sin(longitude_delta) * np.cos(latitude)
    bearing_x = (
        np.cos(center_latitude) * np.sin(latitude)
        - np.sin(center_latitude)
        * np.cos(latitude)
        * np.cos(longitude_delta)
    )
    bearing = np.arctan2(bearing_y, bearing_x)

    values = {
        "distance_to_moscow_center_km": radial_distance_km,
        "approx_mkad_distance_km": np.abs(
            radial_distance_km - APPROX_MKAD_RADIUS_KM
        ),
        "inside_approx_mkad": (
            radial_distance_km < APPROX_MKAD_RADIUS_KM
        ).astype(int),
        "center_bearing_sin": np.sin(bearing),
        "center_bearing_cos": np.cos(bearing),
    }
    for column, column_values in values.items():
        _write_feature(frame, valid, column, column_values)
    return list(values)


def build_geo_dataset() -> pd.DataFrame:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_PREPARED_PATH.exists():
        from mae_optimization.prepare_data import prepare_dataset

        prepare_dataset()
    frame = pd.read_csv(
        BASE_PREPARED_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )
    pois = pd.read_csv(
        POIS_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    added_columns = []
    added_columns.extend(add_listing_density_features(frame))
    added_columns.extend(add_multiscale_poi_features(frame, pois))
    added_columns.extend(add_moscow_position_features(frame))
    frame.to_csv(
        ENRICHED_PREPARED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "rows": int(len(frame)),
        "source_poi_rows": int(len(pois)),
        "unique_poi_rows": int(
            len(pois.drop_duplicates(["category", "osm_uid"]))
        ),
        "coordinate_coverage_rows": int(
            frame[["lat", "lon"]].notna().all(axis=1).sum()
        ),
        "added_feature_count": int(len(added_columns)),
        "added_features": added_columns,
        "poi_radii_m": list(POI_RADII_M),
        "listing_density_radii_m": list(LISTING_DENSITY_RADII_M),
        "raw_sha256": {
            AVITO_RAW_PATH.name: file_sha256(AVITO_RAW_PATH),
            CIAN_RAW_PATH.name: file_sha256(CIAN_RAW_PATH),
        },
        "mkad_note": (
            "Approximate radial distance; useful as a model feature, not a "
            "cartographic measurement."
        ),
    }
    FEATURE_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frame


def main() -> None:
    frame = build_geo_dataset()
    print(
        f"Saved {len(frame)} rows with additional geo features to "
        f"{ENRICHED_PREPARED_PATH}"
    )


if __name__ == "__main__":
    main()
