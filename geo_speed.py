

import os
import json
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.neighbors import BallTree


AVITO_PATH = "avito_raw.csv"
CIAN_PATH = "cian_raw.csv"

OUTPUT_PATH = "apartments_ml_osm_1000m_fast.csv"
TILE_CACHE_PATH = "osm_tile_cache_1000m.jsonl"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

RADIUS_M = 1000
EARTH_RADIUS_M = 6_371_000


TILE_SIZE_M = 5000

REQUEST_TIMEOUT = 120
SLEEP_BETWEEN_TILE_REQUESTS = 1.5
MAX_RETRIES = 4


OSM_SELECTORS = {
    "metro": [
        'nwr["railway"="station"]["station"="subway"]',
        'nwr["subway"="yes"]',
        'node["railway"="subway_entrance"]',
    ],
    "park": [
        'nwr["leisure"="park"]',
        'nwr["landuse"="recreation_ground"]',
        'nwr["natural"="wood"]',
    ],
    "school": [
        'nwr["amenity"="school"]',
    ],
    "kindergarten": [
        'nwr["amenity"="kindergarten"]',
    ],
    "hospital_clinic": [
        'nwr["amenity"="hospital"]',
        'nwr["amenity"="clinic"]',
        'nwr["amenity"="doctors"]',
    ],
    "pharmacy": [
        'nwr["amenity"="pharmacy"]',
    ],
    "supermarket": [
        'nwr["shop"="supermarket"]',
        'nwr["shop"="convenience"]',
    ],
    "public_transport_stop": [
        'node["highway"="bus_stop"]',
        'nwr["public_transport"="platform"]',
        'node["railway"="tram_stop"]',
    ],
    "cafe_restaurant": [
        'nwr["amenity"="cafe"]',
        'nwr["amenity"="restaurant"]',
        'nwr["amenity"="fast_food"]',
    ],
    "fitness_sport": [
        'nwr["leisure"="fitness_centre"]',
        'nwr["leisure"="sports_centre"]',
    ],
}


def read_csv_safe(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        on_bad_lines="skip",
        low_memory=False,
    )


def to_float_coord(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace({"", "nan", "None"}, np.nan)
        .pipe(pd.to_numeric, errors="coerce")
    )


def normalize_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    if "source" not in df.columns:
        df["source"] = source_name
    df["source"] = df["source"].fillna(source_name)
    return df


def load_apartments() -> pd.DataFrame:
    avito = normalize_source(read_csv_safe(AVITO_PATH), "avito")
    cian = normalize_source(read_csv_safe(CIAN_PATH), "cian")

    df = pd.concat([avito, cian], ignore_index=True, sort=False)

    df["lat"] = to_float_coord(df["Широта"])
    df["lon"] = to_float_coord(df["Долгота"])


    df = df[
        df["lat"].between(54.0, 57.0)
        & df["lon"].between(35.0, 40.5)
    ].copy()


    numeric_cols = [
        "Цена числом",
        "Цена за м² числом",
        "Общая площадь",
        "Жилая площадь",
        "Площадь кухни",
        "Количество комнат",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
                .replace({"nan": np.nan, "": np.nan, "None": np.nan})
                .pipe(pd.to_numeric, errors="coerce")
            )

    return df.reset_index(drop=True)


def add_tiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mean_lat_rad = math.radians(df["lat"].mean())

    deg_lat = TILE_SIZE_M / 111_320
    deg_lon = TILE_SIZE_M / (111_320 * math.cos(mean_lat_rad))

    df["tile_lat"] = np.floor(df["lat"] / deg_lat).astype(int)
    df["tile_lon"] = np.floor(df["lon"] / deg_lon).astype(int)

    return df


def bbox_for_group(group: pd.DataFrame, buffer_m: int = RADIUS_M) -> Tuple[float, float, float, float]:
    mean_lat_rad = math.radians(group["lat"].mean())

    buffer_lat = buffer_m / 111_320
    buffer_lon = buffer_m / (111_320 * math.cos(mean_lat_rad))

    south = group["lat"].min() - buffer_lat
    north = group["lat"].max() + buffer_lat
    west = group["lon"].min() - buffer_lon
    east = group["lon"].max() + buffer_lon

    return south, west, north, east


def bbox_cache_key(bbox: Tuple[float, float, float, float]) -> str:
    rounded = tuple(round(x, 6) for x in bbox)
    raw = f"{rounded}|v2|radius={RADIUS_M}|tile={TILE_SIZE_M}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox

    parts = []

    for category, selectors in OSM_SELECTORS.items():
        for selector in selectors:
            parts.append(f'{selector}({south},{west},{north},{east});')

    body = "\n  ".join(parts)

    query = f"""
[out:json][timeout:90];
(
  {body}
);
out center tags;
"""
    return query


def request_overpass(query: str) -> Dict[str, Any]:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "real-estate-ml-osm-enrichment/fast-tiled/1.0"
                },
            )

            if response.status_code in {429, 504}:
                time.sleep(10 * attempt)
                continue

            if response.status_code >= 500:
                time.sleep(5 * attempt)
                continue

            response.raise_for_status()
            return response.json()

        except Exception as e:
            last_error = e
            time.sleep(5 * attempt)

    raise RuntimeError(f"Overpass failed after retries: {last_error}")


def get_element_coord(el: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])

    center = el.get("center")
    if center and "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])

    return None


def tags_match_category(tags: Dict[str, str], category: str) -> bool:
    if category == "metro":
        return (
            tags.get("station") == "subway"
            or tags.get("subway") == "yes"
            or tags.get("railway") == "subway_entrance"
        )

    if category == "park":
        return (
            tags.get("leisure") == "park"
            or tags.get("landuse") == "recreation_ground"
            or tags.get("natural") == "wood"
        )

    if category == "school":
        return tags.get("amenity") == "school"

    if category == "kindergarten":
        return tags.get("amenity") == "kindergarten"

    if category == "hospital_clinic":
        return tags.get("amenity") in {"hospital", "clinic", "doctors"}

    if category == "pharmacy":
        return tags.get("amenity") == "pharmacy"

    if category == "supermarket":
        return tags.get("shop") in {"supermarket", "convenience"}

    if category == "public_transport_stop":
        return (
            tags.get("highway") == "bus_stop"
            or tags.get("public_transport") == "platform"
            or tags.get("railway") == "tram_stop"
        )

    if category == "cafe_restaurant":
        return tags.get("amenity") in {"cafe", "restaurant", "fast_food"}

    if category == "fitness_sport":
        return tags.get("leisure") in {"fitness_centre", "sports_centre"}

    return False


def parse_pois(overpass_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []

    for el in overpass_json.get("elements", []):
        coord = get_element_coord(el)
        if coord is None:
            continue

        lat, lon = coord
        tags = el.get("tags", {})
        osm_uid = f'{el.get("type")}:{el.get("id")}'

        for category in OSM_SELECTORS.keys():
            if tags_match_category(tags, category):
                rows.append({
                    "category": category,
                    "osm_uid": osm_uid,
                    "osm_type": el.get("type"),
                    "osm_id": el.get("id"),
                    "lat": lat,
                    "lon": lon,
                    "name": tags.get("name"),
                })

    return rows


def load_tile_cache(path: str) -> Dict[str, List[Dict[str, Any]]]:
    cache = {}

    if not os.path.exists(path):
        return cache

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cache[obj["key"]] = obj["pois"]

    return cache


def append_tile_cache(path: str, key: str, pois: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"key": key, "pois": pois},
                ensure_ascii=False,
            )
            + "\n"
        )


def collect_pois_by_tiles(df: pd.DataFrame) -> pd.DataFrame:
    cache = load_tile_cache(TILE_CACHE_PATH)

    all_pois = []

    grouped = list(df.groupby(["tile_lat", "tile_lon"]))
    print(f"Тайлов для запроса: {len(grouped)}")
    print(f"Тайлов уже в кэше: {len(cache)}")

    for _, group in tqdm(grouped, desc="Downloading OSM tiles"):
        bbox = bbox_for_group(group)
        key = bbox_cache_key(bbox)

        if key in cache:
            pois = cache[key]
        else:
            query = build_overpass_query(bbox)
            overpass_json = request_overpass(query)
            pois = parse_pois(overpass_json)

            append_tile_cache(TILE_CACHE_PATH, key, pois)
            time.sleep(SLEEP_BETWEEN_TILE_REQUESTS)

        all_pois.extend(pois)

    if not all_pois:
        return pd.DataFrame(columns=["category", "osm_uid", "lat", "lon", "name"])

    pois_df = pd.DataFrame(all_pois)


    pois_df = pois_df.drop_duplicates(["category", "osm_uid"]).reset_index(drop=True)

    return pois_df


def add_balltree_features(apartments: pd.DataFrame, pois: pd.DataFrame) -> pd.DataFrame:
    df = apartments.copy()

    apartment_coords_rad = np.radians(df[["lat", "lon"]].to_numpy())
    radius_rad = RADIUS_M / EARTH_RADIUS_M

    for category in OSM_SELECTORS.keys():
        cat_pois = pois[pois["category"] == category].copy()

        count_col = f"osm_{category}_count_500m"
        nearest_col = f"osm_{category}_nearest_m"
        has_col = f"has_{category}_500m"

        if cat_pois.empty:
            df[count_col] = 0
            df[nearest_col] = np.nan
            df[has_col] = 0
            continue

        poi_coords_rad = np.radians(cat_pois[["lat", "lon"]].to_numpy())

        tree = BallTree(poi_coords_rad, metric="haversine")

        indices = tree.query_radius(
            apartment_coords_rad,
            r=radius_rad,
            return_distance=False,
        )

        counts = np.array([len(x) for x in indices], dtype=int)

        distances_rad, _ = tree.query(apartment_coords_rad, k=1)
        nearest_m = distances_rad[:, 0] * EARTH_RADIUS_M

        nearest_m = np.where(nearest_m <= RADIUS_M, nearest_m, np.nan)

        df[count_col] = counts
        df[nearest_col] = np.round(nearest_m, 1)
        df[has_col] = (counts > 0).astype(int)


    if not pois.empty:
        unique_pois = pois.drop_duplicates("osm_uid").copy()
        poi_coords_rad = np.radians(unique_pois[["lat", "lon"]].to_numpy())

        tree = BallTree(poi_coords_rad, metric="haversine")
        indices = tree.query_radius(
            apartment_coords_rad,
            r=radius_rad,
            return_distance=False,
        )

        df["osm_total_poi_count_500m"] = [len(x) for x in indices]
    else:
        df["osm_total_poi_count_500m"] = 0

    return df


def add_ml_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["infrastructure_score_500m"] = (
        df["osm_metro_count_500m"].fillna(0) * 4
        + df["osm_park_count_500m"].fillna(0) * 2
        + df["osm_school_count_500m"].fillna(0) * 1
        + df["osm_kindergarten_count_500m"].fillna(0) * 1
        + df["osm_supermarket_count_500m"].fillna(0) * 1
        + df["osm_public_transport_stop_count_500m"].fillna(0) * 1
        + df["osm_pharmacy_count_500m"].fillna(0) * 0.5
        + df["osm_hospital_clinic_count_500m"].fillna(0) * 1
    )


    for col in df.columns:
        if col.startswith("osm_") and col.endswith("_count_500m"):
            df[f"log1p_{col}"] = np.log1p(df[col].fillna(0))

    return df


def main():
    apartments = load_apartments()
    apartments = add_tiles(apartments)

    print(f"Квартир с валидными координатами: {len(apartments)}")

    pois = collect_pois_by_tiles(apartments)

    print(f"Уникальных OSM POI найдено: {len(pois)}")
    if not pois.empty:
        print(pois["category"].value_counts())

    enriched = add_balltree_features(apartments, pois)
    enriched = add_ml_geo_features(enriched)

    enriched.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    pois_output = OUTPUT_PATH.replace(".csv", "_pois.csv")
    pois.to_csv(pois_output, index=False, encoding="utf-8-sig")

    print(f"Готово: {OUTPUT_PATH}")
    print(f"POI сохранены отдельно: {pois_output}")
    print(f"Размер итоговой таблицы: {enriched.shape}")


if __name__ == "__main__":
    main()