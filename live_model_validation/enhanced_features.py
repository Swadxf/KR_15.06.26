import re
from typing import Any

import numpy as np
import pandas as pd

from .config import ENHANCED_CATEGORICAL, GEO_CELL_STEPS


def normalize_category(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).lower().replace("ё", "е").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    return text or "unknown"


def normalize_address_zone(value: Any) -> str:
    parts = [
        normalize_category(part)
        for part in str(value).split(",")
        if normalize_category(part) != "unknown"
    ]
    without_house_numbers = [part for part in parts if not re.search(r"\d", part)]
    selected = (without_house_numbers or parts[:1])[:3]
    return ", ".join(selected) or "unknown"


def normalize_metro(value: Any) -> str:
    text = normalize_category(value).split(";")[0]
    text = re.sub(r"\s+", " ", text).strip(" ,-")
    return text[:120] or "unknown"


def _raw_extra_frame(avito: pd.DataFrame, cian: pd.DataFrame) -> pd.DataFrame:
    avito_extra = pd.DataFrame(
        {
            "source": "avito",
            "source_listing_id": avito["source_listing_id"].astype(str),
            "raw_address": avito["Полный адрес"],
            "raw_metro": avito["Метро"],
            "raw_complex": avito["О жилом комплексе: Название новостройки"],
            "raw_building": avito["О жилом комплексе: Корпус, строение"],
        }
    )
    cian_extra = pd.DataFrame(
        {
            "source": "cian",
            "source_listing_id": cian["source_listing_id"].astype(str),
            "raw_address": cian["Полный адрес"],
            "raw_metro": cian["Метро"],
            "raw_complex": "",
            "raw_building": cian["Дом"],
        }
    )
    return pd.concat([avito_extra, cian_extra], ignore_index=True)


def add_enhanced_features(
    base: pd.DataFrame,
    avito: pd.DataFrame,
    cian: pd.DataFrame,
) -> pd.DataFrame:
    result = base.copy()
    result["source_listing_id"] = result["source_listing_id"].astype(str)

    extras = _raw_extra_frame(avito, cian)
    result = result.merge(
        extras,
        on=["source", "source_listing_id"],
        how="left",
        validate="one_to_one",
    )

    result["address_exact"] = result["raw_address"].map(normalize_category)
    result["address_zone"] = result["raw_address"].map(normalize_address_zone)
    result["residential_complex"] = result["raw_complex"].map(normalize_category)
    result["building_key"] = result["raw_building"].map(normalize_category)
    result["metro_primary"] = result["raw_metro"].map(normalize_metro)

    lat = pd.to_numeric(result["lat"], errors="coerce")
    lon = pd.to_numeric(result["lon"], errors="coerce")
    for column, step in GEO_CELL_STEPS.items():
        lat_cell = np.floor(lat / step).astype("Int64").astype(str)
        lon_cell = np.floor(lon / step).astype("Int64").astype(str)
        result[column] = lat_cell + "_" + lon_cell

    result = result.drop(
        columns=["raw_address", "raw_metro", "raw_complex", "raw_building"],
        errors="ignore",
    )

    for column in ENHANCED_CATEGORICAL:
        result[column] = result[column].fillna("unknown").astype(str)

    return result

