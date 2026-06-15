import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    AVITO_RAW_PATH,
    CIAN_RAW_PATH,
    OSM_FEATURES_PATH,
    PREPARATION_REPORT_PATH,
    PREPARED_PATH,
    ROOT_DIR,
)
from .feature_engineering import add_enhanced_features


def _import_existing_feature_builder():
    root = str(ROOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    from ml_features import (
        GEO_RADIUS_M,
        build_ml_dataset,
        merge_osm_features,
    )

    return build_ml_dataset, merge_osm_features, GEO_RADIUS_M


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    avito = pd.read_csv(
        AVITO_RAW_PATH,
        sep=";",
        encoding="utf-8-sig",
        low_memory=False,
    )
    cian = pd.read_csv(
        CIAN_RAW_PATH,
        sep=";",
        encoding="utf-8-sig",
        low_memory=False,
    )

    build_ml_dataset, merge_osm_features, geo_radius = _import_existing_feature_builder()
    raw = pd.concat([avito, cian], ignore_index=True, sort=False)
    prepared = build_ml_dataset(raw)
    prepared = merge_osm_features(prepared, str(OSM_FEATURES_PATH), geo_radius)
    prepared = add_enhanced_features(prepared, avito, cian)
    prepared.to_csv(PREPARED_PATH, index=False, encoding="utf-8-sig")

    report = {
        "avito_rows": int(len(avito)),
        "cian_rows": int(len(cian)),
        "prepared_rows": int(len(prepared)),
        "prepared_columns": int(prepared.shape[1]),
        "osm_features_available": int(prepared["osm_features_available"].sum()),
        "duplicate_source_ids": int(
            prepared.duplicated(["source", "source_listing_id"]).sum()
        ),
        "raw_sha256": {
            AVITO_RAW_PATH.name: file_sha256(AVITO_RAW_PATH),
            CIAN_RAW_PATH.name: file_sha256(CIAN_RAW_PATH),
        },
        "price_m2_is_retained_for_audit_only": True,
        "price_m2_is_never_used_as_a_model_feature": True,
    }
    PREPARATION_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return PREPARED_PATH


def main() -> None:
    path = prepare_dataset()
    print(f"Prepared dataset: {path}")


if __name__ == "__main__":
    main()
