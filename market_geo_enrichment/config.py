from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

BASE_PREPARED_PATH = (
    ROOT_DIR / "mae_optimization" / "artifacts" / "prepared_enhanced.csv"
)
BASE_TEST_PREDICTIONS_PATH = (
    ROOT_DIR / "mae_optimization" / "artifacts" / "test_predictions.csv"
)
POIS_PATH = ROOT_DIR / "apartments_ml_osm_1000m_fast_pois.csv"
AVITO_RAW_PATH = ROOT_DIR / "avito_raw.csv"
CIAN_RAW_PATH = ROOT_DIR / "cian_raw.csv"

ENRICHED_PREPARED_PATH = ARTIFACTS_DIR / "prepared_market_geo.csv"
TRAIN_READY_PATH = ARTIFACTS_DIR / "train_ready_market_geo.csv"
FEATURE_REPORT_PATH = ARTIFACTS_DIR / "feature_report.json"
FEATURE_CONFIG_PATH = ARTIFACTS_DIR / "feature_config.json"
POLICIES_PATH = ARTIFACTS_DIR / "blend_policies.json"
VALIDATION_PREDICTIONS_PATH = ARTIFACTS_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARABLES_REFERENCE_PATH = ARTIFACTS_DIR / "comparables_reference.csv"

EARTH_RADIUS_M = 6_371_000.0
MOSCOW_CENTER = (55.751244, 37.618423)
APPROX_MKAD_RADIUS_KM = 17.2

POI_RADII_M = (250, 500, 750)
LISTING_DENSITY_RADII_M = (250, 500, 1000, 2000, 5000, 10000)

COMPARABLE_PARAMS = {
    "candidate_count": 200,
    "neighbor_count": 3,
    "distance_scale_km": 0.5,
    "area_penalty": 5.0,
    "room_penalty": 1.0,
    "market_penalty": 1.0,
    "house_age_penalty": 0.3,
    "exact_address_bonus": 4.0,
    "complex_bonus": 2.0,
}

FINAL_BLEND_EDGES_RUB = (
    0.0,
    5_000_000.0,
    8_000_000.0,
    12_000_000.0,
    20_000_000.0,
    40_000_000.0,
    float("inf"),
)
FINAL_BLEND_SHRINK_ROWS = 20
