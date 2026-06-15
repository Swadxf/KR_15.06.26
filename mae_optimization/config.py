from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

AVITO_RAW_PATH = ROOT_DIR / "avito_raw.csv"
CIAN_RAW_PATH = ROOT_DIR / "cian_raw.csv"
OSM_FEATURES_PATH = ROOT_DIR / "apartments_ml_osm_1000m_fast.csv"
BASE_PARAMS_PATH = ROOT_DIR / "model_artifacts" / "optuna_best_params.json"

PREPARED_PATH = ARTIFACTS_DIR / "prepared_enhanced.csv"
TRAIN_READY_PATH = ARTIFACTS_DIR / "train_ready.csv"
EXCLUDED_PATH = ARTIFACTS_DIR / "excluded_rows.csv"
SPLIT_PATH = ARTIFACTS_DIR / "split_assignments.csv"
VALIDATION_PREDICTIONS_PATH = ARTIFACTS_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
BLEND_CONFIG_PATH = ARTIFACTS_DIR / "blend_config.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREPARATION_REPORT_PATH = ARTIFACTS_DIR / "preparation_report.json"
FEATURE_CONFIG_PATH = ARTIFACTS_DIR / "feature_config.json"
DATA_DIAGNOSTICS_PATH = ARTIFACTS_DIR / "data_diagnostics.json"

TUNED_TOTAL_PARAMS_PATH = ARTIFACTS_DIR / "tuned_total_params.json"
TUNED_UNIT_PARAMS_PATH = ARTIFACTS_DIR / "tuned_unit_params.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE = 0.16

DEFAULT_SEEDS = [42]
GEO_CELL_STEPS = {
    "geo_cell_2000m": 0.020,
    "geo_cell_1000m": 0.010,
    "geo_cell_500m": 0.005,
    "geo_cell_200m": 0.002,
}

ENHANCED_CATEGORICAL = [
    "address_exact",
    "address_zone",
    "residential_complex",
    "building_key",
    "metro_primary",
    *GEO_CELL_STEPS.keys(),
]


LEAKAGE_COLUMNS = {
    "price_rub",
    "price_log_rub",
    "price_m2_rub",
    "target_unit_log",
}
