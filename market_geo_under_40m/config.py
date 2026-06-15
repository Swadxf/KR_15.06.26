from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

MAX_PRICE_RUB = 40_000_000.0

TRAIN_READY_PATH = ARTIFACTS_DIR / "train_ready_under_40m.csv"
REMOVED_ROWS_PATH = ARTIFACTS_DIR / "removed_over_40m.csv"
FEATURE_CONFIG_PATH = ARTIFACTS_DIR / "feature_config.json"
SPLIT_REPORT_PATH = ARTIFACTS_DIR / "split_report.json"
POLICIES_PATH = ARTIFACTS_DIR / "blend_policies.json"
VALIDATION_PREDICTIONS_PATH = ARTIFACTS_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARABLES_REFERENCE_PATH = ARTIFACTS_DIR / "comparables_reference.csv"

FULL_MARKET_METRICS_PATH = (
    ROOT_DIR / "market_geo_enrichment" / "artifacts" / "metrics.json"
)
FULL_MARKET_TEST_PREDICTIONS_PATH = (
    ROOT_DIR
    / "market_geo_enrichment"
    / "artifacts"
    / "test_predictions.csv"
)
