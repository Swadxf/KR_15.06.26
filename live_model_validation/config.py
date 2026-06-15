from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
BUNDLE_DIR = PROJECT_DIR / "model_bundle"
RUNS_DIR = PROJECT_DIR / "runs"

AVITO_PARSER_PATH = ROOT_DIR / "avito_parser.py"
CIAN_PARSER_PATH = ROOT_DIR / "cian_parser.py"
AVITO_PROFILE_DIR = ROOT_DIR / "avito_profile"
CIAN_PROFILE_DIR = ROOT_DIR / "cian_profile"

TOTAL_MODEL_PATH = BUNDLE_DIR / "total_price_seed_42.cbm"
UNIT_MODEL_PATH = BUNDLE_DIR / "unit_price_seed_42.cbm"
FEATURE_CONFIG_PATH = BUNDLE_DIR / "feature_config.json"
BLEND_POLICIES_PATH = BUNDLE_DIR / "blend_policies.json"
COMPARABLES_REFERENCE_PATH = BUNDLE_DIR / "comparables_reference.csv"
POIS_PATH = BUNDLE_DIR / "osm_pois.csv"
MANIFEST_PATH = BUNDLE_DIR / "manifest.json"

EARTH_RADIUS_M = 6_371_000.0
OSM_RADIUS_M = 1000
OSM_NOT_FOUND_DISTANCE_M = 1001.0
MAX_TRAINING_PRICE_RUB = 40_000_000.0
MOSCOW_CENTER = (55.751244, 37.618423)
APPROX_MKAD_RADIUS_KM = 17.2
POI_RADII_M = (250, 500, 750)
LISTING_DENSITY_RADII_M = (250, 500, 1000, 2000, 5000, 10000)
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

OSM_CATEGORIES = (
    "metro",
    "park",
    "school",
    "kindergarten",
    "hospital_clinic",
    "pharmacy",
    "supermarket",
    "public_transport_stop",
    "cafe_restaurant",
    "fitness_sport",
)
