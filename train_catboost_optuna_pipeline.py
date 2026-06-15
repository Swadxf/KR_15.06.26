# категориальные признаки НЕ кодируются числами, а остаются строками для CatBoost;
# бинарные/multi-hot признаки приводятся к 0/1;
# price_m2_rub удаляется как leakage при предсказании price_rub;
# служебные поля url/id/address/parsed_at удаляются из признаков;
# при target-mode=log модель обучается на log1p(price_rub), а метрики считаются в рублях.


from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from catboost import CatBoostRegressor
except ImportError as e:
    raise ImportError(
        "Не установлен catboost. Установи: pip install catboost"
    ) from e

try:
    import optuna
except ImportError:
    optuna = None


DEFAULT_INPUT_PATH = "apartments_ml.csv"
DEFAULT_OUTPUT_DIR = "model_artifacts"

TARGET_COL = "price_rub"
ALT_LOG_TARGET_COL = "price_log_rub"


LEAKAGE_COLUMNS = [
    "price_m2_rub",
]


SERVICE_COLUMNS = [
    "url",
    "source_listing_id",
    "parsed_at",
    "link_collected_at",
    "address",
    "repair_source",
    "finish_source",
    "house_type_source",
    "ceiling_height_source",
    "osm_radius_m",
]


USE_STREET_FEATURE = False


MIN_PRICE_RUB = 1_000_000
MAX_PRICE_RUB = 250_000_000
MIN_TOTAL_AREA_M2 = 10
MAX_TOTAL_AREA_M2 = 400
DROP_ROWS_WITHOUT_COORDS = True


OSM_RADIUS_M = 1000
OSM_NOT_FOUND_DISTANCE_M = OSM_RADIUS_M + 1


REFERENCE_YEAR = 2026


BASE_CATEGORICAL_COLUMNS = [
    "source",
    "region",
    "okrug",
    "settlement",
    "street",
    "property_format",
    "house_type",
    "housing_market",
    "housing_class",
    "heating_type",
    "overlap_type",
    "gas_supply_type",
    "redevelopment_status",
    "repair_type",
    "finish_type",
    "sale_type",
    "bathroom_type",
    "balcony_loggia_type",
    "floor_group",
    "furniture_set",
    "appliances_set",
]


BINARY_COLUMNS = [
    "is_studio",
    "is_first_floor",
    "is_last_floor",
    "is_new_building_year",
    "is_share_sale",
    "is_apartment_format",
    "is_penthouse_format",
    "has_elevator",
    "has_balcony",
    "has_loggia",
    "windows_yard",
    "windows_street",
    "windows_park_or_forest",
    "windows_water",
    "windows_sunny",
    "windows_panoramic",
    "windows_two_sides",
    "windows_quiet",
    "parking_underground",
    "parking_surface",
    "parking_multilevel",
    "parking_yard",
    "parking_open_yard",
    "parking_barrier",
    "parking_guest",
    "yard_closed",
    "yard_playground",
    "yard_sportground",
    "yard_no_cars",
    "yard_barrier",
    "has_furniture",
    "has_kitchen_furniture",
    "has_wardrobe_storage",
    "has_sleeping_places",
    "has_fridge",
    "has_washer",
    "has_dishwasher",
    "has_ac",
    "has_water_heater",
    "has_oven",
    "has_cooktop",
    "has_hood",
    "has_appliances",
    "mortgage_possible",
    "one_owner",
    "no_encumbrance",
    "quick_deal",
    "has_warm_floor",
    "is_emergency_house",
    "has_ramp",
    "has_minor_owners",
    "maternity_capital_used",
    "has_garbage_chute",
    "has_concierge",
    "entrance_clean",
    "entrance_needs_repair",
    "has_security",
    "has_storage_room",
    "has_mall_nearby_text",
    "has_park_nearby_text",
    "has_school_nearby_text",
    "has_kindergarten_nearby_text",
    "has_metro_1000m",
    "has_park_1000m",
    "has_school_1000m",
    "has_kindergarten_1000m",
    "has_hospital_clinic_1000m",
    "has_pharmacy_1000m",
    "has_supermarket_1000m",
    "has_public_transport_stop_1000m",
    "has_cafe_restaurant_1000m",
    "has_fitness_sport_1000m",
    "osm_features_available",
]

ALWAYS_DROP_COLUMNS = set(SERVICE_COLUMNS + LEAKAGE_COLUMNS)


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def normalize_missing_string(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    s_low = s.lower()
    if s_low in {"nan", "none", "null", "нет", "не указано", "неизвестно", "-"}:
        return "unknown"
    return s


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan}),
        errors="coerce",
    )


def to_binary(series: pd.Series) -> pd.Series:
    true_values = {"1", "1.0", "true", "yes", "y", "да", "есть", "имеется", "present"}
    false_values = {"0", "0.0", "false", "no", "n", "нет", "отсутствует", "none", "nan", "", "unknown"}

    def convert(x: Any) -> int:
        if pd.isna(x):
            return 0
        if isinstance(x, (bool, np.bool_)):
            return int(x)
        if isinstance(x, (int, float, np.integer, np.floating)):
            if pd.isna(x):
                return 0
            return 1 if float(x) > 0 else 0
        x_str = str(x).strip().lower()
        if x_str in true_values:
            return 1
        if x_str in false_values:
            return 0

        return 1

    return series.map(convert).astype("int8")


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    result = num / den.replace({0: np.nan})
    return result.replace([np.inf, -np.inf], np.nan)


def existing(columns: List[str], df: pd.DataFrame) -> List[str]:
    return [c for c in columns if c in df.columns]


def build_exclusion_mask(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    reasons = pd.Series("", index=df.index, dtype="object")

    def add_reason(mask: pd.Series, reason: str) -> None:
        nonlocal reasons
        reasons.loc[mask] = reasons.loc[mask].apply(
            lambda old: reason if not old else f"{old}; {reason}"
        )

    if TARGET_COL not in df.columns:
        raise ValueError(f"В данных нет целевой колонки {TARGET_COL!r}")

    price = to_numeric(df[TARGET_COL])
    add_reason(price.isna(), "missing_target_price")
    add_reason(price < MIN_PRICE_RUB, "price_too_low")
    add_reason(price > MAX_PRICE_RUB, "price_too_high")

    if "total_area_m2" in df.columns:
        area = to_numeric(df["total_area_m2"])
        add_reason(area.isna(), "missing_total_area")
        add_reason(area < MIN_TOTAL_AREA_M2, "total_area_too_low")
        add_reason(area > MAX_TOTAL_AREA_M2, "total_area_too_high")

    if "rooms" in df.columns:
        rooms = to_numeric(df["rooms"])
        add_reason(rooms.isna(), "missing_rooms")
        add_reason(rooms < 0, "rooms_negative")
        add_reason(rooms > 10, "rooms_too_high")

    if DROP_ROWS_WITHOUT_COORDS and {"lat", "lon"}.issubset(df.columns):
        lat = to_numeric(df["lat"])
        lon = to_numeric(df["lon"])
        add_reason(lat.isna() | lon.isna(), "missing_coordinates")
        add_reason(~lat.between(54.0, 57.0) | ~lon.between(35.0, 40.5), "coordinates_out_of_bbox")

    excluded_mask = reasons.ne("")
    excluded = df.loc[excluded_mask].copy()
    if not excluded.empty:
        excluded.insert(0, "exclude_reason", reasons.loc[excluded_mask])

    return excluded_mask, excluded


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    original_shape = df.shape
    df = df.drop_duplicates().copy()

    duplicated_rows_removed = 0
    if "url" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["url"], keep="first").copy()
        duplicated_rows_removed = before - len(df)

    df[TARGET_COL] = to_numeric(df[TARGET_COL])

    excluded_mask, excluded_rows = build_exclusion_mask(df)
    df = df.loc[~excluded_mask].copy().reset_index(drop=True)

    df[ALT_LOG_TARGET_COL] = np.log1p(df[TARGET_COL])

    drop_cols = [c for c in ALWAYS_DROP_COLUMNS if c in df.columns]
    if not USE_STREET_FEATURE and "street" in df.columns:
        drop_cols.append("street")

    df = df.drop(columns=drop_cols, errors="ignore")

    numeric_base_cols = [
        "lat", "lon", "rooms", "total_area_m2", "kitchen_area_m2", "living_area_m2",
        "floor", "floors_total", "house_year", "ceiling_height_m", "balcony_count",
        "loggia_count", "bathroom_combined_count", "bathroom_separate_count",
        "bathroom_total_count", "passenger_elevator_count", "freight_elevator_count",
        "elevator_total_count", "metro_min_listed", "metro_count_listed", "mkad_distance_km",
        "ownership_share_fraction", "housing_class_rank", "finish_rank",
        "description_len_chars", "description_len_words", "windows_view_count",
    ]

    for col in existing(numeric_base_cols, df):
        df[col] = to_numeric(df[col])

    if {"kitchen_area_m2", "total_area_m2"}.issubset(df.columns):
        df["kitchen_area_share"] = safe_divide(df["kitchen_area_m2"], df["total_area_m2"])
        df.loc[~df["kitchen_area_share"].between(0, 0.8), "kitchen_area_share"] = np.nan

    if {"living_area_m2", "total_area_m2"}.issubset(df.columns):
        df["living_area_share"] = safe_divide(df["living_area_m2"], df["total_area_m2"])
        df.loc[~df["living_area_share"].between(0, 1.0), "living_area_share"] = np.nan

    if {"floor", "floors_total"}.issubset(df.columns):
        df["floor_ratio"] = safe_divide(df["floor"], df["floors_total"])
        df.loc[~df["floor_ratio"].between(0, 1.0), "floor_ratio"] = np.nan
        df["is_first_floor"] = (df["floor"] == 1).astype("int8")
        df["is_last_floor"] = (
            df["floor"].notna()
            & df["floors_total"].notna()
            & (df["floor"] == df["floors_total"])
        ).astype("int8")

    if "house_year" in df.columns:
        df.loc[~df["house_year"].between(1800, REFERENCE_YEAR + 10), "house_year"] = np.nan
        df["house_age"] = REFERENCE_YEAR - df["house_year"]
        df["is_new_building_year"] = (df["house_year"] >= REFERENCE_YEAR).fillna(False).astype("int8")

    if {"rooms", "total_area_m2"}.issubset(df.columns):
        rooms_for_div = df["rooms"].fillna(1).clip(lower=1)
        df["area_per_room_m2"] = df["total_area_m2"] / rooms_for_div

    if {"floor", "floors_total"}.issubset(df.columns):
        def floor_group(row: pd.Series) -> str:
            floor = row.get("floor")
            total = row.get("floors_total")
            if pd.isna(floor) or pd.isna(total) or total <= 0:
                return "unknown"
            if floor == 1:
                return "first"
            if floor == total:
                return "last"
            ratio = floor / total
            if ratio <= 0.33:
                return "low"
            if ratio <= 0.66:
                return "middle"
            return "high"

        df["floor_group"] = df.apply(floor_group, axis=1)

    categorical_cols = existing(BASE_CATEGORICAL_COLUMNS, df)
    if not USE_STREET_FEATURE and "street" in categorical_cols:
        categorical_cols.remove("street")

    for col in categorical_cols:
        df[col] = df[col].map(normalize_missing_string).astype(str)

    binary_cols = existing(BINARY_COLUMNS, df)
    for col in binary_cols:
        df[col] = to_binary(df[col])

    osm_numeric_patterns = (
        r"^osm_.*_count_1000m$",
        r"^log1p_osm_.*_count_1000m$",
        r"^osm_total_poi_count_1000m$",
        r"^log1p_osm_total_poi_count_1000m$",
        r"^infrastructure_score_1000m$",
    )

    for col in df.columns:
        if any(re.match(pattern, col) for pattern in osm_numeric_patterns):
            df[col] = to_numeric(df[col]).fillna(0)

    nearest_cols = [c for c in df.columns if re.match(r"^osm_.*_nearest_1000m$", c)]
    osm_available = df["osm_features_available"] if "osm_features_available" in df.columns else pd.Series(1, index=df.index)

    for nearest_col in nearest_cols:
        df[nearest_col] = to_numeric(df[nearest_col])
        category = nearest_col.replace("osm_", "").replace("_nearest_1000m", "")
        count_col = f"osm_{category}_count_1000m"
        if count_col in df.columns:
            no_object_mask = (df[count_col].fillna(0) == 0) & (osm_available == 1)
            df.loc[no_object_mask, nearest_col] = df.loc[no_object_mask, nearest_col].fillna(OSM_NOT_FOUND_DISTANCE_M)

    protected_cols = set(categorical_cols + [TARGET_COL, ALT_LOG_TARGET_COL])
    for col in df.columns:
        if col in protected_cols:
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            converted = to_numeric(df[col])
            non_null_original = int(df[col].notna().sum())
            if non_null_original > 0 and converted.notna().sum() >= max(1, int(0.8 * non_null_original)):
                df[col] = converted

    df = df.replace([np.inf, -np.inf], np.nan)

    all_missing_cols = [
        c for c in df.columns
        if c not in {TARGET_COL, ALT_LOG_TARGET_COL} and df[c].isna().all()
    ]
    df = df.drop(columns=all_missing_cols, errors="ignore")

    categorical_cols = existing(categorical_cols, df)
    binary_cols = existing(binary_cols, df)

    constant_cols = []
    for col in df.columns:
        if col in {TARGET_COL, ALT_LOG_TARGET_COL}:
            continue
        if df[col].nunique(dropna=False) <= 1:
            constant_cols.append(col)

    df = df.drop(columns=constant_cols, errors="ignore")
    categorical_cols = [c for c in categorical_cols if c not in constant_cols]
    binary_cols = [c for c in binary_cols if c not in constant_cols]

    feature_cols = [c for c in df.columns if c not in {TARGET_COL, ALT_LOG_TARGET_COL}]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols and c not in binary_cols]

    ordered_cols = [TARGET_COL, ALT_LOG_TARGET_COL] + categorical_cols + binary_cols + numeric_cols
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols].copy()

    config: Dict[str, Any] = {
        "target": TARGET_COL,
        "alternative_log_target": ALT_LOG_TARGET_COL,
        "features": [c for c in df.columns if c not in {TARGET_COL, ALT_LOG_TARGET_COL}],
        "cat_features": categorical_cols,
        "binary_features": binary_cols,
        "numeric_features": numeric_cols,
        "dropped_columns": sorted(set(drop_cols + all_missing_cols + constant_cols)),
        "settings": {
            "min_price_rub": MIN_PRICE_RUB,
            "max_price_rub": MAX_PRICE_RUB,
            "min_total_area_m2": MIN_TOTAL_AREA_M2,
            "max_total_area_m2": MAX_TOTAL_AREA_M2,
            "drop_rows_without_coords": DROP_ROWS_WITHOUT_COORDS,
            "use_street_feature": USE_STREET_FEATURE,
            "osm_not_found_distance_m": OSM_NOT_FOUND_DISTANCE_M,
            "reference_year": REFERENCE_YEAR,
        },
        "data_report": {
            "original_rows": int(original_shape[0]),
            "original_columns": int(original_shape[1]),
            "rows_after_duplicate_drop": int(len(df) + len(excluded_rows)),
            "duplicated_url_rows_removed": int(duplicated_rows_removed),
            "excluded_rows": int(len(excluded_rows)),
            "final_rows": int(len(df)),
            "final_columns": int(df.shape[1]),
            "feature_count": int(len(feature_cols)),
            "cat_feature_count": int(len(categorical_cols)),
            "binary_feature_count": int(len(binary_cols)),
            "numeric_feature_count": int(len(numeric_cols)),
        },
    }

    return df, config, excluded_rows


def make_X_y(df: pd.DataFrame, features: List[str], target_col: str, cat_features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[features].copy()
    y = df[target_col].copy()

    # CatBoost корректно работает с NaN в числовых признаках, но категориальные лучше держать строками.
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna("unknown").astype(str)

    return X, y


def inverse_target(values: np.ndarray | pd.Series, target_mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if target_mode == "log":

        return np.expm1(arr)
    return arr


def calc_metrics(y_true_model_scale: pd.Series | np.ndarray, y_pred_model_scale: np.ndarray, target_mode: str) -> Dict[str, float]:
    y_true = inverse_target(y_true_model_scale, target_mode)
    y_pred = inverse_target(y_pred_model_scale, target_mode)


    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan

    return {
        "mae_rub": float(mae),
        "rmse_rub": float(rmse),
        "median_ae_rub": float(medae),
        "mape_pct": float(mape),
        "r2": float(r2),
    }


def select_objective_metric(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == "mae":
        return metrics["mae_rub"]
    if metric_name == "rmse":
        return metrics["rmse_rub"]
    if metric_name == "mape":
        return metrics["mape_pct"]
    raise ValueError(f"Неизвестная метрика для Optuna: {metric_name}")


def make_base_catboost_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": args.random_state,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": args.thread_count,
        "task_type": args.task_type,
    }


def suggest_catboost_params(trial: Any) -> Dict[str, Any]:
    # Пространство поиска Optuna.
    params: Dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", 200, 900),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 2),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
        "border_count": trial.suggest_int("border_count", 64, 192),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

    return params


def default_catboost_params(args: argparse.Namespace) -> Dict[str, Any]:
    params = make_base_catboost_params(args)
    params.update({
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.035,
        "l2_leaf_reg": 6.0,
        "random_strength": 1.0,
        "border_count": 128,
        "min_data_in_leaf": 5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "max_ctr_complexity": 1,
        "one_hot_max_size": 10,
    })
    return params


def train_one_model(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    cat_features: List[str],
    early_stopping_rounds: int | None,
    verbose: bool | int = False,
) -> CatBoostRegressor:
    model = CatBoostRegressor(**params)

    fit_kwargs: Dict[str, Any] = {
        "X": X_train,
        "y": y_train,
        "cat_features": cat_features,
        "verbose": verbose,
    }

    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = (X_valid, y_valid)
        fit_kwargs["use_best_model"] = True
        if early_stopping_rounds and early_stopping_rounds > 0:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

    model.fit(**fit_kwargs)
    return model


def run_optuna_search(
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: List[str],
    target_mode: str,
) -> Tuple[Dict[str, Any], Any]:
    if optuna is None:
        raise ImportError("Не установлен optuna. Установи: pip install optuna")

    base_params = make_base_catboost_params(args)

    def objective(trial: Any) -> float:
        params = base_params.copy()
        params.update(suggest_catboost_params(trial))

        model = train_one_model(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            cat_features=cat_features,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=False,
        )

        pred_valid = model.predict(X_valid)
        metrics = calc_metrics(y_valid, pred_valid, target_mode=target_mode)

        best_iteration = model.get_best_iteration()
        if best_iteration is None or best_iteration <= 0:
            best_iteration = params.get("iterations")

        trial.set_user_attr("best_iteration", int(best_iteration))
        for key, value in metrics.items():
            trial.set_user_attr(key, float(value))

        return select_objective_metric(metrics, args.optuna_metric)

    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=args.study_name,
        storage=args.optuna_storage,
        load_if_exists=bool(args.optuna_storage),
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.optuna_jobs,
        show_progress_bar=args.show_progress,
    )

    best_params = base_params.copy()
    best_params.update(study.best_params)


    best_iter = study.best_trial.user_attrs.get("best_iteration")
    if best_iter is not None:

        tuned_iterations = int(min(best_params.get("iterations", best_iter), max(100, round(best_iter * 1.10))))
        best_params["iterations"] = tuned_iterations

    return best_params, study


def build_artifacts_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "train_ready_csv": output_dir / "apartments_train_ready.csv",
        "excluded_csv": output_dir / "apartments_excluded_from_training.csv",
        "feature_config_json": output_dir / "model_feature_config.json",
        "metadata_json": output_dir / "model_training_metadata.json",
        "metrics_json": output_dir / "model_metrics.json",
        "best_params_json": output_dir / "optuna_best_params.json",
        "trials_csv": output_dir / "optuna_trials.csv",
        "test_predictions_csv": output_dir / "test_predictions.csv",
        "eval_model_cbm": output_dir / "catboost_apartment_price_eval.cbm",
        "final_model_cbm": output_dir / "catboost_apartment_price_final.cbm",
    }


def train_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    global USE_STREET_FEATURE
    USE_STREET_FEATURE = bool(args.use_street) and not bool(args.no_street)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = build_artifacts_paths(output_dir)

    print("1/6 Читаю входной файл...")
    raw_df = read_csv_safely(args.input)

    print("2/6 Готовлю train-ready данные...")
    train_ready, config, excluded_rows = prepare_training_data(raw_df)
    train_ready.to_csv(paths["train_ready_csv"], index=False, encoding="utf-8-sig")
    excluded_rows.to_csv(paths["excluded_csv"], index=False, encoding="utf-8-sig")

    with open(paths["feature_config_json"], "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    target_col = ALT_LOG_TARGET_COL if args.target_mode == "log" else TARGET_COL
    features = config["features"]
    cat_features = [c for c in config["cat_features"] if c in features]

    print("3/6 Делю данные на train / valid / test...")
    train_valid_df, test_df = train_test_split(
        train_ready,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    valid_relative_size = args.valid_size / (1.0 - args.test_size)
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_relative_size,
        random_state=args.random_state,
        shuffle=True,
    )

    X_train, y_train = make_X_y(train_df, features, target_col, cat_features)
    X_valid, y_valid = make_X_y(valid_df, features, target_col, cat_features)
    X_train_valid, y_train_valid = make_X_y(train_valid_df, features, target_col, cat_features)
    X_test, y_test = make_X_y(test_df, features, target_col, cat_features)
    X_all, y_all = make_X_y(train_ready, features, target_col, cat_features)

    study = None
    if args.n_trials > 0:
        print(f"4/6 Запускаю Optuna: n_trials={args.n_trials}, metric={args.optuna_metric}...")
        best_params, study = run_optuna_search(
            args=args,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            cat_features=cat_features,
            target_mode=args.target_mode,
        )

        with open(paths["best_params_json"], "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

        try:
            study.trials_dataframe().to_csv(paths["trials_csv"], index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"Не удалось сохранить optuna_trials.csv: {e}")
    else:
        print("4/6 Optuna пропущена, использую базовые параметры...")
        best_params = default_catboost_params(args)
        with open(paths["best_params_json"], "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

    print("5/6 Обучаю evaluation-модель на train+valid и считаю test-метрики...")
    eval_model = train_one_model(
        params=best_params,
        X_train=X_train_valid,
        y_train=y_train_valid,
        X_valid=None,
        y_valid=None,
        cat_features=cat_features,
        early_stopping_rounds=None,
        verbose=args.train_verbose,
    )

    test_pred_model_scale = eval_model.predict(X_test)
    test_metrics = calc_metrics(y_test, test_pred_model_scale, target_mode=args.target_mode)
    eval_model.save_model(str(paths["eval_model_cbm"]))

    test_predictions = test_df[[TARGET_COL, ALT_LOG_TARGET_COL]].copy()
    test_predictions["prediction_model_scale"] = test_pred_model_scale
    test_predictions["prediction_price_rub"] = inverse_target(test_pred_model_scale, args.target_mode)
    test_predictions["prediction_price_rub"] = np.maximum(test_predictions["prediction_price_rub"], 0)
    test_predictions["abs_error_rub"] = np.abs(test_predictions[TARGET_COL] - test_predictions["prediction_price_rub"])
    test_predictions.to_csv(paths["test_predictions_csv"], index=False, encoding="utf-8-sig")

    print("6/6 Обучаю финальную production-модель на всех очищенных данных и сохраняю...")
    final_model = train_one_model(
        params=best_params,
        X_train=X_all,
        y_train=y_all,
        X_valid=None,
        y_valid=None,
        cat_features=cat_features,
        early_stopping_rounds=None,
        verbose=args.train_verbose,
    )
    final_model.save_model(str(paths["final_model_cbm"]))

    metadata: Dict[str, Any] = {
        "created_at_unix": int(time.time()),
        "input_path": str(args.input),
        "output_dir": str(output_dir),
        "target_mode": args.target_mode,
        "target_col_used_for_training": target_col,
        "production_target_note": "Если target_mode='log', model.predict() возвращает log1p(price_rub); для цены используй np.expm1(pred).",
        "features": features,
        "cat_features": cat_features,
        "binary_features": config["binary_features"],
        "numeric_features": config["numeric_features"],
        "data_report": config["data_report"],
        "split": {
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
            "train_valid_rows": int(len(train_valid_df)),
            "all_clean_rows": int(len(train_ready)),
            "test_size": args.test_size,
            "valid_size": args.valid_size,
            "random_state": args.random_state,
        },
        "best_params": best_params,
        "test_metrics": test_metrics,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "catboost_usage_note": "Для новых данных сначала сделай ту же подготовку признаков, затем X = df[features], категориальные колонки заполни 'unknown' и astype(str).",
    }

    with open(paths["metadata_json"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(paths["metrics_json"], "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print("\nГотово.")
    print(f"Train-ready CSV: {paths['train_ready_csv']}")
    print(f"Feature config:   {paths['feature_config_json']}")
    print(f"Eval model:       {paths['eval_model_cbm']}")
    print(f"Final model:      {paths['final_model_cbm']}")
    print(f"Metadata:         {paths['metadata_json']}")
    print("\nTest metrics в рублях:")
    for k, v in test_metrics.items():
        if k.endswith("_rub"):
            print(f"  {k}: {v:,.0f}")
        elif k.endswith("_pct"):
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск...")

    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Входной CSV, например apartments_ml.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Папка для артефактов модели")

    parser.add_argument("--target-mode", choices=["log", "price"], default="log", help="log=обучать log1p(price_rub), price=обучать price_rub")
    parser.add_argument("--optuna-metric", choices=["mae", "rmse", "mape"], default="mae", help="Метрика, которую минимизирует Optuna в рублях/процентах")
    parser.add_argument("--n-trials", type=int, default=25, help="Количество Optuna trials. 0 = без Optuna, базовые параметры")
    parser.add_argument("--timeout", type=int, default=None, help="Лимит Optuna в секундах. Например 1800")
    parser.add_argument("--optuna-storage", default=None, help="Например sqlite:///optuna_catboost.db для сохранения study")
    parser.add_argument("--study-name", default="catboost_apartment_price", help="Имя Optuna study")
    parser.add_argument("--optuna-jobs", type=int, default=1, help="Параллельные trials. Для CatBoost обычно оставь 1")
    parser.add_argument("--show-progress", action="store_true", help="Показывать progress bar Optuna")

    parser.add_argument("--test-size", type=float, default=0.20, help="Доля test от всех данных")
    parser.add_argument("--valid-size", type=float, default=0.16, help="Доля valid от всех данных")
    parser.add_argument("--random-state", type=int, default=42, help="Seed")

    parser.add_argument("--early-stopping-rounds", type=int, default=150, help="Early stopping для Optuna trials")
    parser.add_argument("--thread-count", type=int, default=-1, help="Потоки CatBoost; -1 = все")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU", help="CPU или GPU")
    parser.add_argument("--train-verbose", type=int, default=100, help="Verbose для финального обучения; 0=False")

    parser.add_argument("--use-street", action="store_true", help="Включить street как категориальный признак. По умолчанию выключено из-за риска переобучения")
    parser.add_argument("--no-street", action="store_true", help="Оставлено для совместимости: принудительно исключить street")

    args = parser.parse_args()

    if args.train_verbose == 0:
        args.train_verbose = False

    if args.test_size <= 0 or args.test_size >= 0.5:
        raise ValueError("--test-size должен быть >0 и <0.5")
    if args.valid_size <= 0 or args.valid_size >= 0.5:
        raise ValueError("--valid-size должен быть >0 и <0.5")
    if args.test_size + args.valid_size >= 0.8:
        raise ValueError("Сумма --test-size и --valid-size слишком большая")

    return args


if __name__ == "__main__":
    train_pipeline(parse_args())
