import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import (
    BASE_PARAMS_PATH,
    ENHANCED_CATEGORICAL,
    LEAKAGE_COLUMNS,
    ROOT_DIR,
    TUNED_TOTAL_PARAMS_PATH,
    TUNED_UNIT_PARAMS_PATH,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_mode: str
    extra_categorical: tuple[str, ...]


TOTAL_SPEC = ModelSpec(
    name="total_price",
    target_mode="total",
    extra_categorical=("address_zone",),
)

UNIT_SPEC = ModelSpec(
    name="unit_price",
    target_mode="unit",
    extra_categorical=(
        "address_exact",
        "address_zone",
        "geo_cell_2000m",
        "geo_cell_1000m",
        "geo_cell_500m",
        "geo_cell_200m",
    ),
)


def _import_existing_training_helpers():
    root = str(ROOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    from train_catboost_optuna_pipeline import prepare_training_data

    return prepare_training_data


def prepare_model_frame(
    prepared: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    prepare_training_data = _import_existing_training_helpers()
    ready, config, excluded = prepare_training_data(prepared)

    extra_categorical = [
        column
        for column in ENHANCED_CATEGORICAL
        if column in ready.columns and ready[column].nunique(dropna=False) > 1
    ]
    for column in extra_categorical:
        ready[column] = ready[column].fillna("unknown").astype(str)

    config["cat_features"] = list(
        dict.fromkeys(config["cat_features"] + extra_categorical)
    )
    config["numeric_features"] = [
        column
        for column in config["numeric_features"]
        if column not in extra_categorical
    ]

    leaked = LEAKAGE_COLUMNS.intersection(config["features"])
    if leaked:
        raise RuntimeError(f"Target leakage columns reached model features: {sorted(leaked)}")

    return ready, config, excluded


def model_columns(config: dict, spec: ModelSpec) -> tuple[list[str], list[str]]:
    enhanced = set(ENHANCED_CATEGORICAL)
    base_features = [
        column for column in config["features"] if column not in enhanced
    ]
    features = list(dict.fromkeys(base_features + list(spec.extra_categorical)))
    categorical = [
        column
        for column in config["cat_features"]
        if column in features
    ]
    return features, categorical


def target_for_spec(frame: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    if spec.target_mode == "total":
        return frame["price_log_rub"]
    if spec.target_mode == "unit":
        unit_price = frame["price_rub"] / frame["total_area_m2"]
        return np.log1p(unit_price)
    raise ValueError(f"Unknown target mode: {spec.target_mode}")


def prediction_to_price(
    raw_prediction: np.ndarray,
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> np.ndarray:
    prediction = np.expm1(np.asarray(raw_prediction, dtype=float))
    if spec.target_mode == "unit":
        prediction = prediction * frame["total_area_m2"].to_numpy(dtype=float)
    return np.maximum(prediction, 0)


def make_xy(
    frame: pd.DataFrame,
    indices: Iterable[int],
    config: dict,
    spec: ModelSpec,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    index = np.asarray(list(indices), dtype=int)
    features, categorical = model_columns(config, spec)
    X = frame.iloc[index][features].copy()
    for column in categorical:
        X[column] = X[column].fillna("unknown").astype(str)
    y = target_for_spec(frame.iloc[index], spec)
    return X, y, categorical


def load_params(spec: ModelSpec) -> dict:
    tuned_path = (
        TUNED_TOTAL_PARAMS_PATH
        if spec.target_mode == "total"
        else TUNED_UNIT_PARAMS_PATH
    )
    params_path = tuned_path if tuned_path.exists() else BASE_PARAMS_PATH
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params.update(
        {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "allow_writing_files": False,
            "verbose": False,
        }
    )
    return params


def train_ensemble(
    frame: pd.DataFrame,
    train_indices: Iterable[int],
    config: dict,
    spec: ModelSpec,
    seeds: list[int],
) -> list[CatBoostRegressor]:
    X_train, y_train, categorical = make_xy(
        frame, train_indices, config, spec
    )
    models = []
    for seed in seeds:
        params = load_params(spec)
        params["random_seed"] = int(seed)
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, cat_features=categorical, verbose=False)
        models.append(model)
    return models


def predict_ensemble(
    models: list[CatBoostRegressor],
    frame: pd.DataFrame,
    indices: Iterable[int],
    config: dict,
    spec: ModelSpec,
) -> np.ndarray:
    index = np.asarray(list(indices), dtype=int)
    features, categorical = model_columns(config, spec)
    X = frame.iloc[index][features].copy()
    for column in categorical:
        X[column] = X[column].fillna("unknown").astype(str)

    model_scale = np.mean([model.predict(X) for model in models], axis=0)
    return prediction_to_price(model_scale, frame.iloc[index], spec)


def save_models(
    models: list[CatBoostRegressor],
    directory: Path,
    spec: ModelSpec,
    seeds: list[int],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for model, seed in zip(models, seeds):
        model.save_model(str(directory / f"{spec.name}_seed_{seed}.cbm"))

