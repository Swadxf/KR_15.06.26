import argparse
import json

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor

from .config import (
    PREPARED_PATH,
    RANDOM_STATE,
    TUNED_TOTAL_PARAMS_PATH,
    TUNED_UNIT_PARAMS_PATH,
)
from .evaluation import regression_metrics
from .modeling import (
    TOTAL_SPEC,
    UNIT_SPEC,
    make_xy,
    prediction_to_price,
    prepare_model_frame,
)
from .train_models import split_indices


def suggest_params(trial: optuna.Trial) -> dict:
    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
    )
    params = {
        "iterations": trial.suggest_int("iterations", 350, 1400),
        "depth": trial.suggest_int("depth", 5, 9),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.015, 0.15, log=True
        ),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 40, log=True),
        "random_strength": trial.suggest_float("random_strength", 0, 4),
        "border_count": trial.suggest_int("border_count", 96, 224),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 60),
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 3),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 25),
        "bootstrap_type": bootstrap_type,
    }
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0, 5
        )
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    return params


def tune(spec, frame, config, train, valid, trials: int) -> dict:
    X_train, y_train, categorical = make_xy(frame, train, config, spec)
    X_valid, _, _ = make_xy(frame, valid, config, spec)
    actual = frame.iloc[valid]["price_rub"].to_numpy(dtype=float)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        params.update(
            {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": RANDOM_STATE,
                "allow_writing_files": False,
                "verbose": False,
            }
        )
        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            cat_features=categorical,
            verbose=False,
        )
        prediction = prediction_to_price(
            model.predict(X_valid),
            frame.iloc[valid],
            spec,
        )
        return regression_metrics(actual, prediction)["mae_rub"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    best = {
        **study.best_params,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
        "verbose": False,
    }
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    prepared = pd.read_csv(PREPARED_PATH, encoding="utf-8-sig", low_memory=False)
    frame, config, _ = prepare_model_frame(prepared)
    train, valid, _, _ = split_indices(len(frame))

    for spec, path in [
        (TOTAL_SPEC, TUNED_TOTAL_PARAMS_PATH),
        (UNIT_SPEC, TUNED_UNIT_PARAMS_PATH),
    ]:
        params = tune(spec, frame, config, train, valid, args.trials)
        path.write_text(
            json.dumps(params, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"{spec.name}: {path}")


if __name__ == "__main__":
    main()
