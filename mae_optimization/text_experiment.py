import json
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from .config import (
    ARTIFACTS_DIR,
    AVITO_RAW_PATH,
    CIAN_RAW_PATH,
    EXCLUDED_PATH,
    PREPARED_PATH,
    SPLIT_PATH,
    TEST_PREDICTIONS_PATH,
    VALIDATION_PREDICTIONS_PATH,
)


REPORT_PATH = ARTIFACTS_DIR / "text_experiment.json"
ALPHAS = [10.0, 30.0, 100.0, 300.0]
BLEND_WEIGHTS = np.linspace(0.0, 0.30, 31)


def normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def sanitize_text(value: object) -> str:
    text = str(value) if pd.notna(value) else ""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+(?:[.,]\d+)*", " <num> ", text)
    text = re.sub(r"[^\w<>]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_raw_text() -> pd.DataFrame:
    avito = pd.read_csv(
        AVITO_RAW_PATH,
        sep=";",
        encoding="utf-8-sig",
        on_bad_lines="skip",
        low_memory=False,
    )
    cian = pd.read_csv(
        CIAN_RAW_PATH,
        sep=";",
        encoding="utf-8-sig",
        on_bad_lines="skip",
        low_memory=False,
    )

    avito_text = (
        avito.iloc[:, 2].fillna("").astype(str)
        + " "
        + avito.iloc[:, 21].fillna("").astype(str)
        + " "
        + avito.iloc[:, -1].fillna("").astype(str)
    )
    cian_text = cian.iloc[:, -1].fillna("").astype(str)

    avito_output = pd.DataFrame(
        {
            "source": "avito",
            "source_listing_id": normalize_id(avito["source_listing_id"]),
            "text": avito_text.map(sanitize_text),
        }
    )
    cian_output = pd.DataFrame(
        {
            "source": "cian",
            "source_listing_id": normalize_id(cian["source_listing_id"]),
            "text": cian_text.map(sanitize_text),
        }
    )
    return pd.concat([avito_output, cian_output], ignore_index=True)


def build_model_rows() -> pd.DataFrame:
    prepared = pd.read_csv(PREPARED_PATH, encoding="utf-8-sig", low_memory=False)
    excluded = pd.read_csv(EXCLUDED_PATH, encoding="utf-8-sig", low_memory=False)
    raw_text = read_raw_text()

    for frame in [prepared, excluded]:
        frame["source_listing_id"] = normalize_id(frame["source_listing_id"])

    excluded_keys = set(
        zip(excluded["source"].astype(str), excluded["source_listing_id"])
    )
    keep = [
        (str(source), listing_id) not in excluded_keys
        for source, listing_id in zip(
            prepared["source"],
            prepared["source_listing_id"],
        )
    ]
    rows = prepared.loc[keep].reset_index(drop=True)
    rows = rows.merge(
        raw_text,
        on=["source", "source_listing_id"],
        how="left",
        validate="one_to_one",
    )
    rows["text"] = rows["text"].fillna("")

    split = pd.read_csv(SPLIT_PATH)
    if len(rows) != len(split):
        raise RuntimeError(
            f"Text rows ({len(rows)}) do not match saved split ({len(split)})"
        )
    rows["row_index"] = np.arange(len(rows))
    rows["split"] = split["split"].to_numpy()
    return rows


def make_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=60_000,
        sublinear_tf=True,
        strip_accents=None,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=4,
        max_features=60_000,
        sublinear_tf=True,
    )
    return word, char


def vectorize(
    train_text: pd.Series,
    evaluation_text: pd.Series,
) -> tuple[object, object]:
    word, char = make_vectorizers()
    word_train = word.fit_transform(train_text)
    char_train = char.fit_transform(train_text)
    train_matrix = hstack([word_train, char_train], format="csr")
    evaluation_matrix = hstack(
        [word.transform(evaluation_text), char.transform(evaluation_text)],
        format="csr",
    )
    return train_matrix, evaluation_matrix


def choose_text_model(
    train_matrix: object,
    valid_matrix: object,
    target_train: np.ndarray,
    actual_valid: np.ndarray,
    area_valid: np.ndarray,
    baseline_valid: np.ndarray,
) -> dict:
    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(train_matrix, target_train)
        prediction = np.expm1(model.predict(valid_matrix)) * area_valid
        text_mae = float(mean_absolute_error(actual_valid, prediction))

        for weight in BLEND_WEIGHTS:
            blended = (1 - weight) * baseline_valid + weight * prediction
            mae = float(mean_absolute_error(actual_valid, blended))
            candidate = {
                "alpha": alpha,
                "text_weight": float(weight),
                "validation_mae_rub": mae,
                "text_only_validation_mae_rub": text_mae,
            }
            if best is None or mae < best["validation_mae_rub"]:
                best = candidate
    return best


def run_experiment() -> dict:
    rows = build_model_rows()
    train_mask = rows["split"].eq("train").to_numpy()
    valid_mask = rows["split"].eq("valid").to_numpy()
    test_mask = rows["split"].eq("test").to_numpy()

    valid_baseline_frame = pd.read_csv(VALIDATION_PREDICTIONS_PATH)
    test_baseline_frame = pd.read_csv(TEST_PREDICTIONS_PATH)
    baseline_valid_by_index = valid_baseline_frame.set_index("row_index")[
        "prediction_price_rub"
    ]
    baseline_test_by_index = test_baseline_frame.set_index("row_index")[
        "prediction_price_rub"
    ]
    baseline_valid = baseline_valid_by_index.loc[
        rows.loc[valid_mask, "row_index"]
    ].to_numpy()
    baseline_test = baseline_test_by_index.loc[
        rows.loc[test_mask, "row_index"]
    ].to_numpy()

    train_matrix, valid_matrix = vectorize(
        rows.loc[train_mask, "text"],
        rows.loc[valid_mask, "text"],
    )
    target_train = np.log1p(
        rows.loc[train_mask, "price_rub"].to_numpy()
        / rows.loc[train_mask, "total_area_m2"].to_numpy()
    )
    actual_valid = rows.loc[valid_mask, "price_rub"].to_numpy()
    selection = choose_text_model(
        train_matrix,
        valid_matrix,
        target_train,
        actual_valid,
        rows.loc[valid_mask, "total_area_m2"].to_numpy(),
        baseline_valid,
    )

    train_valid_mask = ~test_mask
    train_valid_matrix, test_matrix = vectorize(
        rows.loc[train_valid_mask, "text"],
        rows.loc[test_mask, "text"],
    )
    target_train_valid = np.log1p(
        rows.loc[train_valid_mask, "price_rub"].to_numpy()
        / rows.loc[train_valid_mask, "total_area_m2"].to_numpy()
    )
    model = Ridge(alpha=selection["alpha"], solver="lsqr")
    model.fit(train_valid_matrix, target_train_valid)
    text_test = (
        np.expm1(model.predict(test_matrix))
        * rows.loc[test_mask, "total_area_m2"].to_numpy()
    )
    actual_test = rows.loc[test_mask, "price_rub"].to_numpy()
    weight = selection["text_weight"]
    blended_test = (1 - weight) * baseline_test + weight * text_test

    baseline_valid_mae = float(
        mean_absolute_error(actual_valid, baseline_valid)
    )
    baseline_test_mae = float(mean_absolute_error(actual_test, baseline_test))
    report = {
        "accepted_by_validation": (
            selection["validation_mae_rub"] < baseline_valid_mae
            and selection["text_weight"] > 0
        ),
        "numbers_removed_from_text": True,
        "selection": selection,
        "baseline_validation_mae_rub": baseline_valid_mae,
        "baseline_test_mae_rub": baseline_test_mae,
        "text_blend_test_mae_rub": float(
            mean_absolute_error(actual_test, blended_test)
        ),
        "text_only_test_mae_rub": float(
            mean_absolute_error(actual_test, text_test)
        ),
        "rows": {
            "train": int(train_mask.sum()),
            "valid": int(valid_mask.sum()),
            "test": int(test_mask.sum()),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = run_experiment()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
