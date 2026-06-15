import json

import pandas as pd

from .config import BLEND_CONFIG_PATH, METRICS_PATH, TEST_PREDICTIONS_PATH
from .evaluation import segmented_metrics


def evaluate() -> dict:
    predictions = pd.read_csv(
        TEST_PREDICTIONS_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )
    blend_config = json.loads(BLEND_CONFIG_PATH.read_text(encoding="utf-8"))
    metrics = {
        "blend": blend_config,
        "test": segmented_metrics(predictions),
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    metrics = evaluate()
    print(json.dumps(metrics["test"]["overall"], ensure_ascii=False, indent=2))
    print(f"Full report: {METRICS_PATH}")


if __name__ == "__main__":
    main()

