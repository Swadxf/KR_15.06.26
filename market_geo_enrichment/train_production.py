import json

import numpy as np
import pandas as pd

from mae_optimization.modeling import (
    TOTAL_SPEC,
    UNIT_SPEC,
    save_models,
    train_ensemble,
)

from .comparables import REFERENCE_COLUMNS
from .config import (
    COMPARABLES_REFERENCE_PATH,
    FEATURE_CONFIG_PATH,
    MODELS_DIR,
    TRAIN_READY_PATH,
)


def train_production_models(seeds: list[int]) -> None:
    ready = pd.read_csv(
        TRAIN_READY_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )
    feature_config = json.loads(
        FEATURE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    all_indices = np.arange(len(ready))

    for spec in [TOTAL_SPEC, UNIT_SPEC]:
        models = train_ensemble(
            ready,
            all_indices,
            feature_config,
            spec,
            seeds,
        )
        save_models(
            models,
            MODELS_DIR / "production",
            spec,
            seeds,
        )

    ready[REFERENCE_COLUMNS].to_csv(
        COMPARABLES_REFERENCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    train_production_models([42])
    print(f"Production models saved to {MODELS_DIR / 'production'}")


if __name__ == "__main__":
    main()
