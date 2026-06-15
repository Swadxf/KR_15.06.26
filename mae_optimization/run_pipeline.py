import argparse

from .data_diagnostics import build_diagnostics
from .evaluate_models import evaluate
from .prepare_data import prepare_dataset
from .train_models import parse_seeds, train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--skip-production", action="store_true")
    args = parser.parse_args()

    prepare_dataset()
    train_pipeline(
        seeds=parse_seeds(args.seeds),
        train_production=not args.skip_production,
    )
    metrics = evaluate()
    build_diagnostics()
    overall = metrics["test"]["overall"]
    print(
        f"MAE={overall['mae_rub']:,.0f} RUB, "
        f"MedAE={overall['median_ae_rub']:,.0f} RUB, "
        f"R2={overall['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
