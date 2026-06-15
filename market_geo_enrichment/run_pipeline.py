import argparse

from mae_optimization.train_models import parse_seeds

from .train_extended import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--skip-production", action="store_true")
    args = parser.parse_args()
    report = train_pipeline(
        seeds=parse_seeds(args.seeds),
        train_production=not args.skip_production,
    )
    overall = report["test"]["overall"]
    print(
        f"MAE={overall['mae_rub']:,.0f} RUB, "
        f"MedAE={overall['median_ae_rub']:,.0f} RUB"
    )


if __name__ == "__main__":
    main()
