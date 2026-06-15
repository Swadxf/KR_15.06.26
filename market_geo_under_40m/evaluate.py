import json

from .config import METRICS_PATH


def main() -> None:
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(json.dumps(report["test"]["overall"], ensure_ascii=False, indent=2))
    print(
        "Retraining improvement on the same retained test rows: "
        f"{report['retraining_improvement_mae_rub']:,.0f} RUB"
    )


if __name__ == "__main__":
    main()
