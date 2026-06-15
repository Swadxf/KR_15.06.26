import json

from .config import METRICS_PATH


def main() -> None:
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(json.dumps(report["test"]["overall"], ensure_ascii=False, indent=2))
    if "improvement_vs_previous_mae_rub" in report:
        print(
            "Improvement vs previous pipeline: "
            f"{report['improvement_vs_previous_mae_rub']:,.0f} RUB"
        )


if __name__ == "__main__":
    main()
