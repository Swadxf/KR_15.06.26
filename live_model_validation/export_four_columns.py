import argparse
from pathlib import Path

import pandas as pd


def export_four_columns(input_path: Path, output_path: Path) -> pd.DataFrame:
    source = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    required = {"price_rub", "prediction_price_rub"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = pd.DataFrame(
        {
            "№ строки": range(1, len(source) + 1),
            "Цена реальная": source["price_rub"],
            "Цена предсказанная": source["prediction_price_rub"],
        }
    )
    result["Разница (предсказанная - реальная)"] = (
        result["Цена предсказанная"] - result["Цена реальная"]
    )
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_name("price_comparison.csv")
    result = export_four_columns(args.input, output)
    print(f"Saved {len(result)} rows to {output.resolve()}")


if __name__ == "__main__":
    main()
