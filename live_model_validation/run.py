import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import RUNS_DIR
from .parser_runner import run_avito_parser, run_cian_parser
from .predict import predict_raw_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Use existing raw files instead of opening Avito and CIAN.",
    )
    parser.add_argument("--avito-raw", type=Path)
    parser.add_argument("--cian-raw", type=Path)
    parser.add_argument(
        "--max-cards",
        type=int,
        default=0,
        help="Maximum cards from the first page. 0 = every card on the page.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir or (
        RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    parser_errors = {}

    if args.skip_parse:
        if args.avito_raw is None and args.cian_raw is None:
            raise SystemExit(
                "--skip-parse requires --avito-raw and/or --cian-raw"
            )
        avito_raw = args.avito_raw or run_dir / "empty_avito.csv"
        cian_raw = args.cian_raw or run_dir / "empty_cian.csv"
    else:
        avito_raw = run_dir / "avito_raw.csv"
        cian_raw = run_dir / "cian_raw.csv"
        try:
            avito_raw = run_avito_parser(
                run_dir,
                headless=args.headless,
                max_cards=args.max_cards,
            )
        except Exception as error:
            parser_errors["avito"] = str(error)
            print(f"Avito parser failed: {error}")
        try:
            cian_raw = run_cian_parser(
                run_dir,
                headless=args.headless,
                max_cards=args.max_cards,
            )
        except Exception as error:
            parser_errors["cian"] = str(error)
            print(f"CIAN parser failed: {error}")

    if parser_errors:
        (run_dir / "parser_errors.json").write_text(
            json.dumps(parser_errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if not avito_raw.exists() and not cian_raw.exists():
        raise RuntimeError(
            "Neither parser produced raw rows. See parser_errors.json."
        )

    report = predict_raw_files(avito_raw, cian_raw, run_dir)
    report["parser_errors"] = parser_errors
    (run_dir / "fresh_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
