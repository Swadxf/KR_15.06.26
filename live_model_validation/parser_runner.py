import subprocess
import sys
from pathlib import Path

from .config import (
    AVITO_PARSER_PATH,
    AVITO_PROFILE_DIR,
    CIAN_PARSER_PATH,
    CIAN_PROFILE_DIR,
)


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def run_avito_parser(
    run_dir: Path,
    headless: bool,
    max_cards: int,
) -> Path:
    links_path = run_dir / "avito_links.csv"
    raw_path = run_dir / "avito_raw.csv"
    command = [
        sys.executable,
        str(AVITO_PARSER_PATH),
        "--mode",
        "all",
        "--links-csv",
        str(links_path),
        "--csv",
        str(raw_path),
        "--profile-dir",
        str(AVITO_PROFILE_DIR),
        "--max-pages-per-group",
        "1",
        "--max-segments",
        "1",
        "--max-links",
        str(max_cards),
        "--max-items",
        str(max_cards),
    ]
    if headless:
        command.append("--headless")
    _run(command, AVITO_PARSER_PATH.parent)
    return raw_path


def run_cian_parser(
    run_dir: Path,
    headless: bool,
    max_cards: int,
) -> Path:
    links_path = run_dir / "cian_links.csv"
    raw_path = run_dir / "cian_raw.csv"
    command = [
        sys.executable,
        str(CIAN_PARSER_PATH),
        "--mode",
        "all",
        "--links-csv",
        str(links_path),
        "--csv",
        str(raw_path),
        "--profile-dir",
        str(CIAN_PROFILE_DIR),
        "--max-pages-per-group",
        "1",
        "--max-segments",
        "1",
        "--max-links",
        str(max_cards),
        "--max-items",
        str(max_cards),
    ]
    if headless:
        command.append("--headless")
    _run(command, CIAN_PARSER_PATH.parent)
    return raw_path
