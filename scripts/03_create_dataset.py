import calendar
from pathlib import Path
from typing import Any

import camelot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# FUNCTIONS
def get_config() -> dict:
    with Path("config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_year_month_from_report_path(filepath: Path) -> tuple[int, str]:
    parts = filepath.stem.split("-")
    year, month_name = int(parts[-2]), str(parts[-1])
    return year, month_name


def extract_report_data(filepath: Path) -> dict[str, Any] | None:
    production_kg = np.nan

    tables = camelot.read_pdf(filepath, flavor="lattice")
    for col in [2, 3]:
        try:
            production_kg = pd.to_numeric(
                pd.to_numeric(
                    tables[0].df[col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ).astype(float)
            )
            break
        except Exception:
            continue

    year, month = extract_year_month_from_report_path(filepath)

    return {
        "date": pd.to_datetime(f"{month} {year}", format="%B %Y"),
        "year": year,
        "month": month,
        "metric": "tea_production",
        "value": np.max(production_kg),
        "unit": "kg",
        "region": "all",
        "source": filepath.name,
    }


# MAIN
def main():
    config = get_config()
    ocr_dir = Path(config["ocr_dir"])
    data_file = Path(config["data_file"])

    # Build tea performance dataset
    data = []

    for filepath in ocr_dir.glob("*.pdf"):
        year, month_name = extract_year_month_from_report_path(filepath)

        try:
            data.append(extract_report_data(filepath))
            print(f"  ✓ success | extracted data for {year}-{month_name}")
        except Exception:
            print(f"  ✗ failed  | could not extract data for {year}-{month_name}")

    # Write out results
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(data).set_index("date").sort_index()
    data.to_csv(data_file)


if __name__ == "__main__":
    main()
