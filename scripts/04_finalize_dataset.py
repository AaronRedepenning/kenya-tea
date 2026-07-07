import argparse
import calendar
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

FINAL_COLUMNS = [
    "date",
    "year",
    "month",
    "metric",
    "region",
    "value",
    "unit",
    "source",
    "source_type",
]

AUDIT_COLUMNS = FINAL_COLUMNS + [
    "source_priority",
    "confidence",
    "period_relation",
    "report_id",
    "report_period_start",
    "observed_period_start",
    "period_label",
    "extraction_method",
    "notes",
]

MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

CONFIDENCE_RANK = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "unknown": 0,
    "": 0,
}

PERIOD_RELATION_PRIORITY = {
    "current_report_month": 1,
    "current_report_year": 1,
    "same_month_previous_year": 2,
    "previous_month": 2,
    "reported_comparison": 3,
    "year_only": 3,
    "unknown": 5,
    "": 5,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create best Kenya tea observations dataset."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/clean/observations_raw.csv"),
        help="Path to observations_raw.csv from the extraction script.",
    )
    parser.add_argument(
        "--hand",
        type=Path,
        default=Path("data/clean/observations_hand_labeled.csv"),
        help="Optional hand-labeled fallback CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/tbk_dataset.csv"),
        help="Output path for the minimal best dataset.",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("data/clean/observations_best_audit.csv"),
        help="Output path for the audit version of the best dataset.",
    )
    parser.add_argument(
        "--include-ytd",
        action="store_true",
        help="Include multi-month/YTD observations. Default keeps monthly observations only.",
    )
    return parser.parse_args()


def month_name(month_num: int) -> str:
    return calendar.month_name[int(month_num)]


def normalize_region(value: Any) -> str:
    """Normalize extracted/manual region values to final output labels."""
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in {"all", "total", "grand total"}:
        return "all"
    if "west" in text:
        return "west_of_rift"
    if "east" in text:
        return "east_of_rift"

    return text or "unknown"


def normalize_metric(value: Any) -> str:
    if pd.isna(value):
        return "tea_production"

    text = str(value).strip().lower()

    if text in {"production", "tea_production"}:
        return "tea_production"

    return text or "tea_production"


def parse_date(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT

    return pd.to_datetime(value, errors="coerce")


def infer_observed_date_from_label(
    period_label: Any, report_year: Any | None = None
) -> pd.Timestamp:
    """
    Fallback parser for period labels like jan_24, jan_2024, april_2022, or 2025.
    This is only used if observed_period_start is missing.
    """
    if pd.isna(period_label):
        return pd.NaT

    label = str(period_label).strip().lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")

    # Year-only column, e.g. 2025. Treat as January of that year only if needed.
    # The main script should usually provide observed_period_start for year-only rows.
    if re.fullmatch(r"\d{4}", label):
        return pd.Timestamp(year=int(label), month=1, day=1)

    # Single month, e.g. jan_24, jan_2024, april_2022.
    match = re.fullmatch(r"([a-z]+)_(\d{2}|\d{4})", label)
    if match:
        month_text, year_text = match.groups()
        month = MONTH_LOOKUP.get(month_text)
        if month is None:
            return pd.NaT
        year = int(year_text)
        if year < 100:
            year += 2000
        return pd.Timestamp(year=year, month=month, day=1)

    # Month range, e.g. jan_april_2022. Use the end month as the observed month.
    match = re.fullmatch(r"([a-z]+)_([a-z]+)_(\d{2}|\d{4})", label)
    if match:
        _start_month_text, end_month_text, year_text = match.groups()
        month = MONTH_LOOKUP.get(end_month_text)
        if month is None:
            return pd.NaT
        year = int(year_text)
        if year < 100:
            year += 2000
        return pd.Timestamp(year=year, month=month, day=1)

    return pd.NaT


def is_monthly_observation(row: pd.Series) -> bool:
    """Keep only single-month observations by default."""
    start = parse_date(row.get("observed_period_start"))
    end = parse_date(row.get("observed_period_end"))
    label = str(row.get("period_label", "")).lower()

    # Exclude obvious YTD / month-range labels.
    if re.search(r"(^|_)jan_[a-z]+_\d{2,4}$", label):
        return False
    if "to" in label or "ytd" in label or "year_to_date" in label:
        return False

    if pd.isna(start):
        start = infer_observed_date_from_label(
            row.get("period_label"), row.get("report_year")
        )

    if pd.isna(start):
        return False

    # If end is missing, assume the label/date is monthly unless label says otherwise.
    if pd.isna(end):
        return True

    # Monthly periods usually span 28-31 days.
    days = (end - start).days + 1
    return 1 <= days <= 31


def safe_numeric(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    text = text.replace(",", "")
    return pd.to_numeric(text, errors="coerce")


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


# -----------------------------------------------------------------------------
# Load and normalize extracted observations
# -----------------------------------------------------------------------------


def load_extracted_observations(path: Path, include_ytd: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw observations file not found: {path}")

    raw = pd.read_csv(path)

    raw = ensure_columns(
        raw,
        [
            "report_id",
            "report_year",
            "report_month",
            "report_period_start",
            "observed_year",
            "observed_month",
            "observed_period_start",
            "observed_period_end",
            "period_label",
            "period_relation",
            "source_priority",
            "metric",
            "sector",
            "region",
            "value",
            "unit",
            "source_file",
            "extraction_method",
            "confidence",
            "notes",
        ],
    )

    # Keep all-sector observations only. Some extractor versions store the all-sector
    # total as region='total'; final output uses region='all'.
    if "sector" in raw.columns:
        raw = raw[
            raw["sector"]
            .astype("string")
            .str.lower()
            .fillna("all")
            .isin(["all", "<na>", "nan"])
        ]

    raw["date"] = raw["observed_period_start"].apply(parse_date)
    missing_date = raw["date"].isna()
    if missing_date.any():
        raw.loc[missing_date, "date"] = raw.loc[missing_date].apply(
            lambda r: infer_observed_date_from_label(
                r.get("period_label"), r.get("report_year")
            ),
            axis=1,
        )

    raw = raw[raw["date"].notna()].copy()

    if not include_ytd:
        raw = raw[raw.apply(is_monthly_observation, axis=1)].copy()

    raw["year"] = raw["date"].dt.year.astype(int)
    raw["month"] = raw["date"].dt.month.apply(month_name)
    raw["metric"] = raw["metric"].apply(normalize_metric)
    raw["region"] = raw["region"].apply(normalize_region)
    raw["value"] = raw["value"].apply(safe_numeric)
    raw["unit"] = raw["unit"].fillna("kg")
    raw["source"] = raw["source_file"].fillna(
        raw.get("report_id", pd.Series(index=raw.index, dtype="object"))
    )
    raw["source_type"] = "extracted"

    raw = raw[raw["region"].isin(["all", "west_of_rift", "east_of_rift"])]
    raw = raw[raw["value"].notna()].copy()

    # Ranking fields.
    raw["source_priority"] = pd.to_numeric(raw["source_priority"], errors="coerce")
    fallback_priority = (
        raw["period_relation"].fillna("").map(PERIOD_RELATION_PRIORITY).fillna(5)
    )
    raw["source_priority"] = raw["source_priority"].fillna(fallback_priority).fillna(5)

    raw["confidence"] = raw["confidence"].fillna("unknown").astype(str).str.lower()
    raw["confidence_rank"] = raw["confidence"].map(CONFIDENCE_RANK).fillna(0)

    # Prefer values from reports whose report period matches the observed period.
    raw["report_period_start_dt"] = raw["report_period_start"].apply(parse_date)
    raw["is_direct_report"] = raw["report_period_start_dt"].notna() & (
        raw["report_period_start_dt"].dt.to_period("M") == raw["date"].dt.to_period("M")
    )

    return raw


# -----------------------------------------------------------------------------
# Load and normalize hand-labeled fallback observations
# -----------------------------------------------------------------------------


def load_hand_labeled_observations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=AUDIT_COLUMNS + ["confidence_rank", "is_direct_report"]
        )

    hand = pd.read_csv(path)
    hand = ensure_columns(
        hand, ["date", "year", "month", "metric", "value", "unit", "region", "source"]
    )

    hand["date"] = hand["date"].apply(parse_date)
    hand = hand[hand["date"].notna()].copy()

    hand["year"] = hand["date"].dt.year.astype(int)
    hand["month"] = hand["date"].dt.month.apply(month_name)
    hand["metric"] = hand["metric"].apply(normalize_metric)
    hand["region"] = hand["region"].apply(normalize_region)
    hand["value"] = hand["value"].apply(safe_numeric)
    hand["unit"] = hand["unit"].fillna("kg")
    hand["source"] = hand["source"].fillna("")
    hand["source_type"] = "hand_labeled"

    # Manual data is only all-sector total in your provided CSV.
    hand = hand[hand["region"].eq("all")]
    hand = hand[hand["value"].notna()].copy()

    # Audit compatibility fields.
    hand["source_priority"] = 99
    hand["confidence"] = "manual_fallback"
    hand["confidence_rank"] = 0
    hand["period_relation"] = "manual_fallback"
    hand["report_id"] = pd.NA
    hand["report_period_start"] = pd.NA
    hand["observed_period_start"] = hand["date"].dt.strftime("%Y-%m-%d")
    hand["period_label"] = hand["date"].dt.strftime("%b_%Y").str.lower()
    hand["extraction_method"] = "hand_labeled_fallback"
    hand["notes"] = "used only when extracted value missing"
    hand["is_direct_report"] = False

    return hand


# -----------------------------------------------------------------------------
# Select best observations
# -----------------------------------------------------------------------------


def choose_best_extracted(extracted: pd.DataFrame) -> pd.DataFrame:
    if extracted.empty:
        return extracted.copy()

    df = extracted.copy()

    # Sort so the first row per key is the preferred observation.
    # Preference order:
    #   1. direct report for that same month
    #   2. lower source_priority
    #   3. higher confidence
    #   4. non-null source
    #   5. later report period, useful for revised/backfilled values if priority ties
    df["has_source"] = df["source"].astype("string").fillna("").ne("")

    df = df.sort_values(
        by=[
            "date",
            "metric",
            "region",
            "unit",
            "is_direct_report",
            "source_priority",
            "confidence_rank",
            "has_source",
            "report_period_start_dt",
        ],
        ascending=[True, True, True, True, False, True, False, False, False],
        na_position="last",
    )

    best = df.drop_duplicates(
        subset=["date", "metric", "region", "unit"],
        keep="first",
    ).copy()

    return best


def add_hand_labeled_gaps(
    best_extracted: pd.DataFrame, hand: pd.DataFrame
) -> pd.DataFrame:
    if hand.empty:
        return best_extracted.copy()

    if best_extracted.empty:
        combined = hand.copy()
        return combined

    existing_keys = set(
        zip(
            best_extracted["date"].dt.strftime("%Y-%m-%d"),
            best_extracted["metric"],
            best_extracted["region"],
            best_extracted["unit"],
        )
    )

    hand = hand.copy()
    hand_keys = list(
        zip(
            hand["date"].dt.strftime("%Y-%m-%d"),
            hand["metric"],
            hand["region"],
            hand["unit"],
        )
    )

    hand_to_add = hand[[key not in existing_keys for key in hand_keys]].copy()

    return pd.concat([best_extracted, hand_to_add], ignore_index=True, sort=False)


def finalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.apply(month_name)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Keep kg production as whole numbers where possible.
    if "unit" in df.columns:
        kg_mask = df["unit"].astype("string").str.lower().eq("kg") & df["value"].notna()
        df.loc[kg_mask, "value"] = df.loc[kg_mask, "value"].round().astype("Int64")

    for col in AUDIT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    audit = df[AUDIT_COLUMNS].sort_values(["date", "region"]).reset_index(drop=True)
    minimal = audit[FINAL_COLUMNS].copy()

    # Write date as YYYY-MM-DD for consistency.
    audit["date"] = pd.to_datetime(audit["date"]).dt.strftime("%Y-%m-%d")
    minimal["date"] = pd.to_datetime(minimal["date"]).dt.strftime("%Y-%m-%d")

    return minimal, audit


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    extracted = load_extracted_observations(args.raw, include_ytd=args.include_ytd)
    hand = load_hand_labeled_observations(args.hand)

    best_extracted = choose_best_extracted(extracted)
    combined = add_hand_labeled_gaps(best_extracted, hand)
    minimal, audit = finalize_columns(combined)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)

    minimal.to_csv(args.out, index=False)
    audit.to_csv(args.audit_out, index=False)

    extracted_count = len(best_extracted)
    hand_count = (
        int((audit["source_type"] == "hand_labeled").sum()) if not audit.empty else 0
    )

    print(f"✓ wrote minimal best dataset: {args.out} ({len(minimal)} rows)")
    print(f"✓ wrote audit dataset:        {args.audit_out} ({len(audit)} rows)")
    print(f"  extracted best rows: {extracted_count}")
    print(f"  hand-labeled fallback rows added: {hand_count}")

    if not minimal.empty:
        print()
        print(minimal.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
