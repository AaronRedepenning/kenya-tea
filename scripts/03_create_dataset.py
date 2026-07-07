import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import camelot
import numpy as np
import pandas as pd
import yaml

# =============================================================================
# CONFIG
# =============================================================================


def get_config() -> dict[str, Any]:
    with Path("config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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

OBSERVATION_COLUMNS = [
    "observation_id",
    "report_id",
    "report_year",
    "report_month",
    "report_period_start",
    "report_period_end",
    "observed_year",
    "observed_month",
    "observed_period_start",
    "observed_period_end",
    "period_label",
    "period_relation",
    "source_priority",
    "metric",
    "sector",
    "sector_raw",
    "region",
    "region_raw",
    "value",
    "unit",
    "source_file",
    "source_page",
    "source_table_index",
    "source_row_index",
    "extraction_method",
    "confidence",
    "scale_factor",
    "notes",
]

REPORT_COLUMNS = [
    "report_id",
    "report_year",
    "report_month",
    "report_period_start",
    "report_period_end",
    "source_file",
    "source_path",
    "extracted_at",
]


# =============================================================================
# BASIC CLEANING
# =============================================================================


def normalize_col_name(col: object) -> str:
    col = str(col).strip().lower()
    col = col.replace("\r", " ").replace("\n", " ")
    col = re.sub(r"\s+", " ", col)

    # Normalize month/year labels before generic replacement:
    #   Feb.-22 -> feb_22
    #   Feb-22  -> feb_22
    #   April 2022 -> april_2022
    #   Jan - April 2022 -> jan_to_april_2022
    col = re.sub(
        r"\b([a-z]{3,9})\.?\s*[-–—]\s*(\d{2,4})\b",
        r"\1_\2",
        col,
    )
    col = re.sub(
        r"\b([a-z]{3,9})\.?\s+(\d{2,4})\b",
        r"\1_\2",
        col,
    )
    col = re.sub(
        r"\b([a-z]{3,9})\s*[-–—]\s*([a-z]{3,9})\s+(\d{2,4})\b",
        r"\1_to_\2_\3",
        col,
    )

    replacements = {
        "+/-": "variance",
        "variance +/-": "variance",
        "var.(%)": "variance_pct",
        "var. (%)": "variance_pct",
        "var (%)": "variance_pct",
        "%": "pct",
        " - ": "_to_",
        "-": "_",
        " ": "_",
        "sub_sector": "sector",
        "subsector": "sector",
        "block": "region",
        "sub-sector/producer_block": "sector",
        "sector/factories": "sector",
        "sector/produ_cers": "sector",
        "sectorifactory": "sector",
        "sector_factory": "sector",
        "sector_region": "sector",
        "producer_region": "region",
        "producerregion": "region",
    }

    for old, new in replacements.items():
        col = col.replace(old, new)

    col = re.sub(r"[^a-z0-9_]+", "", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def dedupe_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output = []

    for col in columns:
        col = col or "blank"
        counts[col] = counts.get(col, 0) + 1
        output.append(col if counts[col] == 1 else f"{col}_{counts[col]}")

    return output


def clean_string_cell(x: object) -> object:
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        x = x.replace("\r", " ").replace("\n", " ")
        x = x.replace("|", " ")
        x = re.sub(r"\s+", " ", x).strip()

        if x in {"", "-", "–", "—"}:
            return np.nan

    return x


def clean_numeric_cell(x: object) -> float:
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    if x in {"", "-", "–", "—"}:
        return np.nan

    # Normalize common OCR/PDF artifacts before parsing:
    #   "17,886,857]" -> "17886857"
    #   "20,385,428!" -> "20385428"
    #   "11 905,641"  -> "11905641"
    #   "- 2,330,338" -> "-2330338"
    x = x.replace("−", "-").replace("–", "-").replace("—", "-")
    x = re.sub(r"^[_=]+\s*", "", x)
    x = re.sub(r"-\s+", "-", x)

    is_parentheses_negative = bool(re.match(r"^\(.*\)$", x))

    # Keep only numeric syntax characters. This strips OCR junk like ], !, A, _.
    x = re.sub(r"[^0-9,%.()\-]", "", x)
    x = x.replace(",", "")
    x = x.replace("%", "")
    x = x.replace("(", "")
    x = x.replace(")", "")

    value = pd.to_numeric(x, errors="coerce")

    if is_parentheses_negative and pd.notna(value):
        value = -abs(value)

    return value


def looks_numeric(series: pd.Series, threshold: float = 0.7) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    converted = non_null.apply(clean_numeric_cell)
    return converted.notna().mean() >= threshold


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = text.replace("|", " ")
    text = re.sub(r"[^a-z0-9&]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# FUZZY NORMALIZATION
# =============================================================================


def fuzzy_score(text: str, pattern: str) -> int:
    if not text or not pattern:
        return 0

    direct = SequenceMatcher(None, text, pattern).ratio()
    shorter, longer = sorted([text, pattern], key=len)

    if len(shorter) == 0:
        partial = 0
    elif len(shorter) == len(longer):
        partial = direct
    else:
        partial = max(
            SequenceMatcher(None, shorter, longer[i : i + len(shorter)]).ratio()
            for i in range(len(longer) - len(shorter) + 1)
        )

    return round(max(direct, partial) * 100)


REGION_PATTERNS = {
    "west_of_rift": [
        "west of rift",
        "west rift",
        "westorrin",
        "west orrin",
        "west of rin",
        "west of rit",
        "west or rin",
        "nest anus",  # observed parser corruption
    ],
    "east_of_rift": [
        "east of rift",
        "east rift",
        "eastorrin",
        "eastorrint",
        "cast of rit",
        "east of rit",
        "east of rin",
    ],
    "total": [
        "total",
        "grand total",
        "taal",
        "rota",
        "rasta",
    ],
}

SECTOR_PATTERNS = {
    "plantation": [
        "plantation",
        "plantations",
        "plantation estates",
        "estates",
    ],
    "smallholder": [
        "smallholder",
        "smallholders",
        "smallholde",
        "ktda",
        "kida",
        "ktda managed factories",
        "ktda managed smallholder factories",
        "kida managed maliholaer factories",
        "kida managed smallholder factories",
    ],
    "independent": [
        "independent",
        "independents",
    ],
    "nyayo_tea_zones": [
        "nyayo tea zones",
        "nyayo t zones",
        "nyayo teazones",
        "nyayo tea zones",
    ],
    "all": [
        "plantation & smallholder",
        "plantation and smallholder",
        "plantation smallholder",
        "estates smallholder independents nyayo tea zones",
        "estates smallholders independents nyayo tea zones",
        "plantation smallholder independents nyayo tea zones",
        "total",
        "grand total",
        "rota",
    ],
}


def fuzzy_lookup(value: object, patterns: dict[str, list[str]], min_score: int) -> str:
    text = normalize_text(value)
    if not text:
        return "unknown"

    best_label = "unknown"
    best_score = 0

    for label, label_patterns in patterns.items():
        for pattern in label_patterns:
            score = fuzzy_score(text, pattern)
            if score > best_score:
                best_score = score
                best_label = label

    return best_label if best_score >= min_score else "unknown"


def normalize_region(value: object) -> str:
    text = normalize_text(value)

    if not text:
        return "unknown"

    if "west" in text and ("rift" in text or "rit" in text):
        return "west_of_rift"

    if (
        "east" in text
        or "easto" in text
        or text in {"east i", "east of i", "east of i)"}
    ) and ("rift" in text or "rit" in text or "i" in text):
        return "east_of_rift"

    total_like = {
        "total",
        "tota",
        "tae",
        "atu",
        "oad total",
        "grand total",
    }

    if "total" in text or text in total_like:
        return "total"

    return "unknown"


def normalize_sector(value: object) -> str:
    text = normalize_text(value)

    # Common mangled versions of Total / grand-total sector rows.
    # Check these FIRST so labels like "Baton" do not get misclassified.
    total_like_sector_labels = {
        "total",
        "grand total",
        "baton",
        "bation",
        "botan",
        "tota",
        "rota",
    }
    if text in total_like_sector_labels:
        return "all"

    has_plantation = "plantation" in text or "estate" in text or "estates" in text
    has_smallholder = any(
        token in text
        for token in [
            "smallholder",
            "smallholders",
            "smallholde",
            "small",
            "ktda",
            "kida",
        ]
    )
    has_independent = "independent" in text or "independents" in text
    has_nyayo = "nyayo" in text

    # Combined/grand-total sector labels.
    if (
        (has_plantation and has_smallholder)
        or (has_smallholder and has_independent)
        or (has_plantation and has_independent)
        or (has_plantation and has_nyayo)
        or (has_smallholder and has_nyayo)
        or (has_independent and has_nyayo)
        or "plantation smallholder" in text
        or "estates smallholder" in text
        or "estates smallholders" in text
    ):
        return "all"

    if has_plantation:
        return "plantation"
    if has_smallholder:
        return "smallholder"
    if has_independent:
        return "independent"
    if has_nyayo:
        return "nyayo_tea_zones"

    return fuzzy_lookup(value, SECTOR_PATTERNS, min_score=70)


# =============================================================================
# TABLE REPAIRS
# =============================================================================

MONTH_NAMES = set(MONTH_LOOKUP.keys())


def is_period_value_col(col: object) -> bool:
    """
    Match production period value columns like:
      jan_24, jan_2024, january_2024, april_2022, sept_2023,
      and year-only columns like 2025.
    """
    col_text = str(col).lower().strip()
    col_text = re.sub(r"_+", "_", col_text)

    # Year-only columns: 2025
    if re.match(r"^\d{4}$", col_text):
        return True

    # Single month columns: april_2022, apr_22, sept_2023
    match = re.match(r"^([a-z]+)_(?:\d{2}|\d{4})$", col_text)
    if match:
        return match.group(1) in MONTH_NAMES

    return False


def get_period_value_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_period_value_col(c)]


def drop_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated table header rows that survived Camelot extraction/concat."""
    if df.empty:
        return df

    header_tokens = {
        "sector",
        "sub sector",
        "subsector",
        "region",
        "variance",
        "variance variance",
        "var",
        "var pct",
        "var percentage",
    }

    def is_header_like_row(row: pd.Series) -> bool:
        values = [
            normalize_text(v).replace("_", " ") for v in row.tolist() if pd.notna(v)
        ]
        if not values:
            return False

        hits = 0
        for value in values:
            if value in header_tokens or any(token == value for token in header_tokens):
                hits += 1
            elif value in MONTH_LOOKUP or is_period_value_col(
                normalize_col_name(value)
            ):
                hits += 1

        return hits >= 2

    mask = df.apply(is_header_like_row, axis=1)
    return df.loc[~mask].reset_index(drop=True)


MONTH_NAMES_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)


def repair_raw_header_cells(columns: list[object]) -> list[object]:
    """
    Fix Camelot header cells where a period label and 'Variance +/-'
    got merged into one column name.

    Example:
      'July 2021 Variance +/-', nan
    becomes:
      'July 2021', 'Variance +/-'
    """
    repaired = list(columns)

    for i in range(len(repaired) - 1):
        current = "" if pd.isna(repaired[i]) else str(repaired[i]).strip()
        nxt = "" if pd.isna(repaired[i + 1]) else str(repaired[i + 1]).strip()

        pattern = rf"^({MONTH_NAMES_PATTERN})\s*[-]?\s*(\d{{2,4}})\s+variance\s*\+/-$"

        if re.match(pattern, current, flags=re.IGNORECASE) and not nxt:
            match = re.match(pattern, current, flags=re.IGNORECASE)
            repaired[i] = f"{match.group(1)} {match.group(2)}"
            repaired[i + 1] = "Variance +/-"

    return repaired


def repair_number_embedded_in_region(
    df: pd.DataFrame,
    region_col: str = "region",
) -> pd.DataFrame:
    """
    Fix rows where the first period value was pushed into the region cell.

    Example:
      region = "West of Rift 35,136,947", july_2023 = NaN
    becomes:
      region = "West of Rift", july_2023 = 35,136,947

      region = "East 9,559,668", july_2023 = NaN
    becomes:
      region = "East", july_2023 = 9,559,668
    """
    df = df.copy()

    if region_col not in df.columns:
        return df

    value_cols = get_period_value_columns(df)
    if not value_cols:
        return df

    first_value_col = value_cols[0]

    pattern = r"^(?P<label>.*?[A-Za-z][A-Za-z\s]+?)\s+(?P<number>-?\d[\d,\s]*\d)$"

    region_text = df[region_col].astype("string")

    extracted = region_text.str.extract(pattern)

    has_embedded_number = (
        extracted["label"].notna()
        & extracted["number"].notna()
        & df[first_value_col].isna()
    )

    if not has_embedded_number.any():
        return df

    df.loc[has_embedded_number, region_col] = (
        extracted.loc[has_embedded_number, "label"].astype("string").str.strip()
    )

    df.loc[has_embedded_number, first_value_col] = (
        extracted.loc[has_embedded_number, "number"]
        .astype("string")
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", "", regex=False)
    )

    return df


def repair_embedded_number_percent_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles cells where a number and percent got pushed into one cell, e.g.
    "3,463,082 -20.71%". This intentionally does not try to repair cells
    containing multiple plain numbers like "10,094,658 9,563,090".
    """
    df = df.copy()
    mask_pattern = r"^\s*[+-]?[\d,]+\s+-?\d+(?:\.\d+)?%\s*$"
    extract_pattern = r"^\s*([+-]?[\d,]+)\s+(-?\d+(?:\.\d+)?%)\s*$"

    for col_idx in range(1, len(df.columns)):
        col = df.columns[col_idx]
        prev_col = df.columns[col_idx - 1]

        text = df[col].astype("string")
        mask = text.str.contains(mask_pattern, regex=True, na=False)

        if not mask.any():
            continue

        extracted = text[mask].str.extract(extract_pattern)
        extracted_number = extracted[0]
        extracted_pct = extracted[1]

        previous_was_dash = (
            df.loc[mask, prev_col]
            .astype("string")
            .str.strip()
            .isin(["-", "–", "—", "_", "_ -", "-", "="])
        )

        dash_idx = previous_was_dash[previous_was_dash].index
        non_dash_idx = previous_was_dash[~previous_was_dash].index

        if len(dash_idx):
            df.loc[dash_idx, prev_col] = "-" + extracted_number.loc[dash_idx].astype(
                str
            )
        if len(non_dash_idx):
            df.loc[non_dash_idx, prev_col] = extracted_number.loc[non_dash_idx]

        df.loc[mask, col] = extracted_pct

    return df


def repair_region_order_for_three_row_groups(
    df: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Repair region_norm in 3-row groups that follow:
      West of Rift
      East of Rift
      Total

    Only repairs if at least one period column validates:
      row0 + row1 ≈ row2
    """
    df = df.copy()

    if "region_norm" not in df.columns or not value_cols:
        return df

    for start in range(0, len(df) - 2):
        idx = [start, start + 1, start + 2]
        region_values = df.loc[idx, "region_norm"].tolist()

        known_pattern = region_values in [
            ["unknown", "east_of_rift", "total"],
            ["west_of_rift", "unknown", "total"],
            ["west_of_rift", "east_of_rift", "unknown"],
        ]

        if not known_pattern:
            continue

        group = df.loc[idx]

        valid_math = False

        for col in value_cols:
            west = pd.to_numeric(group.iloc[0][col], errors="coerce")
            east = pd.to_numeric(group.iloc[1][col], errors="coerce")
            total = pd.to_numeric(group.iloc[2][col], errors="coerce")

            if values_add_up(west, east, total):
                valid_math = True
                break

        if not valid_math:
            continue

        if region_values == ["unknown", "east_of_rift", "total"]:
            df.loc[start, "region_norm"] = "west_of_rift"
            df.loc[start, "region"] = "West of Rift"

        elif region_values == ["west_of_rift", "unknown", "total"]:
            df.loc[start + 1, "region_norm"] = "east_of_rift"
            df.loc[start + 1, "region"] = "East of Rift"

        elif region_values == ["west_of_rift", "east_of_rift", "unknown"]:
            df.loc[start + 2, "region_norm"] = "total"
            df.loc[start + 2, "region"] = "Total"

    return df


def repair_region_embedded_in_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles rows where Camelot merged sector and region into one column.

    Examples:
      "Plantation & West of Rift"       -> sector keeps text, region = West of Rift
      "Smallholder East of Rift"        -> region = East of Rift
      "Total"                           -> region = Total when nearby group context validates
      "West of Rift Smallholder East..." -> can still be handled by row-order fallback
    """
    df = df.copy()

    df = ensure_text_column(df, "sector")
    df = ensure_text_column(df, "region")

    sector_text = df["sector"].apply(normalize_text)
    region_text = df["region"].apply(normalize_text)

    region_missing = region_text.eq("") | region_text.eq("nan") | region_text.isna()

    df.loc[
        region_missing & sector_text.str.contains(r"\bwest\b", regex=True),
        "region",
    ] = "West of Rift"

    df.loc[
        region_missing & sector_text.str.contains(r"\beast\b", regex=True),
        "region",
    ] = "East of Rift"

    df.loc[
        region_missing & sector_text.str.contains(r"\btotal\b", regex=True),
        "region",
    ] = "Total"

    return df


def repair_all_sector_from_three_row_group(
    df: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """
    If a 3-row group is region-ordered west/east/total and validates numerically,
    mark the whole group as sector_norm='all' when the text indicates a combined
    plantation + smallholder total split across rows.

    Handles:
      row0: "Plantation & West of Rift"
      row1: "Smallholder East of Rift"
      row2: "Total"
    """
    df = df.copy()

    if not value_cols or not {"sector", "region_norm", "sector_norm"}.issubset(
        df.columns
    ):
        return df

    for start in range(0, len(df) - 2):
        idx = [start, start + 1, start + 2]
        group = df.loc[idx]

        region_pattern = group["region_norm"].tolist()
        if region_pattern != ["west_of_rift", "east_of_rift", "total"]:
            continue

        valid_math = False
        for col in value_cols:
            west = pd.to_numeric(group.iloc[0][col], errors="coerce")
            east = pd.to_numeric(group.iloc[1][col], errors="coerce")
            total = pd.to_numeric(group.iloc[2][col], errors="coerce")

            if values_add_up(west, east, total):
                valid_math = True
                break

        if not valid_math:
            continue

        combined_text = " ".join(group["sector"].astype("string").fillna("").tolist())
        combined_norm = normalize_text(combined_text)

        has_plantation = "plantation" in combined_norm or "estate" in combined_norm
        has_smallholder = (
            "smallholder" in combined_norm or "smallholde" in combined_norm
        )

        if has_plantation and has_smallholder:
            df.loc[idx, "sector_norm"] = "all"

    return df


def ensure_text_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Ensure a column exists and can safely receive string labels."""
    df = df.copy()
    if col not in df.columns:
        df[col] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df[col] = df[col].astype("string")
    return df


def extract_number_like_tokens(value: object) -> list[str]:
    """
    Extract large number-like tokens from a cell while preserving OCR spaces inside numbers.

    Examples:
      "54 365,267 58,966,889" -> ["54 365,267", "58,966,889"]
      "17,886,857]" -> ["17,886,857"]
      "11 905,641" -> ["11 905,641"]
    """
    if pd.isna(value):
        return []

    text = str(value)
    pattern = r"(?<!\d)(?:\d{1,3}(?:[,\s]\d{3})+|\d+)(?!\d)"
    return re.findall(pattern, text)


def repair_shifted_period_pair_cells(
    df: pd.DataFrame, value_cols: list[str]
) -> pd.DataFrame:
    """
    Repair rows where the current-period value was pushed into the next period column.

    Example from Jan 2025 report:
      jan_2025 = NaN
      jan_2024 = "54 365,267 58,966,889"

    becomes:
      jan_2025 = "54 365,267"
      jan_2024 = "58,966,889"

    This only runs when the left/previous period column is blank, so it should not
    disturb normal cells like "15 799 147" already sitting in the right column.
    """
    if len(value_cols) < 2:
        return df

    df = df.copy()
    ordered_value_cols = [c for c in df.columns if c in value_cols]

    for left_col, right_col in zip(ordered_value_cols, ordered_value_cols[1:]):
        left_blank = df[left_col].isna() | df[left_col].astype(
            "string"
        ).str.strip().isin(["", "<NA>"])

        for idx in df.index[left_blank]:
            tokens = extract_number_like_tokens(df.at[idx, right_col])
            if len(tokens) < 2:
                continue

            # Use the last two tokens. In these shifted cells, they are the two
            # period values; earlier tokens, if any, are usually OCR noise.
            df.at[idx, left_col] = tokens[-2]
            df.at[idx, right_col] = tokens[-1]

    return df


def repair_split_category_labels(
    df: pd.DataFrame, category_col: str = "sector"
) -> pd.DataFrame:
    """
    Repairs category labels split across adjacent rows.
    Example: "Plantation &" followed by "Smallholder".
    """
    if category_col not in df.columns:
        return df

    df = df.copy()
    s = df[category_col].astype("string")

    for i in range(len(df) - 1):
        current = s.iloc[i]
        nxt = s.iloc[i + 1]

        if pd.notna(current) and str(current).strip().endswith("&"):
            if pd.notna(nxt) and str(nxt).strip():
                combined = f"{str(current).strip()} {str(nxt).strip()}"
                df.loc[i, category_col] = combined
                df.loc[i + 1, category_col] = combined

    return df


def find_candidate_sector_column(df: pd.DataFrame) -> str | None:
    preferred = [
        "sector",
        "sectorproducer_region",
        "sectorproducerregion",
        "sector_region",
        "producer_region",
        "producerregion",
    ]

    for col in preferred:
        if col in df.columns:
            return col

    # Fallback: choose the first mostly-text column that is not region and not a period value column.
    for col in df.columns:
        if col == "region" or is_period_value_col(col):
            continue
        if not looks_numeric(df[col], threshold=0.5):
            return col

    return None


def values_add_up(
    west: float,
    east: float,
    total: float,
    tolerance_ratio: float = 0.0005,
    absolute_tolerance: float = 1_000,
) -> bool:
    if pd.isna(west) or pd.isna(east) or pd.isna(total):
        return False

    expected = west + east
    diff = abs(expected - total)

    tolerance = max(absolute_tolerance, abs(total) * tolerance_ratio)

    return diff <= tolerance


def recover_region_from_three_row_groups(
    df: pd.DataFrame,
    value_cols: list[str],
    sector_col: str | None = None,
    tolerance_ratio: float = 0.01,
) -> pd.DataFrame:
    """
    Recovery for tables where the region column is missing/merged, but rows follow:
      West of Rift
      East of Rift
      Total
    Repeats by sector, and validates using West + East ~= Total.
    """
    if not value_cols:
        return df

    df = df.copy()
    sector_col = sector_col or find_candidate_sector_column(df)

    # These may have been inferred as float columns if the parser produced only blanks.
    # Force string dtype before assigning labels like "West of Rift".
    df = ensure_text_column(df, "sector")
    df = ensure_text_column(df, "region")

    # Only use this fallback when region is missing or almost entirely unknown.
    existing_region_norm = df["region"].apply(normalize_region)
    known_region_rate = existing_region_norm.ne("unknown").mean() if len(df) else 0
    if known_region_rate > 0.25:
        return df

    for start in range(0, len(df) - 2):
        group_idx = [start, start + 1, start + 2]
        group = df.loc[group_idx]

        valid_group = False
        for col in value_cols:
            west = pd.to_numeric(group.iloc[0][col], errors="coerce")
            east = pd.to_numeric(group.iloc[1][col], errors="coerce")
            total = pd.to_numeric(group.iloc[2][col], errors="coerce")

            if values_add_up(west, east, total, tolerance_ratio=tolerance_ratio):
                valid_group = True
                break

        if not valid_group:
            continue

        sector_label = group.iloc[0].get(sector_col, pd.NA) if sector_col else pd.NA

        df.loc[start, "sector"] = sector_label
        df.loc[start + 1, "sector"] = sector_label
        df.loc[start + 2, "sector"] = sector_label

        df.loc[start, "region"] = "West of Rift"
        df.loc[start + 1, "region"] = "East of Rift"
        df.loc[start + 2, "region"] = "Total"

    return df


def apply_contextual_sector_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Context-specific sector fixes for known PDF parse/OCR corruptions."""
    if not {"sector", "sector_norm"}.issubset(df.columns):
        return df

    df = df.copy()
    sector_text = df["sector"].apply(normalize_text)

    # Observed corruption: "Baton" is a grand-total/all-sector label.
    df.loc[
        sector_text.isin(["baton", "bation", "botan", "rota", "tota"]), "sector_norm"
    ] = "all"

    return df


def infer_scale_factor(values: list[object]) -> int:
    """
    Handles parsed million-kg values like 35.87796 instead of 35,877,960.
    If all values in a west/east/total triple are tiny for kg, scale to kg.
    """
    clean_values = [abs(float(v)) for v in values if pd.notna(v)]
    if not clean_values:
        return 1

    max_value = max(clean_values)
    if 0 < max_value < 1_000:
        return 1_000_000

    return 1


def force_first_two_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force the first two columns to be sector and region.

    This handles messy headers like:
      Sectorifactory Block | NaN | May-25 | ...
      Sub-Sector/Producer Block | NaN | Jan. 2024 | ...
    """
    df = df.copy()

    columns = list(df.columns)

    if len(columns) >= 1:
        columns[0] = "sector"

    if len(columns) >= 2:
        columns[1] = "region"

    df.columns = columns
    return df


def clean_report_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unnamed / blank index columns from CSV imports or parser artifacts.
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed|^,$|^\s*$")]

    # Normalize headers.
    raw_columns = repair_raw_header_cells(list(df.columns))
    df.columns = [normalize_col_name(c) for c in raw_columns]
    df.columns = dedupe_columns(df.columns)

    df = force_first_two_columns(df)

    # Clean all string cells.
    df = df.map(clean_string_cell)
    df = df.dropna(how="all").reset_index(drop=True)
    df = drop_repeated_header_rows(df)

    if df.empty:
        return df

    df = repair_number_embedded_in_region(df, "region")

    # Repair malformed cells before type conversion.
    df = repair_embedded_number_percent_cells(df)

    value_cols = get_period_value_columns(df)
    df = repair_shifted_period_pair_cells(df, value_cols=value_cols)

    # Make expected text columns exist and string-safe before any recovery logic assigns labels.
    df = ensure_text_column(df, "sector")
    df = ensure_text_column(df, "region")

    # Convert numeric-looking columns dynamically.
    for col in df.columns:
        if col in {"sector", "region"}:
            continue
        if looks_numeric(df[col]):
            df[col] = df[col].apply(clean_numeric_cell)

    value_cols = get_period_value_columns(df)

    # If sector/region are merged or missing, recover by validated 3-row groups.
    df = recover_region_from_three_row_groups(df, value_cols=value_cols)

    # Repair split category labels and forward-fill sector within normal tables.
    df = repair_split_category_labels(df, "sector")
    df["sector"] = df["sector"].ffill()

    # Normalize common text columns.
    for col in ["sector", "region"]:
        df[col] = (
            df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        )

    # Drop repeated header rows that survived concat.
    repeated_header_mask = pd.Series(False, index=df.index)
    for col in ["sector", "region"]:
        repeated_header_mask |= df[col].astype("string").str.lower().eq(col)
    df = df.loc[~repeated_header_mask].reset_index(drop=True)

    df = repair_region_embedded_in_sector(df)

    # Normalize dimensions.
    df["sector_norm"] = df["sector"].apply(normalize_sector)
    df["region_norm"] = df["region"].apply(normalize_region)

    df = apply_contextual_sector_fixes(df)

    value_cols = get_period_value_columns(df)
    df = repair_region_order_for_three_row_groups(df, value_cols)
    df = repair_all_sector_from_three_row_group(df, value_cols)
    df = keep_only_best_all_sector_group(df, value_cols)

    return df


# =============================================================================
# CAMELOT TABLE READING
# =============================================================================


def normalized_row_values(row: pd.Series) -> list[str]:
    return [normalize_col_name(x) for x in row.tolist()]


def row_looks_like_header(row: pd.Series) -> bool:
    values = normalized_row_values(row)
    has_period = any(is_period_value_col(v) for v in values)
    has_dimension = any(
        v in {"sector", "region", "block", "sub_sector"} or "sector" in v
        for v in values
    )
    return has_period or has_dimension


def apply_header_to_raw_table(
    raw: pd.DataFrame, previous_header: list[object] | None
) -> tuple[pd.DataFrame | None, list[object] | None]:
    """
    Turns a raw Camelot table into a DataFrame with usable headers.

    Split-table behavior:
      - First table with a detectable header: row 0 becomes the header; rows 1+ are data.
      - Later continuation tables with the same number of columns and no header: reuse the
        previous header and keep *all* rows as data.
      - If a later table repeats the header anyway, drop only that repeated header row.
    """
    if raw.empty:
        return None, previous_header

    raw = raw.copy()
    raw = raw.map(clean_string_cell).dropna(how="all").reset_index(drop=True)
    if raw.empty:
        return None, previous_header

    first_row_is_header = row_looks_like_header(raw.iloc[0])

    if first_row_is_header:
        header = list(raw.iloc[0])
        body = raw.iloc[1:].reset_index(drop=True)
    elif previous_header is not None and raw.shape[1] == len(previous_header):
        header = previous_header
        body = raw.reset_index(drop=True)
    else:
        # Cannot confidently identify headers.
        return None, previous_header

    body.columns = header

    # Drop repeated header row in continuation tables.
    if not body.empty:
        first_norm = normalized_row_values(body.iloc[0])
        header_norm = [normalize_col_name(h) for h in header]
        if first_norm == header_norm:
            body = body.iloc[1:].reset_index(drop=True)

    return body, header


def read_camelot_table_groups(
    filepath: Path, flavor: str = "lattice"
) -> list[pd.DataFrame]:
    """
    Read all Camelot tables and group adjacent tables that share the same headers.
    This supports tables split across pages while still allowing unrelated tables to be ignored later.
    """
    tables = camelot.read_pdf(str(filepath), flavor=flavor, pages="all")

    frames: list[pd.DataFrame] = []
    previous_header: list[object] | None = None

    for table_index, table in enumerate(tables):
        frame, previous_header = apply_header_to_raw_table(table.df, previous_header)
        if frame is None or frame.empty:
            continue

        frame = frame.copy()
        frame["__source_page"] = getattr(table, "page", None)
        frame["__source_table_index"] = table_index
        frames.append(frame)

    if not frames:
        return []

    groups: list[pd.DataFrame] = []
    current_group: list[pd.DataFrame] = []
    current_cols: list[str] | None = None

    for frame in frames:
        comparable_cols = [
            normalize_col_name(c)
            for c in frame.columns
            if not str(c).startswith("__source_")
        ]

        if current_cols is None or comparable_cols == current_cols:
            current_group.append(frame)
            current_cols = comparable_cols
        else:
            groups.append(pd.concat(current_group, ignore_index=True))
            current_group = [frame]
            current_cols = comparable_cols

    if current_group:
        groups.append(pd.concat(current_group, ignore_index=True))

    return groups


def read_report_table_groups(filepath: Path) -> list[pd.DataFrame]:
    """Try lattice first, then stream if lattice finds nothing usable."""
    groups = read_camelot_table_groups(filepath, flavor="lattice")
    if groups:
        return groups

    return read_camelot_table_groups(filepath, flavor="stream")


# =============================================================================
# PERIOD / REPORT METADATA
# =============================================================================


def extract_year_month_from_report_path(filepath: Path) -> tuple[int, str]:
    parts = filepath.stem.split("-")
    year, month_name = int(parts[-2]), str(parts[-1])
    return year, month_name


def month_name_to_number(month_name: str) -> int:
    key = month_name.strip().lower()[:3]
    if key not in MONTH_LOOKUP:
        raise ValueError(f"Could not parse month name: {month_name}")
    return MONTH_LOOKUP[key]


def month_period_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    return start, end


def normalize_year_text(year_text: str) -> int:
    if len(year_text) == 4:
        return int(year_text)

    yy = int(year_text)
    # These reports are modern. Interpret 00-79 as 2000s, 80-99 as 1900s.
    return 2000 + yy if yy <= 79 else 1900 + yy


def parse_period_info(period_label: str, report_year: int) -> dict[str, Any]:
    """
    Parse period labels into observed period bounds.

    Supports:
      feb_22              -> Feb 2022
      february_2022       -> Feb 2022
      jan_april_2022      -> Jan-Apr 2022
      jan_to_april_2022   -> Jan-Apr 2022
      2025                -> calendar year 2025
    """
    label = str(period_label).lower().strip()
    label = re.sub(r"_+", "_", label)

    if re.match(r"^\d{4}$", label):
        year = int(label)
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
        return {
            "observed_year": year,
            "observed_month": None,
            "observed_period_start": start.date().isoformat(),
            "observed_period_end": end.date().isoformat(),
            "period_type": "annual",
        }

    range_match = re.match(r"^([a-z]+)(?:_to)?_([a-z]+)_(\d{2}|\d{4})$", label)
    if range_match:
        start_month_text, end_month_text, year_text = range_match.groups()
        start_month = MONTH_LOOKUP.get(start_month_text)
        end_month = MONTH_LOOKUP.get(end_month_text)
        if start_month and end_month:
            year = normalize_year_text(year_text)
            start = pd.Timestamp(year=year, month=start_month, day=1)
            end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(
                0
            )
            return {
                "observed_year": year,
                "observed_month": end_month,
                "observed_period_start": start.date().isoformat(),
                "observed_period_end": end.date().isoformat(),
                "period_type": "year_to_date" if start_month == 1 else "period_range",
            }

    month_match = re.match(r"^([a-z]+)_(\d{2}|\d{4})$", label)
    if month_match:
        month_text, year_text = month_match.groups()
        month = MONTH_LOOKUP.get(month_text)
        if month:
            year = normalize_year_text(year_text)
            start, end = month_period_bounds(year, month)
            return {
                "observed_year": year,
                "observed_month": month,
                "observed_period_start": start.date().isoformat(),
                "observed_period_end": end.date().isoformat(),
                "period_type": "monthly",
            }

    return {
        "observed_year": None,
        "observed_month": None,
        "observed_period_start": None,
        "observed_period_end": None,
        "period_type": "unknown",
    }


def classify_period_relation(
    report_year: int,
    report_month_num: int,
    period_info: dict[str, Any],
) -> tuple[str, int]:
    observed_year = period_info.get("observed_year")
    observed_month = period_info.get("observed_month")
    period_type = period_info.get("period_type")

    if observed_year is None:
        return "unknown", 9

    if period_type == "annual":
        if observed_year == report_year:
            return "current_report_year", 2
        if observed_year == report_year - 1:
            return "previous_year", 4
        return "other_year", 5

    if period_type in {"year_to_date", "period_range"}:
        # For Jan-April in an April report, this is the current YTD/range for that report.
        observed_end = pd.to_datetime(
            period_info.get("observed_period_end"), errors="coerce"
        )
        report_start = pd.Timestamp(year=report_year, month=report_month_num, day=1)
        report_end = report_start + pd.offsets.MonthEnd(0)

        if pd.notna(observed_end) and observed_end == report_end:
            return "current_report_year_to_date", 2
        if observed_year == report_year - 1 and observed_month == report_month_num:
            return "same_period_previous_year", 4
        return "other_prior_period_range", 5

    if observed_month is None:
        return "unknown", 9

    report_start = pd.Timestamp(year=report_year, month=report_month_num, day=1)
    observed_start = pd.Timestamp(year=observed_year, month=observed_month, day=1)

    if observed_start == report_start:
        return "current_report_month", 1
    if observed_start == report_start - pd.DateOffset(months=1):
        return "previous_month", 2
    if observed_month == report_month_num and observed_year == report_year - 1:
        return "same_month_previous_year", 3
    if observed_start < report_start:
        return "other_prior_period", 4
    if observed_start > report_start:
        return "future_or_mislabeled_period", 8

    return "unknown", 9


def make_report_record(filepath: Path) -> dict[str, Any]:
    report_year, report_month_name = extract_year_month_from_report_path(filepath)
    report_month_num = month_name_to_number(report_month_name)
    start, end = month_period_bounds(report_year, report_month_num)

    report_id = f"TBK_{report_year}_{report_month_num:02d}"

    return {
        "report_id": report_id,
        "report_year": report_year,
        "report_month": report_month_num,
        "report_period_start": start.date().isoformat(),
        "report_period_end": end.date().isoformat(),
        "source_file": filepath.name,
        "source_path": str(filepath),
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }


# =============================================================================
# OBSERVATION EXTRACTION
# =============================================================================


def make_observations_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=OBSERVATION_COLUMNS)


def extract_semantic_observations(
    df: pd.DataFrame,
    value_cols: list[str],
    report: dict[str, Any],
) -> pd.DataFrame:
    """
    Extract only all-sector rows with known West/East/Total regions.
    Sector-level detail rows are intentionally ignored.
    """
    records: list[dict[str, Any]] = []

    usable = df[
        df["sector_norm"].eq("all")
        & df["region_norm"].isin(["west_of_rift", "east_of_rift", "total"])
    ].copy()

    for source_row_index, row in usable.iterrows():
        for col in value_cols:
            value = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(value):
                continue

            period_info = parse_period_info(col, int(report["report_year"]))
            observed_year = period_info["observed_year"]
            observed_month = period_info["observed_month"]
            observed_start_out = period_info["observed_period_start"]
            observed_end_out = period_info["observed_period_end"]

            period_relation, source_priority = classify_period_relation(
                int(report["report_year"]),
                int(report["report_month"]),
                period_info,
            )

            records.append(
                {
                    "observation_id": None,  # filled after concat
                    "report_id": report["report_id"],
                    "report_year": report["report_year"],
                    "report_month": report["report_month"],
                    "report_period_start": report["report_period_start"],
                    "report_period_end": report["report_period_end"],
                    "observed_year": observed_year,
                    "observed_month": observed_month,
                    "observed_period_start": observed_start_out,
                    "observed_period_end": observed_end_out,
                    "period_label": col,
                    "period_relation": period_relation,
                    "source_priority": source_priority,
                    "metric": "production",
                    "sector": row["sector_norm"],
                    "sector_raw": row.get("sector"),
                    "region": row["region_norm"],
                    "region_raw": row.get("region"),
                    "value": float(value),
                    "unit": "kg",
                    "source_file": report["source_file"],
                    "source_page": row.get("__source_page"),
                    "source_table_index": row.get("__source_table_index"),
                    "source_row_index": int(source_row_index),
                    "extraction_method": "semantic_sector_region_match",
                    "confidence": "high",
                    "scale_factor": 1,
                    "notes": None,
                }
            )

    return make_observations_dataframe(records)


def extract_math_only_all_sector_observations(
    df: pd.DataFrame,
    value_cols: list[str],
    report: dict[str, Any],
    tolerance_ratio: float = 0.01,
) -> pd.DataFrame:
    """
    Last-resort fallback for severely mangled tables where sector/region text is
    missing or merged into unusable strings, but the numeric rows still follow:

        row N     = West of Rift
        row N + 1 = East of Rift
        row N + 2 = Total

    This intentionally ignores sector/region text and picks the largest valid
    west + east ~= total triple for each period column. Because this is only
    used to recover all-sector totals, it emits sector='all' with medium
    confidence.
    """
    records: list[dict[str, Any]] = []

    if df.empty or not value_cols:
        return make_observations_dataframe(records)

    for col in value_cols:
        candidates: list[dict[str, Any]] = []

        for start in range(0, len(df) - 2):
            west_raw = df.iloc[start].get(col)
            east_raw = df.iloc[start + 1].get(col)
            total_raw = df.iloc[start + 2].get(col)

            west = pd.to_numeric(west_raw, errors="coerce")
            east = pd.to_numeric(east_raw, errors="coerce")
            total = pd.to_numeric(total_raw, errors="coerce")

            if not values_add_up(west, east, total, tolerance_ratio=tolerance_ratio):
                continue

            scale_factor = infer_scale_factor([west, east, total])
            west_scaled = float(west) * scale_factor
            east_scaled = float(east) * scale_factor
            total_scaled = float(total) * scale_factor

            # Prefer candidates that look like grand/all-sector blocks, but do not
            # require this because these tables can be badly OCR-mangled.
            group_text = normalize_text(
                " ".join(
                    str(v)
                    for v in df.iloc[start : start + 3].to_numpy().ravel().tolist()
                    if pd.notna(v)
                )
            )
            all_sector_bonus = int(
                any(
                    token in group_text for token in ["plantation", "estate", "estates"]
                )
                and any(
                    token in group_text
                    for token in ["smallholder", "smallholders", "ktda"]
                )
            )
            total_word_bonus = int(
                any(token in group_text for token in ["total", "grand total", "all"])
            )

            candidates.append(
                {
                    "start": start,
                    "west": west_scaled,
                    "east": east_scaled,
                    "total": total_scaled,
                    "scale_factor": scale_factor,
                    "all_sector_bonus": all_sector_bonus,
                    "total_word_bonus": total_word_bonus,
                    "total_sort_value": total_scaled,
                }
            )

        if not candidates:
            continue

        # Pick one triple per period column. For all-sector production totals,
        # the grand total should usually be the largest valid triple. The bonus
        # fields break ties in favor of text that looks like all-sector totals.
        best = sorted(
            candidates,
            key=lambda x: (
                x["all_sector_bonus"],
                x["total_word_bonus"],
                x["total_sort_value"],
            ),
            reverse=True,
        )[0]

        period_info = parse_period_info(col, int(report["report_year"]))
        period_relation, source_priority = classify_period_relation(
            int(report["report_year"]),
            int(report["report_month"]),
            period_info,
        )

        row_specs = [
            (best["start"], "west_of_rift", best["west"]),
            (best["start"] + 1, "east_of_rift", best["east"]),
            (best["start"] + 2, "total", best["total"]),
        ]

        for source_row_index, region, value in row_specs:
            row = df.iloc[source_row_index]
            note_bits = [
                "sector_region_text_unusable",
                "recovered_from_numeric_row_group",
            ]
            if best["scale_factor"] != 1:
                note_bits.append("values_scaled_from_million_kg_to_kg")

            records.append(
                {
                    "observation_id": None,
                    "report_id": report["report_id"],
                    "report_year": report["report_year"],
                    "report_month": report["report_month"],
                    "report_period_start": report["report_period_start"],
                    "report_period_end": report["report_period_end"],
                    "observed_year": period_info["observed_year"],
                    "observed_month": period_info["observed_month"],
                    "observed_period_start": period_info["observed_period_start"],
                    "observed_period_end": period_info["observed_period_end"],
                    "period_label": col,
                    "period_relation": period_relation,
                    "source_priority": max(source_priority, 5),
                    "metric": "production",
                    "sector": "all",
                    "sector_raw": row.get("sector", pd.NA),
                    "region": region,
                    "region_raw": row.get("region", pd.NA),
                    "value": value,
                    "unit": "kg",
                    "source_file": report["source_file"],
                    "source_page": row.get("__source_page"),
                    "source_table_index": row.get("__source_table_index"),
                    "source_row_index": int(source_row_index),
                    "extraction_method": "math_only_row_group_recovery",
                    "confidence": "medium",
                    "scale_factor": best["scale_factor"],
                    "notes": "; ".join(note_bits),
                }
            )

    return make_observations_dataframe(records)


def choose_best_all_sector_group(
    df: pd.DataFrame,
    value_cols: list[str],
) -> list[int] | None:
    """
    Choose exactly one 3-row group to represent all-sector totals.

    Prefers:
      1. groups whose text looks like all-sector total
      2. groups with the largest total value
      3. groups later in the table
    """
    candidates = []

    for start in range(0, len(df) - 2):  # sliding window
        idx = [start, start + 1, start + 2]
        group = df.loc[idx]

        region_pattern = group["region_norm"].tolist()
        if region_pattern != ["west_of_rift", "east_of_rift", "total"]:
            continue

        valid_cols = []
        total_sum = 0.0
        max_total = 0.0

        for col in value_cols:
            west = pd.to_numeric(group.iloc[0][col], errors="coerce")
            east = pd.to_numeric(group.iloc[1][col], errors="coerce")
            total = pd.to_numeric(group.iloc[2][col], errors="coerce")

            if values_add_up(west, east, total):
                valid_cols.append(col)
                total_sum += float(total)
                max_total = max(max_total, float(total))

        if not valid_cols:
            continue

        text = " ".join(
            group["sector"].astype("string").fillna("").tolist()
            + group["region"].astype("string").fillna("").tolist()
        )
        text_norm = normalize_text(text)

        all_text_bonus = int(
            (
                ("estate" in text_norm or "plantation" in text_norm)
                and "smallholder" in text_norm
                and ("independent" in text_norm or "nyayo" in text_norm)
            )
            or "grand total" in text_norm
        )

        bad_all_penalty = int(
            "independent" in text_norm
            and not (
                "estate" in text_norm
                or "plantation" in text_norm
                or "smallholder" in text_norm
            )
        )

        candidates.append(
            {
                "idx": idx,
                "start": start,
                "valid_cols": valid_cols,
                "all_text_bonus": all_text_bonus,
                "bad_all_penalty": bad_all_penalty,
                "total_sum": total_sum,
                "max_total": max_total,
            }
        )

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda c: (
            c["all_text_bonus"],
            -c["bad_all_penalty"],
            c["max_total"],
            c["total_sum"],
            c["start"],
        ),
        reverse=True,
    )

    return candidates[0]["idx"]


def keep_only_best_all_sector_group(
    df: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()

    if not value_cols:
        return df

    best_idx = choose_best_all_sector_group(df, value_cols)

    if best_idx is None:
        return df

    # Only demote rows that were promoted by weak/mathy recovery.
    # Keep explicit semantic all rows if they are the chosen group.
    candidate_mask = df["sector_norm"].eq("all") & df["region_norm"].isin(
        ["west_of_rift", "east_of_rift", "total"]
    )

    df.loc[candidate_mask, "sector_norm"] = "not_all_candidate"

    df.loc[best_idx, "sector_norm"] = "all"
    df.loc[best_idx[0], "region_norm"] = "west_of_rift"
    df.loc[best_idx[1], "region_norm"] = "east_of_rift"
    df.loc[best_idx[2], "region_norm"] = "total"

    df.loc[best_idx[0], "region"] = "West of Rift"
    df.loc[best_idx[1], "region"] = "East of Rift"
    df.loc[best_idx[2], "region"] = "Total"

    return df


def validate_sector_region_totals(
    observations: pd.DataFrame, tolerance_ratio: float = 0.01
) -> pd.DataFrame:
    """
    Adds notes when west + east does not match total for the same sector/period.
    Does not drop rows; this is raw data, so questionable rows stay auditable.
    """
    if observations.empty:
        return observations

    observations = observations.copy()

    group_cols = [
        "report_id",
        "period_label",
        "metric",
        "sector",
        "unit",
    ]

    for _, group in observations.groupby(group_cols, dropna=False):
        by_region = group.set_index("region")
        if not {"west_of_rift", "east_of_rift", "total"}.issubset(by_region.index):
            continue

        west = by_region.loc["west_of_rift", "value"]
        east = by_region.loc["east_of_rift", "value"]
        total = by_region.loc["total", "value"]

        # If duplicate regions exist, skip validation here. Post-processing can handle duplicates.
        if (
            isinstance(west, pd.Series)
            or isinstance(east, pd.Series)
            or isinstance(total, pd.Series)
        ):
            continue

        if values_add_up(west, east, total, tolerance_ratio=tolerance_ratio):
            continue

        idx = group.index
        old_notes = observations.loc[idx, "notes"].fillna("").astype(str)
        observations.loc[idx, "notes"] = (
            old_notes + "; west_plus_east_does_not_equal_total"
        ).str.strip("; ")
        observations.loc[idx, "confidence"] = observations.loc[
            idx, "confidence"
        ].replace({"high": "medium"})

    return observations


def apply_scale_recovery_to_small_kg_values(observations: pd.DataFrame) -> pd.DataFrame:
    """
    If a complete west/east/total triple has values like 35.8, 18.0, 54.5,
    treat it as million kg and convert to kg.
    """
    if observations.empty:
        return observations

    observations = observations.copy()

    group_cols = [
        "report_id",
        "period_label",
        "metric",
        "sector",
        "unit",
    ]

    for _, group in observations.groupby(group_cols, dropna=False):
        by_region = group.set_index("region")
        if not {"west_of_rift", "east_of_rift", "total"}.issubset(by_region.index):
            continue

        west = by_region.loc["west_of_rift", "value"]
        east = by_region.loc["east_of_rift", "value"]
        total = by_region.loc["total", "value"]

        if (
            isinstance(west, pd.Series)
            or isinstance(east, pd.Series)
            or isinstance(total, pd.Series)
        ):
            continue

        if not values_add_up(west, east, total):
            continue

        scale_factor = infer_scale_factor([west, east, total])
        if scale_factor == 1:
            continue

        idx = group.index
        observations.loc[idx, "value"] = observations.loc[idx, "value"] * scale_factor
        observations.loc[idx, "scale_factor"] = scale_factor
        observations.loc[idx, "extraction_method"] = (
            observations.loc[idx, "extraction_method"] + "+scale_recovery"
        )
        observations.loc[idx, "confidence"] = observations.loc[
            idx, "confidence"
        ].replace({"high": "medium"})
        old_notes = observations.loc[idx, "notes"].fillna("").astype(str)
        observations.loc[idx, "notes"] = (
            old_notes + "; values_scaled_from_million_kg_to_kg"
        ).str.strip("; ")

    return observations


def extract_observations_from_clean_table(
    df: pd.DataFrame, report: dict[str, Any]
) -> pd.DataFrame:
    value_cols = get_period_value_columns(df)
    if not value_cols:
        return make_observations_dataframe([])

    observations = extract_semantic_observations(df, value_cols, report)

    # If sector/region text is unusable, fall back to numeric row-group recovery.
    # This scans for consecutive triples where row1 + row2 ~= row3 and treats
    # them as West of Rift / East of Rift / Total for the all sector.
    if observations.empty:
        observations = extract_math_only_all_sector_observations(df, value_cols, report)

    observations = apply_scale_recovery_to_small_kg_values(observations)
    observations = validate_sector_region_totals(observations)
    return observations


def finalize_observation_ids(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.copy().reset_index(drop=True)
    if observations.empty:
        return observations

    observations["observation_id"] = [
        f"{row.report_id}_OBS_{i + 1:06d}"
        for i, row in enumerate(observations.itertuples(index=False))
    ]
    return observations


# =============================================================================
# REPORT EXTRACTION
# =============================================================================


def extract_report_data(filepath: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    report = make_report_record(filepath)
    table_groups = read_report_table_groups(filepath)

    if not table_groups:
        raise ValueError("Camelot found no usable tables")

    extracted_tables: list[pd.DataFrame] = []

    for group in table_groups:
        cleaned = clean_report_dataframe(group)
        if cleaned.empty:
            continue

        observations = extract_observations_from_clean_table(cleaned, report)
        if not observations.empty:
            extracted_tables.append(observations)

    if not extracted_tables:
        raise ValueError("No usable production observations found")

    observations = pd.concat(extracted_tables, ignore_index=True)

    # Drop exact duplicates caused by repeated split-table headers or duplicated Camelot tables.
    dedupe_subset = [
        "report_id",
        "period_label",
        "metric",
        "sector",
        "region",
        "value",
        "unit",
        "source_page",
    ]
    observations = observations.drop_duplicates(subset=dedupe_subset).reset_index(
        drop=True
    )
    observations = finalize_observation_ids(observations)

    return observations, report


# =============================================================================
# OUTPUT
# =============================================================================


def resolve_output_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    """
    Supports either:
      observations_raw_file: data/clean/observations_raw.csv
      reports_file: data/clean/reports.csv

    Or falls back to the old config key:
      data_file: data/clean/observations_raw.csv
    """
    observations_file = Path(
        config.get(
            "observations_raw_file",
            config.get("data_file", "data/clean/observations_raw.csv"),
        )
    )
    reports_file = Path(
        config.get("reports_file", observations_file.with_name("reports.csv"))
    )
    return observations_file, reports_file


def main() -> None:
    config = get_config()
    reports_dir = Path(config["ocr_dir"])
    observations_file, reports_file = resolve_output_paths(config)

    all_observations: list[pd.DataFrame] = []
    report_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    # Full dataset run.
    # For debugging one file, temporarily replace with:
    #   import random
    #   files = random.sample(sorted(reports_dir.glob("*.pdf")), 1)
    files = sorted(reports_dir.glob("*.pdf"))

    for filepath in files:
        try:
            observations, report = extract_report_data(filepath)
            all_observations.append(observations)
            report_records.append(report)
            print(
                f"  ✓ success | {filepath.name} | {len(observations)} raw observations"
            )
        except Exception as exc:
            failures.append({"source_file": filepath.name, "error": str(exc)})
            print(f"  ✗ failed  | {filepath.name}: {exc}")

    observations_file.parent.mkdir(parents=True, exist_ok=True)
    reports_file.parent.mkdir(parents=True, exist_ok=True)

    if all_observations:
        observations_raw = pd.concat(all_observations, ignore_index=True)
        observations_raw = finalize_observation_ids(observations_raw)
        observations_raw = observations_raw[OBSERVATION_COLUMNS]
        observations_raw = observations_raw.sort_values(
            [
                "observed_period_start",
                "report_period_start",
                "sector",
                "region",
                "period_label",
            ],
            na_position="last",
        )
        observations_raw.to_csv(observations_file, index=False)
    else:
        observations_raw = make_observations_dataframe([])
        observations_raw.to_csv(observations_file, index=False)

    reports = pd.DataFrame(report_records, columns=REPORT_COLUMNS)
    reports = (
        reports.drop_duplicates(subset=["report_id"]).sort_values("report_period_start")
        if not reports.empty
        else reports
    )
    reports.to_csv(reports_file, index=False)

    if failures:
        failures_file = observations_file.with_name("extraction_failures.csv")
        pd.DataFrame(failures).to_csv(failures_file, index=False)
        print(f"\nWrote failures to {failures_file}")

    print(f"\nWrote raw observations to {observations_file}")
    print(f"Wrote reports to {reports_file}")
    print(
        f"Extracted {len(observations_raw)} raw observations from {len(report_records)} reports"
    )


if __name__ == "__main__":
    main()
