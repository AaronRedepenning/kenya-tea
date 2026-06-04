import calendar
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import httpx
import pandas as pd
import yaml

# CONSTANTS
MONTHS_PER_YEAR = 12
MAX_REPORTS = 10 * MONTHS_PER_YEAR

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; KenyaTeaDataResearch/0.1; "
        "+https://github.com/AaronRedepenning/kenya-tea)"
    ),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
}

TBK_BASE_URL = "https://www.teaboard.or.ke/images/downloads/performance-highlights/"
TBK_REPORT_TEMPLATE = "kenya-tea-industry-performance-highlights-{year}-{month}.pdf"


# FUNCTIONS
def get_config() -> dict:
    with Path("config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_report_name(year: int, month: int, lower: bool = False) -> str:
    month_name = calendar.month_name[month]
    if lower:
        month_name = month_name.lower()
    return TBK_REPORT_TEMPLATE.format(year=year, month=month_name)


def make_report_urls(year: int, month: int) -> list[str]:
    return [
        urljoin(
            TBK_BASE_URL,
            make_report_name(year, month, option),
        )
        for option in [False, True]
    ]


def is_valid_pdf(response: httpx.Response, first_chunk: bytes) -> bool:
    if response.status_code != 200:
        return False

    content_type = response.headers.get("content-type", "").lower()

    if "application/pdf" in content_type:
        return True

    return first_chunk[:4] == b"%PDF"


def download_pdf(url: str, out_path: Path) -> bool:
    timeout = httpx.Timeout(60.0, connect=20.0)

    with httpx.Client(
        headers=HTTP_HEADERS,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        with client.stream("GET", url) as response:
            iterator = response.iter_bytes(chunk_size=64 * 1024)

            try:
                first_chunk = next(iterator)
            except StopIteration:
                return False

            if not is_valid_pdf(response, first_chunk):
                return False

            with out_path.open("wb") as f:
                f.write(first_chunk)

                for chunk in iterator:
                    if chunk:
                        f.write(chunk)

    return True


# MAIN
def main():
    config = get_config()
    reports_dir = Path(config["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    today = date.today()
    year, month = today.year, today.month
    fail_count = 0
    results = []

    while len(results) < MAX_REPORTS:
        # Try to download report for year / month
        filepath = reports_dir / make_report_name(year, month)
        success_url = None

        for url in make_report_urls(year, month):
            if download_pdf(url, filepath):
                success_url = url
                break

        downloaded = success_url is not None

        # Record download status
        results.append(
            {
                "date": date(year, month, 1),
                "url": success_url if downloaded else "",
                "filepath": filepath if downloaded else "",
                "downloaded": True if downloaded else False,
            }
        )

        # Check download success
        month_name = calendar.month_name[month]

        if not downloaded:
            fail_count += 1
            print(f"  ✗ failed  | could not download data for {year}-{month_name}")

            if fail_count > MONTHS_PER_YEAR:
                break
        else:
            fail_count = 0
            print(f"  ✓ success | downloaded data for {year}-{month_name}")

        # Decrement month / year
        month -= 1

        if month == 0:
            month = 12
            year -= 1

    # Save results
    pd.DataFrame(results).to_csv(reports_dir / "manifest.csv")


if __name__ == "__main__":
    main()
