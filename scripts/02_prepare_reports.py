import subprocess
import tempfile
from pathlib import Path

import yaml


# FUNCTIONS
def get_config() -> dict:
    with Path("config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ocr_pdf(in_path: Path, out_path: Path, pages: list[int] | None = None) -> None:
    if pages:
        subprocess.run(
            [
                "qpdf",
                str(in_path),
                "--pages",
                str(in_path),
                ",".join([str(page) for page in pages]),
                "--",
                str(out_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        in_path = out_path

    subprocess.run(
        [
            "ocrmypdf",
            "--force-ocr",
            "--deskew",
            "--clean",
            str(in_path),
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# MAIN
def main():
    config = get_config()
    reports_dir = Path(config["reports_dir"])
    ocr_dir = Path(config["ocr_dir"])
    ocr_dir.mkdir(parents=True, exist_ok=True)

    for in_path in reports_dir.glob("*.pdf"):
        out_path = ocr_dir / in_path.name
        ocr_pdf(in_path, out_path, [1, 2, 3])
        print(f"  ✓ success | {in_path.name}")


if __name__ == "__main__":
    main()
