"""Command-line reporting entry point."""
from __future__ import annotations
import argparse
from pathlib import Path
from .reporting import generate_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate auditable retail analytics reports.")
    parser.add_argument("--input", default="data/raw/Online Retail.xlsx", help="Input .xlsx, .xls, or .csv file")
    parser.add_argument("--output", default="reports/generated", help="Directory for generated reports")
    args = parser.parse_args()
    result = generate_reports(Path(args.input), Path(args.output))
    print(f"Reports written to: {result['output_directory']}")
    print(f"Files: {result['files']}")

if __name__ == "__main__":
    main()
