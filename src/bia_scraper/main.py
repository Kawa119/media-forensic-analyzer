from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .database import Database
from .scraper import YellGeScraper
from .exporter import export_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape yell.ge business directory")
    parser.add_argument("start", type=int, help="Start company ID")
    parser.add_argument("end", type=int, help="End company ID")
    parser.add_argument("--proxy", action="append", help="HTTP proxy URL, can be used multiple times")
    parser.add_argument("--export", type=Path, help="Path to export CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db = Database()
    scraper = YellGeScraper(db=db, proxies=args.proxy)
    companies = scraper.scrape_range(args.start, args.end)
    if args.export:
        export_to_csv(companies, args.export)
    else:
        for comp in companies:
            print(asdict(comp))


if __name__ == "__main__":
    main()
