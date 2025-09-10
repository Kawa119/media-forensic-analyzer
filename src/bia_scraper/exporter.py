from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .models import Company


def export_to_csv(companies: Iterable[Company], path: Path = Path("yell_companies.csv")) -> None:
    """Export iterable of companies to CSV file."""
    fieldnames = ["identifier", "name", "phone", "email", "address", "website"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for comp in companies:
            writer.writerow({f: getattr(comp, f) for f in fieldnames})
