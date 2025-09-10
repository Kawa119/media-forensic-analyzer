from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?995)?\s?\(?\d{2,3}\)?[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}")


@dataclass
class Company:
    """Data model representing a single company listing."""

    identifier: str
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    website: Optional[str] = None

    raw_html: str = field(repr=False, default="")

    def is_valid(self) -> bool:
        """Run basic validation to ensure extracted data is useful."""
        return self._has_valid_identifier() and self._has_contact_information()

    def _has_valid_identifier(self) -> bool:
        return bool(self.identifier)

    def _has_contact_information(self) -> bool:
        return any([self.phone, self.email, self.address, self.website])

    def normalize(self) -> None:
        """Clean up fields after extraction."""
        if self.email and not EMAIL_RE.fullmatch(self.email):
            self.email = None
        if self.phone and not PHONE_RE.search(self.phone):
            self.phone = None
