from __future__ import annotations

import random
import time
from dataclasses import asdict
from typing import Iterable, List, Optional

import requests
from bs4 import BeautifulSoup

from .models import Company, EMAIL_RE, PHONE_RE
from .database import Database


class YellGeScraper:
    """Scraper implementation targeting yell.ge business directory."""

    BASE_URL = "https://yell.ge/company/{}"

    def __init__(
        self,
        db: Optional[Database] = None,
        proxies: Optional[List[str]] = None,
        rate_limit: float = 1.0,
        timeout: int = 30,
    ):
        self.db = db or Database()
        self.proxies = proxies or []
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self._random_ua(),
            "Accept-Language": "en-US,en;q=0.9,ka;q=0.8",
        })

    # ---------------------------- utility methods ----------------------------
    def _random_ua(self) -> str:
        uas = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0 Safari/537.36",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        ]
        return random.choice(uas)

    def _pick_proxy(self) -> Optional[dict]:
        if not self.proxies:
            return None
        proxy = random.choice(self.proxies)
        return {"http": proxy, "https": proxy}

    def _request(self, url: str) -> Optional[requests.Response]:
        for attempt in range(3):
            proxy = self._pick_proxy()
            try:
                resp = self.session.get(url, proxies=proxy, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 404:
                    return None
                time.sleep(2 ** attempt)
            except requests.RequestException:
                time.sleep(2 ** attempt)
        return None

    # ---------------------------- scraping logic ----------------------------
    def scrape_range(self, start_id: int, end_id: int) -> Iterable[Company]:
        """Scrape a range of numeric company identifiers."""
        for cid in range(start_id, end_id + 1):
            url = self.BASE_URL.format(cid)
            resp = self._request(url)
            time.sleep(self.rate_limit)
            if not resp:
                continue
            company = self._parse_company(cid, resp.text)
            company.normalize()
            self.db.save_company(company)
            self.db.update_progress(cid)
            yield company

    def _parse_company(self, cid: int, html: str) -> Company:
        soup = BeautifulSoup(html, "lxml")
        name = self._extract_text(soup, "h1")
        phone = self._extract_phone(soup)
        email = self._extract_email(soup)
        address = self._extract_text(soup, "address")
        website = self._extract_website(soup)
        return Company(
            identifier=str(cid),
            name=name,
            phone=phone,
            email=email,
            address=address,
            website=website,
            raw_html=html,
        )

    @staticmethod
    def _extract_text(soup: BeautifulSoup, selector: str) -> Optional[str]:
        element = soup.select_one(selector)
        if element:
            return element.get_text(strip=True)
        return None

    @staticmethod
    def _extract_phone(soup: BeautifulSoup) -> Optional[str]:
        link = soup.select_one("a[href^='tel']")
        if link:
            href = link.get("href", "")
            if href:
                return href.split(":", 1)[-1].strip()
            return link.get_text(strip=True)
        text = soup.find(string=PHONE_RE)
        if text:
            match = PHONE_RE.search(text)
            if match:
                return match.group(0)
        return None

    @staticmethod
    def _extract_email(soup: BeautifulSoup) -> Optional[str]:
        link = soup.select_one("a[href^='mailto']")
        if link:
            href = link.get("href", "")
            if href:
                return href.split(":", 1)[-1].strip()
            return link.get_text(strip=True)
        text = soup.find(string=EMAIL_RE)
        if text:
            match = EMAIL_RE.search(text)
            if match:
                return match.group(0)
        return None

    @staticmethod
    def _extract_website(soup: BeautifulSoup) -> Optional[str]:
        link = soup.select_one("a.website, a[href^='http']")
        if link and link.has_attr("href"):
            href = link["href"].strip()
            # Avoid returning yell.ge internal links
            if href and not href.startswith("/") and "yell.ge" not in href:
                return href
        return None


if __name__ == "__main__":
    scraper = YellGeScraper()
    last = scraper.db.get_last_id()
    for company in scraper.scrape_range(last + 1, last + 5):
        print(asdict(company))
