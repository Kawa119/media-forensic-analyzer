# Media Forensic Analyzer & Yell.ge Scraper

This repository contains utilities for media forensics and a modular web scraper
for the Georgian business directory [yell.ge](https://yell.ge/).

## Yell.ge Scraper

The scraper is located in `src/bia_scraper/` and provides:

- **Proxy support** to bypass IP-based blocking.
- **Persistent state tracking** using SQLite to resume interrupted runs.
- **Data validation** for Georgian-specific phone and email patterns.
- **CSV export** for scraped company records.

### Usage

```bash
python -m bia_scraper.main START_ID END_ID --proxy http://proxy:port --export output.csv
```

`--proxy` can be supplied multiple times to rotate through several proxies.

The scraper follows respectful rate limiting by default. See the source code for
additional configuration options.
