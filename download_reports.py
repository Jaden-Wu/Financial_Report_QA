"""Check local report PDFs, or download missing ones from verified source URLs.

Default: python download_reports.py
Download: python download_reports.py --download
Fill each report's url in report_sources.json before downloading missing PDFs.
Existing files are checked and never overwritten. No third-party packages required.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


BASE_DIR = Path(__file__).resolve().parent


def verify_pdf(data, report):
    if not data.startswith(b"%PDF-"):
        raise ValueError("File does not have a PDF header")
    if len(data) != report["bytes"]:
        raise ValueError("File size differs from the recorded report")
    if hashlib.sha256(data).hexdigest() != report["sha256"]:
        raise ValueError("SHA256 differs from the recorded report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(
        (BASE_DIR / "report_sources.json").read_text(encoding="utf-8")
    )
    reports_dir = BASE_DIR / "reports"
    failures = 0
    for report in manifest["reports"]:
        filename = report["filename"]
        if Path(filename).name != filename or not filename.endswith(".pdf"):
            print(f"INVALID FILENAME: {filename}")
            failures += 1
            continue
        target = reports_dir / filename
        try:
            if target.exists():
                verify_pdf(target.read_bytes(), report)
                print(f"OK (existing, skipped): {filename}")
                continue
            if not args.download:
                print(f"MISSING: {filename}")
                failures += 1
                continue
            url = report.get("url")
            if not url:
                raise ValueError("Verified source URL is missing in report_sources.json")
            if urllib.parse.urlparse(url).scheme != "https":
                raise ValueError("Source URL must use HTTPS")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if urllib.parse.urlparse(response.geturl()).scheme != "https":
                    raise ValueError("Redirect target must use HTTPS")
                data = response.read()
            verify_pdf(data, report)
            reports_dir.mkdir(exist_ok=True)
            # Exclusive creation prevents accidental overwrite, including races.
            with target.open("xb") as output:
                output.write(data)
            print(f"DOWNLOADED: {filename}")
        except (OSError, ValueError) as error:
            print(f"ERROR: {filename}: {error}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
