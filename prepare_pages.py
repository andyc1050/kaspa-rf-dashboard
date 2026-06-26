"""Prepare the GitHub Pages artifact from generated dashboard outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SITE = ROOT / "site"


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    copy_if_exists(OUTPUTS / "kaspa_rf_dashboard.html", SITE / "index.html")
    copy_if_exists(OUTPUTS / "kaspa_rf_metrics.json", SITE / "assets" / "kaspa_rf_metrics.json")
    copy_if_exists(OUTPUTS / "kaspa_data_status.json", SITE / "assets" / "kaspa_data_status.json")
    copy_if_exists(ROOT / "data" / "kaspa_daily_ohlcv.csv", SITE / "assets" / "kaspa_daily_ohlcv.csv")

    manifest = {
        "site_entry": "index.html",
        "generated_from": "kaspa_rf_trading",
        "artifacts": sorted(path.name for path in (SITE / "assets").glob("*"))
        if (SITE / "assets").exists()
        else [],
    }
    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    (SITE / "assets" / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(SITE)


if __name__ == "__main__":
    main()
