"""Update the canonical KASPA daily OHLCV CSV from free CoinGecko data.

The original Excel workbook is used as the seed. New rows are appended to a CSV
inside the workspace so we do not mutate the user's attached workbook.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = Path(r"C:\Users\Andy Chi\Desktop\KASPA_Historical_Data.xlsx")
DEFAULT_OUTPUT = ROOT / "data" / "kaspa_daily_ohlcv.csv"
DEFAULT_COMPRESSED_SEED = ROOT / "data" / "kaspa_daily_ohlcv.seed.csv.gz.b64"
DEFAULT_COMPRESSED_SEED_CHUNKS = ROOT / "data" / "kaspa_daily_ohlcv.seed.csv.gz.b64.d"
DEFAULT_STATUS = ROOT / "outputs" / "kaspa_data_status.json"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_OHLC_DAY_BUCKETS = (1, 7, 14, 30, 90, 180, 365)


def latest_complete_utc_date() -> pd.Timestamp:
    # Exclude the current UTC day. It can still move and should not train a
    # daily strategy until the candle is finished.
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    return today - pd.Timedelta(days=1)


def load_existing_or_seed(output_path: Path, seed_path: Path) -> pd.DataFrame:
    if output_path.exists():
        df = pd.read_csv(output_path)
    elif DEFAULT_COMPRESSED_SEED.exists() or DEFAULT_COMPRESSED_SEED_CHUNKS.exists():
        raw = base64.b64decode(read_compressed_seed_text())
        df = pd.read_csv(io.BytesIO(gzip.decompress(raw)))
    else:
        df = pd.read_excel(seed_path, sheet_name="Historical Data")

    column_order = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = set(column_order).difference(df.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")

    df = df[column_order].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().sort_values("Date").drop_duplicates("Date", keep="last")


def read_compressed_seed_text() -> str:
    if DEFAULT_COMPRESSED_SEED.exists():
        return DEFAULT_COMPRESSED_SEED.read_text(encoding="ascii").strip()

    parts = sorted(DEFAULT_COMPRESSED_SEED_CHUNKS.glob("part-*.txt"))
    if not parts:
        raise FileNotFoundError(
            f"No compressed seed or seed chunks found under {DEFAULT_COMPRESSED_SEED_CHUNKS}"
        )
    return "".join(part.read_text(encoding="ascii").strip() for part in parts)


def coingecko_get(path: str, params: dict[str, Any]) -> Any:
    query = urllib.parse.urlencode(params)
    url = f"{COINGECKO_BASE}{path}?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "kaspa-rf-research/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def coingecko_ohlc_days(days: int) -> int:
    """CoinGecko OHLC accepts fixed day buckets, not arbitrary windows."""
    clamped_days = max(1, min(days, COINGECKO_OHLC_DAY_BUCKETS[-1]))
    for bucket in COINGECKO_OHLC_DAY_BUCKETS:
        if clamped_days <= bucket:
            return bucket
    return COINGECKO_OHLC_DAY_BUCKETS[-1]


def fetch_recent_ohlc(days: int) -> pd.DataFrame:
    rows = coingecko_get(
        "/coins/kaspa/ohlc",
        {
            "vs_currency": "usd",
            "days": str(coingecko_ohlc_days(days)),
            "precision": "full",
        },
    )
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected CoinGecko OHLC response: {rows!r}")

    df = pd.DataFrame(rows, columns=["Timestamp", "Open", "High", "Low", "Close"])
    df["DateTime"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df["Date"] = df["DateTime"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    complete = latest_complete_utc_date()
    df = df[df["Date"] <= complete].copy()
    if df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

    daily = (
        df.sort_values("DateTime")
        .groupby("Date", as_index=False)
        .agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    )
    return daily


def fetch_recent_volume(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    from_ts = int((start_date - pd.Timedelta(days=1)).timestamp())
    to_ts = int((end_date + pd.Timedelta(days=1)).timestamp())
    data = coingecko_get(
        "/coins/kaspa/market_chart/range",
        {
            "vs_currency": "usd",
            "from": str(from_ts),
            "to": str(to_ts),
            "interval": "daily",
            "precision": "full",
        },
    )
    volumes = data.get("total_volumes", [])
    df = pd.DataFrame(volumes, columns=["Timestamp", "Volume"])
    if df.empty:
        return pd.DataFrame(columns=["Date", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    return df[["Date", "Volume"]]


def fetch_updates(last_date: pd.Timestamp) -> pd.DataFrame:
    latest_complete = latest_complete_utc_date()
    start_date = last_date + pd.Timedelta(days=1)
    if start_date > latest_complete:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    fetch_days = int((latest_complete - last_date).days + 5)
    ohlc = fetch_recent_ohlc(fetch_days)
    ohlc = ohlc[(ohlc["Date"] >= start_date) & (ohlc["Date"] <= latest_complete)].copy()
    if ohlc.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    volume = fetch_recent_volume(ohlc["Date"].min(), ohlc["Date"].max())
    merged = ohlc.merge(volume, on="Date", how="left")
    merged["Volume"] = merged["Volume"].ffill().bfill().fillna(0).round().astype("int64")
    return merged[["Date", "Open", "High", "Low", "Close", "Volume"]]


def write_canonical_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)


def write_status(status: dict[str, Any], status_path: Path = DEFAULT_STATUS) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def get_data_status(
    output_path: Path = DEFAULT_OUTPUT,
    seed_path: Path = DEFAULT_SEED,
) -> dict[str, Any]:
    df = load_existing_or_seed(output_path, seed_path)
    last_date = pd.Timestamp(df["Date"].max())
    latest_complete = latest_complete_utc_date()
    return {
        "source": "coingecko_free",
        "data_file": str(output_path if output_path.exists() else seed_path),
        "canonical_csv": str(output_path),
        "last_data_date": str(last_date.date()),
        "latest_complete_utc_date": str(latest_complete.date()),
        "needs_update": bool(last_date < latest_complete),
        "rows": int(len(df)),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_update(
    output_path: Path = DEFAULT_OUTPUT,
    seed_path: Path = DEFAULT_SEED,
    status_path: Path = DEFAULT_STATUS,
) -> dict[str, Any]:
    before = load_existing_or_seed(output_path, seed_path)
    last_before = pd.Timestamp(before["Date"].max())
    updates = fetch_updates(last_before)
    combined = before
    if not updates.empty:
        combined = (
            pd.concat([before, updates], ignore_index=True)
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )
    write_canonical_csv(combined, output_path)

    last_after = pd.Timestamp(combined["Date"].max())
    status = get_data_status(output_path, seed_path)
    status.update(
        {
            "previous_last_data_date": str(last_before.date()),
            "updated_last_data_date": str(last_after.date()),
            "new_rows": int(len(combined) - len(before)),
            "appended_dates": [
                str(pd.Timestamp(value).date()) for value in updates["Date"].tolist()
            ],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    write_status(status, status_path)
    # Be gentle with the public demo API if this is called repeatedly.
    time.sleep(0.25)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = get_data_status(args.output, args.seed) if args.status_only else run_update(args.output, args.seed, args.status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
