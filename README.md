# KASPA Random Forest Dashboard

Research dashboard for a long/cash Random Forest strategy on daily KAS-USD data.

This is not financial advice. It is an educational walk-forward backtest and
signal visualizer intended to make the model behavior easier to inspect.

## What It Does

- Pulls daily KAS price and volume data from CoinGecko's free API.
- Trains a lightweight Random Forest ensemble implemented in pure Python,
  pandas, and numpy.
- Runs a walk-forward backtest with retraining every 30 days.
- Compares the model strategy against buy-and-hold.
- Builds a static HTML dashboard for GitHub Pages.
- Runs automatically every day with GitHub Actions.

The default strategy is long/cash only:

- `LONG` means the model probability clears the entry threshold.
- `CASH` means do not enter a long position, or exit/avoid long exposure.
- No shorting is used.
- Default entry threshold is `0.55`; exit threshold is `0.50`.
- Backtest includes 10 bps friction on position changes.

## Run Locally

```bash
python -m pip install -r requirements.txt
python update_kaspa_data.py
python random_forest_kaspa.py --input data/kaspa_daily_ohlcv.csv
python build_dashboard.py
```

Open:

```text
outputs/kaspa_rf_dashboard.html
```

For the local dashboard server with an update button:

```bash
python kaspa_dashboard_server.py
```

Then open:

```text
http://127.0.0.1:8765/
```

## GitHub Pages

The workflow in `.github/workflows/pages.yml`:

- runs daily at `01:15 UTC`
- can be triggered manually from the Actions tab
- updates KAS data from CoinGecko
- retrains the model
- rebuilds the dashboard
- deploys `site/index.html` to GitHub Pages

Enable Pages with:

```text
Settings -> Pages -> Source: GitHub Actions
```

The public site should be:

```text
https://andyc1050.github.io/kaspa-rf-dashboard/
```

## Data Bootstrap

The repo includes a compressed bootstrap seed at:

```text
data/kaspa_daily_ohlcv.seed.csv.gz.b64
```

On the first run, `update_kaspa_data.py` uses that seed to create
`data/kaspa_daily_ohlcv.csv`, then appends new completed daily candles from
CoinGecko.

## Outputs

- `outputs/kaspa_rf_metrics.json`
- `outputs/kaspa_rf_equity_curve.csv`
- `outputs/kaspa_rf_feature_importance.csv`
- `outputs/kaspa_rf_dashboard.html`
- `site/index.html` for GitHub Pages deployment
