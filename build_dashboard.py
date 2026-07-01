"""Build a standalone HTML dashboard for the KASPA Random Forest backtest."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DASHBOARD_PATH = OUTPUTS / "kaspa_rf_dashboard.html"


def pct(value: float | int) -> str:
    return f"{float(value) * 100:.1f}%"


def number(value: float | int, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def money(value: float | int) -> str:
    return f"${float(value):.6f}"


def svg_path(values: list[float], *, x0: int, y0: int, width: int, height: int) -> str:
    if not values:
        return ""
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        low -= 1.0
        high += 1.0
    points: list[str] = []
    denom = max(1, len(values) - 1)
    for i, value in enumerate(values):
        x = x0 + width * i / denom
        y = y0 + height - (value - low) / (high - low) * height
        points.append(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}")
    return " ".join(points)


def sample_equity(equity: pd.DataFrame, max_points: int = 520) -> pd.DataFrame:
    if len(equity) <= max_points:
        return equity.copy()
    step = max(1, len(equity) // max_points)
    sampled = equity.iloc[::step].copy()
    if sampled.index[-1] != equity.index[-1]:
        sampled = pd.concat([sampled, equity.tail(1)])
    return sampled.reset_index(drop=True)


def bars(features: pd.DataFrame) -> str:
    top = features.head(14).copy()
    max_value = max(float(top["importance"].max()), 1e-12)
    rows = []
    for _, row in top.iterrows():
        width = float(row["importance"]) / max_value * 100
        rows.append(
            f"""
            <div class="bar-row">
              <span>{html.escape(str(row["feature"]))}</span>
              <div class="bar-track"><div style="width:{width:.1f}%"></div></div>
              <b>{pct(row["importance"])}</b>
            </div>
            """
        )
    return chr(10).join(rows)


def threshold_rows(thresholds: pd.DataFrame) -> str:
    rows = []
    for _, row in thresholds.head(10).iterrows():
        rows.append(
            f"""
            <tr>
              <td>{pct(row["enter_threshold"])} / {pct(row["exit_threshold"])}</td>
              <td>{number(row["sharpe_0rf"])}</td>
              <td>{pct(row["total_return"])}</td>
              <td>{pct(row["max_drawdown"])}</td>
              <td>{pct(row["exposure"])}</td>
              <td>{int(row["entries"])}</td>
            </tr>
            """
        )
    return chr(10).join(rows)


def feature_value(feature: str, value: float | int | str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if feature == "rsi_14":
        return number(numeric, 1)
    if feature == "day_of_week":
        return str(int(numeric))
    if any(token in feature for token in ["return", "volatility", "range", "body", "drawdown", "atr", "sma", "macd"]):
        return pct(numeric)
    if "ratio" in feature or "position" in feature:
        return number(numeric, 2)
    return number(numeric, 4)


def confidence_meter(latest: dict) -> str:
    prob = float(latest.get("probability_next_day_up", 0.0))
    entry = float(latest.get("entry_threshold", 0.55))
    exit_ = float(latest.get("exit_threshold", 0.50))
    label = html.escape(str(latest.get("confidence_label", latest.get("standalone_action", ""))))
    marker = max(0.0, min(100.0, prob * 100.0))
    entry_pos = max(0.0, min(100.0, entry * 100.0))
    exit_pos = max(0.0, min(100.0, exit_ * 100.0))
    return f"""
      <div class="meter-head"><strong>{label}</strong><span>{pct(prob)} probability up</span></div>
      <div class="meter">
        <div class="zone cash"></div><div class="zone neutral"></div><div class="zone long"></div>
        <i class="marker" style="left:{marker:.1f}%"></i>
        <i class="threshold exit" style="left:{exit_pos:.1f}%"></i>
        <i class="threshold entry" style="left:{entry_pos:.1f}%"></i>
      </div>
      <div class="meter-labels"><span>Cash below {pct(exit_)}</span><span>Long above {pct(entry)}</span></div>
    """


def feature_context_rows(items: list[dict]) -> str:
    if not items:
        return "<p class=\"sub\">Feature context will appear after the next model rebuild.</p>"
    rows = []
    for item in items[:8]:
        feature = str(item.get("feature", ""))
        label = html.escape(str(item.get("label", feature)))
        reading = html.escape(str(item.get("reading", "Normal")))
        value = feature_value(feature, item.get("value", ""))
        percentile = item.get("percentile", float("nan"))
        importance = item.get("importance", 0.0)
        note = html.escape(str(item.get("note", "")))
        rows.append(
            f"""
            <div class="context-row">
              <div><b>{label}</b><small>{note}</small></div>
              <span class="pill">{reading}</span>
              <span>{value}</span>
              <span>{pct(percentile)}</span>
              <span>{pct(importance)}</span>
            </div>
            """
        )
    return chr(10).join(rows)


def regime_card(regime: dict) -> str:
    if not regime:
        return "<p class=\"sub\">Regime data will appear after the next model rebuild.</p>"
    label = html.escape(str(regime.get("label", "Unknown")))
    tags = html.escape(str(regime.get("tags", "")))
    return f"""
      <div class="regime-badge">{label}</div>
      <p class="sub">{tags}</p>
      <div class="mini-stats">
        <div><span>Vol pctile</span><b>{pct(regime.get("volatility_percentile", 0))}</b></div>
        <div><span>Vol ratio</span><b>{number(regime.get("volume_ratio_30d", 0), 2)}</b></div>
        <div><span>50D trend</span><b>{pct(regime.get("close_vs_sma_50d", 0))}</b></div>
      </div>
    """


def trade_journal_rows(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<tr><td colspan=\"8\">Trade journal will appear after the next model rebuild.</td></tr>"
    rows = []
    for _, row in trades.tail(10).iloc[::-1].iterrows():
        exit_raw = row.get("exit_date", "")
        exit_date = "Open" if pd.isna(exit_raw) or str(exit_raw).strip() == "" else str(exit_raw)
        status = html.escape(str(row.get("status", "")))
        rows.append(
            f"""
            <tr>
              <td>{int(row["trade_id"])}</td>
              <td>{html.escape(str(row["entry_date"]))}</td>
              <td>{html.escape(exit_date)}</td>
              <td>{status}</td>
              <td>{int(row["holding_days"])}</td>
              <td>{pct(row["entry_probability"])}</td>
              <td>{pct(row["net_return_after_fees"])}</td>
              <td>{pct(row["max_drawdown"])}</td>
            </tr>
            """
        )
    return chr(10).join(rows)


def daily_archive_rows(archive: pd.DataFrame) -> str:
    if archive.empty:
        return "<tr><td colspan=\"8\">Daily archive will appear after the next model rebuild.</td></tr>"
    rows = []
    for _, row in archive.tail(14).iloc[::-1].iterrows():
        right_raw = row.get("model_was_right", False)
        if pd.isna(right_raw):
            right_bool = False
        elif isinstance(right_raw, str):
            right_bool = right_raw.strip().lower() in {"true", "1", "yes"}
        else:
            right_bool = bool(right_raw)
        right = "Yes" if right_bool else "No"
        rows.append(
            f"""
            <tr>
              <td>{html.escape(str(pd.Timestamp(row["Date"]).date()))}</td>
              <td>{money(row["Close"])}</td>
              <td>{pct(row["prob_up"])}</td>
              <td>{html.escape(str(row["standalone_signal"]))}</td>
              <td>{html.escape(str(row["confidence_label"]))}</td>
              <td>{html.escape(str(row["market_regime"]))}</td>
              <td>{html.escape(str(row["next_day_result"]))}</td>
              <td>{right}</td>
            </tr>
            """
        )
    return chr(10).join(rows)


def build_html() -> str:
    metrics = json.loads((OUTPUTS / "kaspa_rf_metrics.json").read_text(encoding="utf-8"))
    equity = pd.read_csv(OUTPUTS / "kaspa_rf_equity_curve.csv")
    features = pd.read_csv(OUTPUTS / "kaspa_rf_feature_importance.csv")
    thresholds = pd.read_csv(OUTPUTS / "kaspa_rf_threshold_sensitivity.csv")
    trade_path = OUTPUTS / "kaspa_rf_trade_journal.csv"
    archive_path = OUTPUTS / "kaspa_rf_daily_signal_archive.csv"
    trade_journal = pd.read_csv(trade_path) if trade_path.exists() else pd.DataFrame()
    daily_archive = pd.read_csv(archive_path) if archive_path.exists() else pd.DataFrame()
    status_path = OUTPUTS / "kaspa_data_status.json"
    data_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}

    sampled = sample_equity(equity)
    strategy_path = svg_path(
        sampled["strategy_equity"].astype(float).tolist(),
        x0=54,
        y0=28,
        width=820,
        height=270,
    )
    buyhold_path = svg_path(
        sampled["buyhold_equity"].astype(float).tolist(),
        x0=54,
        y0=28,
        width=820,
        height=270,
    )
    probability_path = svg_path(
        sampled["prob_up"].astype(float).tolist(),
        x0=54,
        y0=335,
        width=820,
        height=80,
    )

    latest = metrics["latest_signal"]
    latest_regime = metrics.get("latest_market_regime", {})
    latest_context = metrics.get("latest_feature_context", [])
    strategy = metrics["strategy"]
    buyhold = metrics["buy_and_hold"]
    classification = metrics["classification"]
    trades = metrics["trades"]
    hosting_mode = os.environ.get("KASPA_DASHBOARD_HOSTING_MODE", "local")
    generated = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    pages_notice = (
        f"Published by GitHub Actions. Data current through {metrics['data_end']}."
        if hosting_mode == "pages"
        else "Local mode. Start kaspa_dashboard_server.py to use the update controls."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KASPA Random Forest Dashboard</title>
<style>
:root {{
  --paper:#f4f2ea; --ink:#182320; --muted:#65716c; --panel:#fffffc;
  --line:#d9ddd5; --kas:#18a999; --blue:#2f6fb2; --amber:#d89019;
  --red:#c94d43; --coal:#243330;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; color:var(--ink); font-family:Aptos, "Segoe UI", Candara, sans-serif;
  background:
    radial-gradient(circle at 12% 8%, rgba(24,169,153,.16), transparent 28%),
    radial-gradient(circle at 88% 0%, rgba(216,144,25,.15), transparent 30%),
    linear-gradient(180deg, #fbfaf3, var(--paper));
}}
.shell {{ max-width:1240px; margin:0 auto; padding:28px; }}
.hero {{ display:grid; grid-template-columns:minmax(0,1fr) 290px; gap:18px; align-items:end; }}
.eyebrow {{ color:var(--muted); font-weight:800; font-size:13px; text-transform:uppercase; }}
h1 {{ margin:6px 0 8px; font-size:clamp(32px,5vw,58px); line-height:.96; }}
.lede {{ color:var(--muted); max-width:760px; line-height:1.55; margin:0; }}
.signal {{ background:var(--coal); color:white; border-radius:14px; padding:18px; box-shadow:0 18px 45px rgba(24,35,32,.13); }}
.signal span {{ color:rgba(255,255,255,.66); font-size:13px; }}
.signal strong {{ display:block; font-size:38px; margin:3px 0 4px; }}
.cards {{ display:grid; grid-template-columns:repeat(6,minmax(130px,1fr)); gap:12px; margin:18px 0; }}
.card,.panel {{ background:rgba(255,255,252,.94); border:1px solid var(--line); border-radius:14px; box-shadow:0 12px 32px rgba(24,35,32,.07); }}
.card {{ padding:14px; }}
.card span {{ display:block; color:var(--muted); font-size:12px; font-weight:800; }}
.card strong {{ display:block; font-size:24px; margin-top:5px; }}
.card small {{ display:block; color:var(--muted); margin-top:7px; line-height:1.35; }}
.grid {{ display:grid; grid-template-columns:minmax(0,1.45fr) minmax(310px,.8fr); gap:16px; align-items:start; }}
.decision-grid {{ display:grid; grid-template-columns:minmax(280px,.8fr) minmax(320px,1fr) minmax(340px,1.25fr); gap:16px; margin:16px 0; align-items:stretch; }}
.panel {{ padding:18px; }}
.panel h2 {{ margin:0 0 4px; font-size:20px; }}
.sub {{ margin:0 0 14px; color:var(--muted); font-size:13px; }}
svg {{ width:100%; height:auto; display:block; }}
.axis {{ stroke:var(--line); stroke-dasharray:3 4; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:13px; }}
.legend b {{ display:inline-block; width:22px; height:4px; border-radius:99px; margin-right:7px; vertical-align:middle; }}
.stack {{ display:grid; gap:16px; }}
.bar-row {{ display:grid; grid-template-columns:minmax(120px,1fr) minmax(120px,1.2fr) 52px; gap:10px; align-items:center; min-height:30px; font-size:13px; }}
.bar-row span {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.bar-track {{ height:9px; border-radius:99px; background:#e8ebe5; overflow:hidden; }}
.bar-track div {{ height:100%; background:linear-gradient(90deg,var(--kas),var(--amber)); }}
.matrix {{ display:grid; grid-template-columns:repeat(2,minmax(100px,1fr)); gap:8px; }}
.matrix div {{ border:1px solid var(--line); border-radius:12px; padding:12px; background:white; }}
.matrix span {{ color:var(--muted); font-size:12px; }}
.matrix b {{ display:block; font-size:26px; margin-top:2px; }}
.meter-head {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline; }}
.meter-head strong {{ font-size:26px; }}
.meter-head span {{ color:var(--muted); font-size:13px; }}
.meter {{ position:relative; display:grid; grid-template-columns:50fr 5fr 45fr; height:18px; margin:18px 0 8px; border-radius:999px; overflow:visible; background:#e8ebe5; }}
.zone {{ height:18px; }}
.zone.cash {{ background:linear-gradient(90deg,#d8e0dc,#b7c9c2); border-radius:999px 0 0 999px; }}
.zone.neutral {{ background:#e9d7ae; }}
.zone.long {{ background:linear-gradient(90deg,#7ed4c8,var(--kas)); border-radius:0 999px 999px 0; }}
.meter i {{ position:absolute; top:-5px; transform:translateX(-50%); display:block; }}
.meter .marker {{ width:4px; height:28px; border-radius:999px; background:var(--coal); box-shadow:0 0 0 3px rgba(255,255,252,.9); }}
.meter .threshold {{ width:2px; height:28px; border-left:2px dashed rgba(36,51,48,.55); }}
.meter-labels {{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px; }}
.regime-badge {{ display:inline-flex; align-items:center; min-height:52px; padding:0 16px; border-radius:16px; color:white; background:linear-gradient(135deg,var(--coal),var(--kas)); font-size:25px; font-weight:900; }}
.mini-stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
.mini-stats div {{ background:white; border:1px solid var(--line); border-radius:12px; padding:10px; }}
.mini-stats span {{ display:block; color:var(--muted); font-size:11px; font-weight:800; }}
.mini-stats b {{ display:block; margin-top:4px; font-size:18px; }}
.context-table {{ display:grid; gap:8px; }}
.context-row {{ display:grid; grid-template-columns:minmax(150px,1.5fr) 88px 70px 70px 72px; gap:9px; align-items:center; padding:9px; border:1px solid var(--line); background:white; border-radius:12px; font-size:12px; }}
.context-row b {{ display:block; font-size:13px; }}
.context-row small {{ display:block; color:var(--muted); margin-top:2px; line-height:1.25; }}
.pill {{ display:inline-flex; justify-content:center; border-radius:999px; padding:5px 8px; background:#eff2eb; font-weight:800; color:var(--coal); }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:9px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-size:12px; }}
.update {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:12px; align-items:center; margin:0 0 16px; }}
.update p {{ margin:0; color:var(--muted); font-size:13px; line-height:1.45; }}
.update button {{ border:1px solid var(--line); background:white; color:var(--coal); border-radius:10px; padding:8px 11px; cursor:pointer; }}
.log {{ display:none; grid-column:1/-1; padding:10px; background:#eef1eb; border-radius:10px; color:var(--coal); white-space:pre-wrap; font-size:12px; }}
.foot {{ color:var(--muted); font-size:12px; line-height:1.5; }}
@media (max-width:1100px) {{ .decision-grid {{ grid-template-columns:1fr; }} }}
@media (max-width:980px) {{ .hero,.grid {{ grid-template-columns:1fr; }} .cards {{ grid-template-columns:repeat(3,1fr); }} }}
@media (max-width:640px) {{ .shell {{ padding:18px; }} .cards {{ grid-template-columns:repeat(2,1fr); }} .update {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main class="shell">
  <section class="hero">
    <div>
      <div class="eyebrow">Kaspa Random Forest ensemble research</div>
      <h1>Long or cash?</h1>
      <p class="lede">Daily KAS-USD walk-forward Random Forest model with long/cash positioning, 30-day retraining, and 10 bps trading friction. Use as a research dashboard, not financial advice.</p>
    </div>
    <aside class="signal">
      <span>Latest standalone signal</span>
      <strong>{html.escape(str(latest["standalone_action"]))}</strong>
      <div>{latest["as_of_date"]} close {money(latest["close"])}. {html.escape(str(latest.get("confidence_label", "")))} at {pct(latest["probability_next_day_up"])} next-day up probability.</div>
    </aside>
  </section>

  <section class="cards">
    <div class="card"><span>Strategy return</span><strong>{pct(strategy["total_return"])}</strong><small>CAGR {pct(strategy["cagr"])}</small></div>
    <div class="card"><span>Buy and hold</span><strong>{pct(buyhold["total_return"])}</strong><small>CAGR {pct(buyhold["cagr"])}</small></div>
    <div class="card"><span>Strategy Sharpe</span><strong>{number(strategy["sharpe_0rf"])}</strong><small>0% risk-free</small></div>
    <div class="card"><span>Max drawdown</span><strong>{pct(strategy["max_drawdown"])}</strong><small>Buy/hold {pct(buyhold["max_drawdown"])}</small></div>
    <div class="card"><span>Exposure</span><strong>{pct(trades["exposure"])}</strong><small>{trades["entries"]} entries, {trades["exits"]} exits</small></div>
    <div class="card"><span>Model AUC</span><strong>{number(classification["auc"])}</strong><small>Accuracy {pct(classification["accuracy"])}</small></div>
  </section>

  <section class="panel update">
    <p id="updateText">{html.escape(pages_notice)} Last build {generated}. CoinGecko source: {html.escape(str(data_status.get("source", "coingecko_free")))}.</p>
    <div>
      <button id="checkBtn">Check</button>
      <button id="updateBtn">Update data</button>
    </div>
    <pre class="log" id="log"></pre>
  </section>

  <section class="decision-grid">
    <div class="panel">
      <h2>Signal confidence</h2>
      <p class="sub">Probability location relative to the long/cash thresholds.</p>
      {confidence_meter(latest)}
    </div>
    <div class="panel">
      <h2>Market regime</h2>
      <p class="sub">Latest trend, volatility, and liquidity context.</p>
      {regime_card(latest_regime)}
    </div>
    <div class="panel">
      <h2>Why this signal?</h2>
      <p class="sub">Top model context features. This is not SHAP attribution; it shows important inputs and their current historical percentile.</p>
      <div class="context-table">
        <div class="context-row" style="font-weight:800;color:var(--muted);background:transparent;border-style:dashed;">
          <span>Feature</span><span>Reading</span><span>Value</span><span>Pctile</span><span>Weight</span>
        </div>
        {feature_context_rows(latest_context)}
      </div>
    </div>
  </section>

  <section class="grid">
    <div class="panel">
      <h2>Equity and signal curve</h2>
      <p class="sub">Strategy vs buy-and-hold, with the model probability shown below.</p>
      <svg viewBox="0 0 930 440" role="img" aria-label="KASPA Random Forest backtest chart">
        <line class="axis" x1="54" y1="28" x2="54" y2="298"></line>
        <line class="axis" x1="54" y1="298" x2="874" y2="298"></line>
        <line class="axis" x1="54" y1="375" x2="874" y2="375"></line>
        <path d="{strategy_path}" fill="none" stroke="var(--kas)" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"></path>
        <path d="{buyhold_path}" fill="none" stroke="var(--blue)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" opacity=".86"></path>
        <path d="{probability_path}" fill="none" stroke="var(--amber)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"></path>
      </svg>
      <div class="legend">
        <span><b style="background:var(--kas)"></b>Random Forest strategy</span>
        <span><b style="background:var(--blue)"></b>Buy and hold</span>
        <span><b style="background:var(--amber)"></b>Probability up</span>
      </div>
    </div>

    <div class="stack">
      <div class="panel">
        <h2>Feature importance</h2>
        <p class="sub">Average split gain across walk-forward forests.</p>
        {bars(features)}
      </div>
      <div class="panel">
        <h2>Confusion matrix</h2>
        <p class="sub">Next-day direction classification.</p>
        <div class="matrix">
          <div><span>True up</span><b>{classification["true_positive"]}</b></div>
          <div><span>False up</span><b>{classification["false_positive"]}</b></div>
          <div><span>False cash</span><b>{classification["false_negative"]}</b></div>
          <div><span>True cash</span><b>{classification["true_negative"]}</b></div>
        </div>
      </div>
    </div>
  </section>

  <section class="panel" style="margin-top:16px; overflow-x:auto;">
    <h2>Threshold sensitivity</h2>
    <p class="sub">Top threshold pairs by Sharpe in the small grid search.</p>
    <table>
      <thead><tr><th>Entry / exit</th><th>Sharpe</th><th>Total return</th><th>Max DD</th><th>Exposure</th><th>Entries</th></tr></thead>
      <tbody>{threshold_rows(thresholds)}</tbody>
    </table>
  </section>

  <section class="grid" style="margin-top:16px;">
    <div class="panel" style="overflow-x:auto;">
      <h2>Trade journal</h2>
      <p class="sub">Most recent long/cash trades with estimated round-trip friction.</p>
      <table>
        <thead><tr><th>#</th><th>Entry</th><th>Exit</th><th>Status</th><th>Days</th><th>Entry prob</th><th>Net return</th><th>Max DD</th></tr></thead>
        <tbody>{trade_journal_rows(trade_journal)}</tbody>
      </table>
    </div>
    <div class="panel" style="overflow-x:auto;">
      <h2>Daily signal archive</h2>
      <p class="sub">Recent daily signals, confidence, regime, and next-day outcome.</p>
      <table>
        <thead><tr><th>Date</th><th>Close</th><th>Prob</th><th>Signal</th><th>Confidence</th><th>Regime</th><th>Next</th><th>Right?</th></tr></thead>
        <tbody>{daily_archive_rows(daily_archive)}</tbody>
      </table>
    </div>
  </section>

  <p class="foot">Research only. The model has a thin classification edge, so validate with out-of-sample data and broader market/context features before using any signal operationally.</p>
</main>

<script>
const hostingMode = {json.dumps(hosting_mode)};
const checkBtn = document.getElementById('checkBtn');
const updateBtn = document.getElementById('updateBtn');
const updateText = document.getElementById('updateText');
const log = document.getElementById('log');
function showLog(value) {{
  log.style.display = 'block';
  log.textContent = value;
}}
async function checkStatus() {{
  if (hostingMode === 'pages') {{
    checkBtn.style.display = 'none';
    updateBtn.style.display = 'none';
    return;
  }}
  try {{
    const res = await fetch('/api/status', {{ cache: 'no-store' }});
    const status = await res.json();
    updateText.textContent = `Local data through ${{status.last_data_date}}. Latest complete UTC candle: ${{status.latest_complete_utc_date}}. Needs update: ${{status.needs_update}}.`;
    showLog(JSON.stringify(status, null, 2));
  }} catch (error) {{
    updateText.textContent = 'Local update service is unavailable. Start python kaspa_dashboard_server.py, then reload.';
    showLog(String(error));
  }}
}}
async function runUpdate() {{
  updateBtn.disabled = true;
  try {{
    updateText.textContent = 'Updating data, retraining model, and rebuilding dashboard...';
    const res = await fetch('/api/update', {{ method: 'POST', cache: 'no-store' }});
    const result = await res.json();
    showLog(JSON.stringify(result, null, 2));
    if (result.ok) {{
      updateText.textContent = 'Dashboard updated. Reloading...';
      setTimeout(() => location.reload(), 900);
    }} else {{
      updateText.textContent = 'Update did not complete.';
    }}
  }} catch (error) {{
    updateText.textContent = 'Update failed.';
    showLog(String(error));
  }} finally {{
    updateBtn.disabled = false;
  }}
}}
checkBtn.addEventListener('click', checkStatus);
updateBtn.addEventListener('click', runUpdate);
checkStatus();
</script>
</body>
</html>
"""


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(build_html(), encoding="utf-8")
    print(DASHBOARD_PATH)


if __name__ == "__main__":
    main()
