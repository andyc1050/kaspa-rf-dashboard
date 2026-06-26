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


def build_html() -> str:
    metrics = json.loads((OUTPUTS / "kaspa_rf_metrics.json").read_text(encoding="utf-8"))
    equity = pd.read_csv(OUTPUTS / "kaspa_rf_equity_curve.csv")
    features = pd.read_csv(OUTPUTS / "kaspa_rf_feature_importance.csv")
    thresholds = pd.read_csv(OUTPUTS / "kaspa_rf_threshold_sensitivity.csv")
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
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:9px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-size:12px; }}
.update {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:12px; align-items:center; margin:0 0 16px; }}
.update p {{ margin:0; color:var(--muted); font-size:13px; line-height:1.45; }}
.update button {{ border:1px solid var(--line); background:white; color:var(--coal); border-radius:10px; padding:8px 11px; cursor:pointer; }}
.log {{ display:none; grid-column:1/-1; padding:10px; background:#eef1eb; border-radius:10px; color:var(--coal); white-space:pre-wrap; font-size:12px; }}
.foot {{ color:var(--muted); font-size:12px; line-height:1.5; }}
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
      <div>{latest["as_of_date"]} close {money(latest["close"])}. Next-day up probability {pct(latest["probability_next_day_up"])}.</div>
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
