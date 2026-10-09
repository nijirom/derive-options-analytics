#!/usr/bin/env python3
"""Build multi-currency Derive option surfaces and fair-value diagnostics.

Only public market-data endpoints are used. No wallet, API key, or session key
is read or required.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_BASES = {
    "mainnet": "https://api.derive.xyz/v3",
    "testnet": "https://testnet.api.derive.xyz/v3",
}
USER_AGENT = "derive-options-surface/2.0"


def public_request(base_url: str, method: str, params: dict[str, Any], timeout: float) -> Any:
    """Call one idempotent Derive public endpoint with bounded retries."""
    request = urllib.request.Request(
        f"{base_url}/{method}",
        data=json.dumps(params, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                envelope = json.load(response)
            if "error" in envelope:
                error = envelope["error"]
                raise RuntimeError(f"Derive API error {error.get('code')}: {error.get('message')} ({error.get('data')})")
            if "result" not in envelope:
                raise RuntimeError("Derive API returned neither a result nor an error")
            return envelope["result"]
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                detail = error.read(512).decode("utf-8", errors="replace")
                raise RuntimeError(f"Derive HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt == 2:
                break
        time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Derive request failed after 3 attempts: {last_error}")


def fetch_instruments(base_url: str, timeout: float) -> list[dict[str, Any]]:
    """Fetch every live-option definition, following Derive pagination."""
    instruments: list[dict[str, Any]] = []
    page = 1
    while True:
        result = public_request(
            base_url,
            "public/get_all_instruments",
            {"expired": False, "instrument_type": "option", "page": page, "page_size": 1000},
            timeout,
        )
        instruments.extend(result["instruments"])
        if page >= int(result["pagination"]["num_pages"]):
            return instruments
        page += 1


def fetch_tickers(
    base_url: str,
    requests: list[tuple[str, str]],
    timeout: float,
) -> dict[str, dict[str, Any]]:
    """Fetch option tickers in one batch per currency/expiry pair."""
    tickers: dict[str, dict[str, Any]] = {}

    def fetch(currency: str, expiry_date: str) -> dict[str, dict[str, Any]]:
        result = public_request(
            base_url,
            "public/get_tickers",
            {"instrument_type": "option", "currency": currency, "expiry_date": int(expiry_date)},
            timeout,
        )
        return result["tickers"]

    with ThreadPoolExecutor(max_workers=min(6, len(requests))) as executor:
        futures = {executor.submit(fetch, currency, expiry): (currency, expiry) for currency, expiry in requests}
        for future in as_completed(futures):
            currency, expiry = futures[future]
            try:
                tickers.update(future.result())
            except Exception as error:
                raise RuntimeError(f"Could not fetch {currency} tickers for {expiry}: {error}") from error
    return tickers


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def positive_float(value: Any) -> float | None:
    number = as_float(value)
    return number if number is not None and number > 0 else None


def deviation_percent(price: float | None, fair_value: float) -> float | None:
    return (price / fair_value - 1) * 100 if price is not None and fair_value > 0 else None

def solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Solve a 3x3 linear system with partial pivoting."""
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def fitted_iv(target: dict[str, Any], peers: list[dict[str, Any]]) -> float:
    """Leave-one-out local quadratic fit of mark IV over log-moneyness."""
    target_x = math.log(target["strike"] / target["forward"])
    neighbors = sorted(
        (peer for peer in peers if peer["name"] != target["name"]),
        key=lambda peer: abs(math.log(peer["strike"] / peer["forward"]) - target_x),
    )[:10]
    if len(neighbors) < 3:
        return target["markIv"]

    samples = [
        (math.log(peer["strike"] / peer["forward"]) - target_x, peer["markIv"])
        for peer in neighbors
    ]
    sums = [sum(dx**power for dx, _ in samples) for power in range(5)]
    matrix = [
        [sums[0], sums[1], sums[2]],
        [sums[1], sums[2], sums[3]],
        [sums[2], sums[3], sums[4]],
    ]
    vector = [sum(iv * dx**power for dx, iv in samples) for power in range(3)]
    coefficients = solve_3x3(matrix, vector)
    if coefficients is None:
        return target["markIv"]

    neighbor_ivs = [iv for _, iv in samples]
    lower = max(1.0, min(neighbor_ivs) * 0.5)
    upper = min(500.0, max(neighbor_ivs) * 1.5)
    return min(max(coefficients[0], lower), upper)


def black76_value(
    option_type: str,
    forward: float,
    strike: float,
    years: float,
    volatility_percent: float,
    discount_factor: float,
) -> float:
    """Black-76 option value using forward, discount factor, and annualized IV."""
    if years <= 0 or volatility_percent <= 0:
        intrinsic = max(forward - strike, 0) if option_type == "C" else max(strike - forward, 0)
        return discount_factor * intrinsic
    sigma_sqrt_t = volatility_percent / 100 * math.sqrt(years)
    d1 = (math.log(forward / strike) + 0.5 * sigma_sqrt_t**2) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    normal_cdf = lambda value: 0.5 * (1 + math.erf(value / math.sqrt(2)))
    if option_type == "C":
        value = discount_factor * (forward * normal_cdf(d1) - strike * normal_cdf(d2))
    else:
        value = discount_factor * (strike * normal_cdf(-d2) - forward * normal_cdf(-d1))
    return max(value, 0.0)


def apply_fair_values(records: list[dict[str, Any]]) -> None:
    """Attach smooth-smile fair values and observed-price deviations in place."""
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((row["currency"], row["expiry"], row["type"]), []).append(row)

    for row in records:
        peers = groups[(row["currency"], row["expiry"], row["type"])]
        fair_iv = fitted_iv(row, peers)
        fair_value = black76_value(
            row["type"],
            row["forward"],
            row["strike"],
            row["days"] / 365.25,
            fair_iv,
            row["discountFactor"],
        )
        row["fairIv"] = fair_iv
        row["fairValue"] = fair_value
        row["markDeviation"] = row["markPrice"] - fair_value
        row["markDeviationPct"] = deviation_percent(row["markPrice"], fair_value)
        for field in ("bid", "ask", "mid"):
            price = row[field]
            row[f"{field}Deviation"] = price - fair_value if price is not None else None
            row[f"{field}DeviationPct"] = deviation_percent(price, fair_value)




def build_records(
    instruments: list[dict[str, Any]],
    tickers: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join definitions and tickers, then calculate smooth-smile fair values."""
    records: list[dict[str, Any]] = []
    for instrument in instruments:
        if not instrument.get("is_active"):
            continue
        name = instrument["instrument_name"]
        ticker = tickers.get(name)
        details = instrument.get("option_details")
        pricing = ticker.get("option_pricing") if ticker else None
        if not ticker or not details or not pricing:
            continue

        strike = positive_float(details.get("strike"))
        forward = positive_float(pricing.get("f"))
        mark_iv = positive_float(pricing.get("i"))
        mark_price = as_float(pricing.get("m"))
        discount_factor = positive_float(pricing.get("df"))
        if None in (strike, forward, mark_iv, mark_price) or discount_factor is None:
            continue

        bid = positive_float(ticker.get("b"))
        ask = positive_float(ticker.get("a"))
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        expiry = int(details["expiry"])
        snapshot_ms = int(ticker["t"])
        records.append(
            {
                "currency": instrument["base_currency"],
                "name": name,
                "expiry": expiry,
                "expiryDate": datetime.fromtimestamp(expiry, timezone.utc).strftime("%Y-%m-%d"),
                "days": max((expiry - snapshot_ms / 1000) / 86400, 0),
                "strike": strike,
                "moneyness": strike / forward * 100,
                "type": details["option_type"],
                "forward": forward,
                "discountFactor": discount_factor,
                "index": float(ticker["I"]),
                "markIv": mark_iv * 100,
                "bidIv": (value * 100 if (value := positive_float(pricing.get("bi"))) else None),
                "askIv": (value * 100 if (value := positive_float(pricing.get("ai"))) else None),
                "markPrice": mark_price,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "openInterest": float(ticker["stats"]["oi"]),
                "volume24h": float(ticker["stats"]["c"]),
                "timestamp": snapshot_ms,
            }
        )
    apply_fair_values(records)
    records.sort(key=lambda row: (row["currency"], row["expiry"], row["strike"], row["type"]))
    return records


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Derive options surfaces and fair value</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { color-scheme:light; --canvas:#f4f4f1; --surface:#ffffff; --surface-alt:#f8f8f6; --ink:#1b1f23; --muted:#697078; --line:#d7d9d6; --line-strong:#b9bdb8; --navy:#214f70; --navy-dark:#17384f; --green:#39745a; --red:#a64b40; --amber:#9a6a24; }
    * { box-sizing:border-box; }
    html { background:var(--canvas); }
    body { margin:0; border-top:4px solid var(--navy-dark); background:var(--canvas); color:var(--ink); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }
    header { width:min(1480px,calc(100% - 48px)); margin:0 auto; padding:30px 0 22px; display:flex; justify-content:space-between; align-items:flex-end; gap:32px; border-bottom:1px solid var(--line-strong); }
    .header-copy { min-width:0; }
    .eyebrow { margin-bottom:7px; color:var(--navy); font:600 11px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace; letter-spacing:.12em; text-transform:uppercase; }
    h1 { margin:0; color:var(--ink); font-size:30px; font-weight:620; letter-spacing:-.035em; } h1 span { color:inherit; }
    .subtitle { margin-top:7px; color:var(--muted); font-size:13px; }
    .stamp { padding-left:22px; border-left:1px solid var(--line); color:var(--muted); text-align:right; font:11px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace; white-space:nowrap; }
    .controls { width:min(1480px,calc(100% - 48px)); margin:16px auto; padding:14px 16px; display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:14px; background:var(--surface); border:1px solid var(--line); }
    label { display:flex; flex-direction:column; gap:6px; color:var(--muted); font-size:10px; font-weight:650; text-transform:uppercase; letter-spacing:.09em; }
    select { width:100%; min-height:34px; margin:0; border:1px solid var(--line-strong); border-radius:2px; background:var(--surface); color:var(--ink); padding:6px 9px; font:500 13px/1.2 inherit; outline:none; }
    select:hover { border-color:#878d88; } select:focus { border-color:var(--navy); box-shadow:0 0 0 2px #214f7018; }
    .cards { width:min(1480px,calc(100% - 48px)); margin:0 auto 16px; display:grid; grid-template-columns:repeat(5,1fr); background:var(--surface); border:1px solid var(--line); }
    .card { min-width:0; padding:14px 16px; border-right:1px solid var(--line); } .card:last-child { border-right:0; }
    .card .k { color:var(--muted); font-size:10px; font-weight:650; text-transform:uppercase; letter-spacing:.08em; }
    .card .v { margin-top:5px; color:var(--ink); font-size:21px; font-weight:620; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
    main { width:min(1480px,calc(100% - 48px)); margin:0 auto; padding:0 0 40px; display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .panel { min-width:0; padding:8px; overflow:hidden; background:var(--surface); border:1px solid var(--line); } .wide { grid-column:1/-1; }
    #surface { height:500px; } #term,#smiles,#fairValue,#deviation { height:350px; }
    .surface-panel { background:#0c1525; border-color:#243149; border-radius:4px; }
    .surface-panel .note { color:#91a4bd; }
    .surface-panel .modebar-btn path { fill:#71849d!important; }
    .js-plotly-plot,.plot-container,.svg-container { max-width:100%!important; }
    .chain-panel { padding:0; }
    .table-caption { display:flex; justify-content:space-between; gap:16px; padding:11px 12px; border-bottom:1px solid var(--line); color:#343a40; font-size:12px; font-weight:600; }
    .table-caption .hint { color:var(--muted); font-size:10px; font-weight:500; letter-spacing:.02em; }
    .table-wrap { max-height:460px; overflow:auto; }
    table { width:100%; border-collapse:collapse; color:var(--ink); font-size:12px; font-variant-numeric:tabular-nums; }
    th { position:sticky; top:0; z-index:1; background:#eceeeb; color:#555c63; text-align:right; font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }
    th,td { padding:9px 10px; border-bottom:1px solid #e4e6e3; white-space:nowrap; text-align:right; }
    th:first-child,td:first-child { text-align:left; } tbody tr:nth-child(even) { background:#fafaf8; } tbody tr:hover { background:#eef3f6; }
    .positive { color:var(--green); } .negative { color:var(--red); }
    .note { min-height:36px; padding:3px 10px 9px; color:var(--muted); font-size:11px; line-height:1.45; }
    @media (max-width:1000px) { .controls { grid-template-columns:repeat(3,1fr); } .cards { grid-template-columns:repeat(2,1fr); } .card { border-bottom:1px solid var(--line); } main { grid-template-columns:1fr; } .wide { grid-column:auto; } }
    @media (max-width:650px) { body { border-top-width:3px; overflow-x:hidden; } header { align-items:flex-start; flex-direction:column; padding-top:22px; } .stamp { max-width:100%; padding:0; border:0; text-align:left; white-space:normal; overflow-wrap:anywhere; word-break:break-word; } header,.controls,.cards,main { width:calc(100vw - 24px); max-width:calc(100vw - 24px); } .controls { grid-template-columns:1fr; } label,select,.card,.panel { min-width:0; max-width:100%; } .cards { grid-template-columns:1fr 1fr; } .card .k { overflow-wrap:anywhere; } .card:nth-child(5) { grid-column:1/-1; border-right:0; } h1 { font-size:25px; } #surface { height:400px; } #term,#smiles,#fairValue,#deviation { height:310px; } }
    @media (max-width:480px) { .cards { grid-template-columns:1fr; } .card { border-right:0; } .card:nth-child(5) { grid-column:auto; } }
  </style>
</head>
<body>
  <header><div class="header-copy"><div class="eyebrow">Derive options monitor</div><h1><span id="currencyName"></span> volatility &amp; valuation</h1><div class="subtitle">Cross-sectional volatility, model fair value, and market dislocation</div></div><div class="stamp" id="stamp"></div></header>
  <section class="controls">
    <label>Currency<select id="currency"></select></label>
    <label>Vol source<select id="source"><option value="markIv">Mark IV</option><option value="bidIv">Bid IV</option><option value="askIv">Ask IV</option></select></label>
    <label>Contracts<select id="contract"><option value="otm">OTM composite</option><option value="C">Calls</option><option value="P">Puts</option></select></label>
    <label>Moneyness<select id="range"><option value="60,140">60–140%</option><option value="75,125" selected>75–125%</option><option value="90,110">90–110%</option><option value="40,180">40–180%</option></select></label>
    <label>Valuation expiry<select id="valuationExpiry"></select></label>
  </section>
  <section class="cards">
    <div class="card"><div class="k">Index</div><div class="v" id="index"></div></div>
    <div class="card"><div class="k">ATM IV · nearest expiry</div><div class="v" id="atm"></div></div>
    <div class="card"><div class="k">Active expiries</div><div class="v" id="expiries"></div></div>
    <div class="card"><div class="k">Surface points</div><div class="v" id="points"></div></div>
    <div class="card"><div class="k">Options with live quotes</div><div class="v" id="quotes"></div></div>
  </section>
  <main>
    <section class="panel wide surface-panel"><div id="surface"></div><div class="note">Strike/forward moneyness normalizes each expiry. Empty cells are outside that smile's available range.</div></section>
    <section class="panel"><div id="smiles"></div></section>
    <section class="panel"><div id="term"></div></section>
    <section class="panel"><div id="fairValue"></div><div class="note">Fair value uses Black-76 with a leave-one-out local quadratic fit of neighboring Derive mark IVs. Derive mark and live quotes remain separate observations.</div></section>
    <section class="panel"><div id="deviation"></div><div class="note">Deviation is observed price minus fitted fair value. Bid/ask points appear only when live quotes exist; missing quotes are never treated as zero.</div></section>
    <section class="panel wide chain-panel"><div class="table-caption"><span>Selected expiry option chain</span><span class="hint">Scroll horizontally for all fields</span></div><div class="table-wrap"><table><thead><tr><th>Instrument</th><th>Days</th><th>Moneyness</th><th>Mark IV</th><th>Fair IV</th><th>Mark</th><th>Fair value</th><th>Mark dev</th><th>Bid</th><th>Bid dev</th><th>Ask</th><th>Ask dev</th><th>OI</th></tr></thead><tbody id="rows"></tbody></table></div></section>
  </main>
<script>
const DATA = __PAYLOAD__;
const COLORS = ['#214f70','#9a6a24','#39745a','#7d556f','#50796f','#8b5f46','#536f8a','#7b744a','#48667a','#8a625d','#5f6a72','#6f604e','#3f725f','#72566f'];
const SURFACE_COLORS=[[0,'#440154'],[.13,'#482878'],[.25,'#3e4989'],[.38,'#31688e'],[.5,'#26828e'],[.63,'#1f9e89'],[.75,'#35b779'],[.86,'#6ece58'],[.94,'#b5de2b'],[1,'#fde725']];
const axisStyle={gridcolor:'#e3e5e2',zerolinecolor:'#aeb4b0',linecolor:'#c5c9c5',tickcolor:'#c5c9c5',showline:true,linewidth:1,automargin:true};
const baseLayout={paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',font:{color:'#31363b',family:'-apple-system,BlinkMacSystemFont,\"Segoe UI\",Arial,sans-serif',size:12},title:{font:{size:14,color:'#24292e'},y:.96},margin:{l:62,r:24,t:52,b:52},legend:{orientation:'h',y:-.2,font:{size:11,color:'#555c63'}},xaxis:{...axisStyle},yaxis:{...axisStyle},hoverlabel:{bgcolor:'#1f252a',bordercolor:'#1f252a',font:{color:'#ffffff',size:12}}};
const config = {responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};
const compact=window.matchMedia('(max-width:650px)').matches;
const compactMargin={l:46,r:10,t:46,b:44};
const fmt = (x,d=1) => x == null ? '—' : Number(x).toFixed(d);
const pct = (x,d=1) => x == null ? '—' : Number(x).toFixed(d)+'%';
const tone = value => value == null ? '' : (value >= 0 ? 'positive' : 'negative');
const price = x => x == null ? '—' : Number(x).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:6});
const groupBy = (rows,key) => rows.reduce((groups,row)=>{ const value=row[key]; (groups[value]??=[]).push(row); return groups; },{});
const currencySelect=document.querySelector('#currency'), expirySelect=document.querySelector('#valuationExpiry');
for (const currency of DATA.currencies) currencySelect.add(new Option(currency,currency));
const requestedCurrency=new URLSearchParams(location.search).get('currency')?.toUpperCase();
currencySelect.value=DATA.currencies.includes(requestedCurrency)?requestedCurrency:(DATA.currencies.includes('ETH')?'ETH':DATA.currencies[0]);
function currencyRows() { return DATA.records.filter(r=>r.currency===currencySelect.value); }
function eligible(r) { const contract=document.querySelector('#contract').value; return contract==='otm' ? ((r.moneyness<=100&&r.type==='P')||(r.moneyness>100&&r.type==='C')) : r.type===contract; }
function rangeBounds() { return document.querySelector('#range').value.split(',').map(Number); }
function chosen() { const source=document.querySelector('#source').value,[lo,hi]=rangeBounds(); return currencyRows().filter(r=>r[source]!=null&&r.moneyness>=lo&&r.moneyness<=hi&&eligible(r)); }
function syncExpiries() { const previous=expirySelect.value, rows=currencyRows(), expiries=[...new Set(rows.map(r=>r.expiryDate))].sort(), preferred=expiries.find(expiry=>rows.some(r=>r.expiryDate===expiry&&r.days>=7))??expiries[0]; expirySelect.replaceChildren(...expiries.map(value=>new Option(value,value))); expirySelect.value=expiries.includes(previous)?previous:preferred; }
function interpolate(points,x,source) { if (!points.length||x<points[0].moneyness||x>points[points.length-1].moneyness) return null; for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i];if(x<=b.moneyness){const width=b.moneyness-a.moneyness;return width===0?(a[source]+b[source])/2:a[source]+(b[source]-a[source])*(x-a.moneyness)/width;}} return points.at(-1)[source]; }
function emptyLayout(title,message) { return {...baseLayout,title:{text:title,x:.04},annotations:[{text:message,x:.5,y:.5,xref:'paper',yref:'paper',showarrow:false,font:{color:'#697078',size:13}}]}; }
function renderVolatility() {
  const currency=currencySelect.value,source=document.querySelector('#source').value,rows=chosen(),groups=groupBy(rows,'expiryDate'),expiries=Object.keys(groups).sort();
  const [lo,hi]=rangeBounds(),step=(hi-lo)/40,xs=Array.from({length:41},(_,i)=>lo+i*step);
  if (!rows.length) {
    Plotly.react('surface',[],emptyLayout(`${currency} volatility surface`,'No IV observations for this selection'),config);
    Plotly.react('smiles',[],emptyLayout('Volatility smiles','No IV observations for this selection'),config);
    Plotly.react('term',[],emptyLayout('ATM term structure','No IV observations for this selection'),config);
    document.querySelector('#atm').textContent='—'; document.querySelector('#points').textContent='0'; return;
  }
  const z=[],ys=[];
  for(const expiry of expiries){const points=groups[expiry].sort((a,b)=>a.moneyness-b.moneyness);z.push(xs.map(x=>interpolate(points,x,source)));ys.push(points[0].days);}
  const surfaceData=[{
    type:'surface',x:xs,y:ys,z,connectgaps:false,colorscale:SURFACE_COLORS,
    colorbar:{title:{text:'IV %',font:{size:12,color:'#d7e0ea'}},thickness:12,outlinewidth:0,tickfont:{size:10,color:'#d7e0ea'},tickformat:'.0f'},
    customdata:z.map((row,i)=>row.map(()=>expiries[i])),hovertemplate:'Expiry %{customdata}<br>DTE %{y:.1f}<br>Moneyness %{x:.1f}%<br>IV %{z:.2f}%<extra></extra>',
    contours:{z:{show:true,usecolormap:true,highlightcolor:'#d8fff7',project:{z:true}}},
    lighting:{ambient:.72,diffuse:.82,roughness:.7,specular:.16,fresnel:.08},lightposition:{x:100,y:-120,z:180}
  }];
  Plotly.react('surface',surfaceData,{...baseLayout,paper_bgcolor:'#0c1525',plot_bgcolor:'#0c1525',font:{...baseLayout.font,color:'#d7e0ea'},title:{text:`${currency} ${document.querySelector('#source').selectedOptions[0].text} surface`,x:.035,font:{size:14,color:'#d7e0ea'}},scene:{bgcolor:'#0c1525',camera:{eye:{x:-1.38,y:1.58,z:1.02}},aspectmode:'manual',aspectratio:{x:1.45,y:1,z:.8},xaxis:{title:'Strike / forward (%)',color:'#d7e0ea',gridcolor:'#223249',zerolinecolor:'#3a4a61',backgroundcolor:'#0c1525',showbackground:true},yaxis:{title:'Days to expiry',color:'#d7e0ea',gridcolor:'#223249',zerolinecolor:'#3a4a61',backgroundcolor:'#0c1525',showbackground:true},zaxis:{title:'Implied volatility (%)',color:'#d7e0ea',gridcolor:'#223249',zerolinecolor:'#3a4a61',backgroundcolor:'#0c1525',showbackground:true}},margin:compact?{l:0,r:0,t:48,b:0}:{l:8,r:26,t:52,b:8}},config);
  const smileTraces=expiries.map((expiry,i)=>({type:'scatter',mode:'lines+markers',name:expiry,x:groups[expiry].map(r=>r.moneyness),y:groups[expiry].map(r=>r[source]),line:{color:COLORS[i%COLORS.length],width:1.35},marker:{size:3,color:COLORS[i%COLORS.length]},hovertemplate:'%{x:.1f}% moneyness<br>%{y:.2f}% IV<extra>'+expiry+'</extra>'}));
  Plotly.react('smiles',smileTraces,{...baseLayout,title:{text:'Volatility smiles',x:.04},xaxis:{...baseLayout.xaxis,title:'Moneyness (%)'},yaxis:{...baseLayout.yaxis,title:'IV (%)'},showlegend:false,margin:compact?compactMargin:{l:58,r:18,t:50,b:48}},config);
  const atm=expiries.map(expiry=>groups[expiry].reduce((best,r)=>Math.abs(r.moneyness-100)<Math.abs(best.moneyness-100)?r:best));
  Plotly.react('term',[{type:'scatter',mode:'lines+markers',x:atm.map(r=>r.days),y:atm.map(r=>r[source]),text:atm.map(r=>r.expiryDate),line:{color:'#214f70',width:2},marker:{color:'#ffffff',line:{color:'#214f70',width:1.5},size:6},hovertemplate:'%{text}<br>DTE %{x:.1f}<br>ATM IV %{y:.2f}%<extra></extra>'}],{...baseLayout,title:{text:'ATM term structure',x:.04},xaxis:{...baseLayout.xaxis,title:'Days to expiry'},yaxis:{...baseLayout.yaxis,title:'IV (%)'},margin:compact?compactMargin:{l:58,r:18,t:50,b:48}},config);
  document.querySelector('#atm').textContent=pct(atm[0][source],2); document.querySelector('#points').textContent=rows.length;
}
function renderValuation() {
  const currency=currencySelect.value,[lo,hi]=rangeBounds(),expiry=expirySelect.value;
  const rows=currencyRows().filter(r=>r.expiryDate===expiry&&r.moneyness>=lo&&r.moneyness<=hi&&eligible(r)).sort((a,b)=>a.strike-b.strike);
  const fairTrace={type:'scatter',mode:'lines',name:'Fitted fair',x:rows.map(r=>r.strike),y:rows.map(r=>r.fairValue),line:{color:'#17384f',width:2.25},customdata:rows.map(r=>[r.name,r.moneyness,r.fairIv]),hovertemplate:'%{customdata[0]}<br>Strike %{x}<br>Fair %{y:.6f}<br>Fair IV %{customdata[2]:.2f}%<br>Moneyness %{customdata[1]:.1f}%<extra></extra>'};
  const markTrace={type:'scatter',mode:'markers',name:'Derive mark',x:rows.map(r=>r.strike),y:rows.map(r=>r.markPrice),marker:{color:'#9a6a24',size:6,line:{color:'#ffffff',width:.75}},customdata:rows.map(r=>r.name),hovertemplate:'%{customdata}<br>Strike %{x}<br>Mark %{y:.6f}<extra></extra>'};
  const bidTrace={type:'scatter',mode:'markers',name:'Bid',x:rows.filter(r=>r.bid!=null).map(r=>r.strike),y:rows.filter(r=>r.bid!=null).map(r=>r.bid),marker:{color:'#a64b40',symbol:'triangle-down',size:7}};
  const askTrace={type:'scatter',mode:'markers',name:'Ask',x:rows.filter(r=>r.ask!=null).map(r=>r.strike),y:rows.filter(r=>r.ask!=null).map(r=>r.ask),marker:{color:'#39745a',symbol:'triangle-up',size:7}};
  const valueLayout={...baseLayout,title:{text:`Fair value by strike · ${expiry}`,x:.04},xaxis:{...baseLayout.xaxis,title:'Strike'},yaxis:{...baseLayout.yaxis,title:'Option value'},margin:compact?compactMargin:{l:66,r:18,t:50,b:52}};
  if (!rows.length) valueLayout.annotations=emptyLayout('', 'No options for this selection').annotations;
  Plotly.react('fairValue',[fairTrace,markTrace,bidTrace,askTrace],valueLayout,config);
  const deviationTraces=[
    {type:'bar',name:'Mark deviation',x:rows.map(r=>r.strike),y:rows.map(r=>r.markDeviation),marker:{color:rows.map(r=>r.markDeviation>=0?'#5f806d':'#b35d52'),line:{width:0}},opacity:.72,customdata:rows.map(r=>[r.name,r.markDeviationPct]),hovertemplate:'%{customdata[0]}<br>Deviation %{y:.6f}<br>%{customdata[1]:.2f}%<extra></extra>'},
    {type:'scatter',mode:'markers',name:'Bid deviation',x:rows.filter(r=>r.bidDeviation!=null).map(r=>r.strike),y:rows.filter(r=>r.bidDeviation!=null).map(r=>r.bidDeviation),marker:{color:'#a64b40',symbol:'triangle-down',size:7}},
    {type:'scatter',mode:'markers',name:'Ask deviation',x:rows.filter(r=>r.askDeviation!=null).map(r=>r.strike),y:rows.filter(r=>r.askDeviation!=null).map(r=>r.askDeviation),marker:{color:'#39745a',symbol:'triangle-up',size:7}}
  ];
  const deviationLayout={...baseLayout,title:{text:`Price deviation from fitted fair · ${expiry}`,x:.04},xaxis:{...baseLayout.xaxis,title:'Strike'},yaxis:{...baseLayout.yaxis,title:'Observed − fair',zeroline:true,zerolinewidth:1.5},margin:compact?compactMargin:{l:66,r:18,t:50,b:52},barmode:'overlay'};
  Plotly.react('deviation',deviationTraces,deviationLayout,config);
}
function renderTable() {
  const rows=chosen().filter(r=>r.expiryDate===expirySelect.value).sort((a,b)=>a.strike-b.strike);
  document.querySelector('#rows').innerHTML=rows.map(r=>`<tr><td>${r.name}</td><td>${fmt(r.days,1)}</td><td>${pct(r.moneyness,1)}</td><td>${pct(r.markIv,2)}</td><td>${pct(r.fairIv,2)}</td><td>${price(r.markPrice)}</td><td>${price(r.fairValue)}</td><td class="${tone(r.markDeviationPct)}">${pct(r.markDeviationPct,2)}</td><td>${price(r.bid)}</td><td class="${tone(r.bidDeviationPct)}">${pct(r.bidDeviationPct,2)}</td><td>${price(r.ask)}</td><td class="${tone(r.askDeviationPct)}">${pct(r.askDeviationPct,2)}</td><td>${fmt(r.openInterest,1)}</td></tr>`).join('');
}
function render() {
  const rows=currencyRows(),nearest=rows.reduce((best,r)=>!best||r.days<best.days?r:best,null),expiries=new Set(rows.map(r=>r.expiryDate));
  document.querySelector('#currencyName').textContent=currencySelect.value;
  requestAnimationFrame(()=>document.querySelectorAll('.js-plotly-plot').forEach(plot=>Plotly.Plots.resize(plot)));
  document.querySelector('#index').textContent=nearest?'$'+nearest.index.toLocaleString(undefined,{maximumFractionDigits:6}):'—';
  document.querySelector('#expiries').textContent=expiries.size;
  document.querySelector('#quotes').textContent=rows.filter(r=>r.bid!=null||r.ask!=null).length;
  renderVolatility(); renderValuation(); renderTable();
}
document.querySelector('#stamp').innerHTML=`Snapshot ${new Date(DATA.timestamp).toLocaleString()}<br>${DATA.environment} · ${DATA.currencies.length} option currencies · public data · no credential used`;
currencySelect.addEventListener('change',()=>{syncExpiries();render();});
document.querySelectorAll('#source,#contract,#range,#valuationExpiry').forEach(el=>el.addEventListener('change',render));
syncExpiries(); render();
</script>
</body>
</html>'''


def render_html(environment: str, records: list[dict[str, Any]]) -> str:
    payload = {
        "environment": environment,
        "timestamp": max(row["timestamp"] for row in records),
        "currencies": sorted({row["currency"] for row in records}),
        "records": records,
    }
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__PAYLOAD__", encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create multi-currency Derive option surfaces and fair-value diagnostics.")
    parser.add_argument("--currencies", default="ALL", help="Comma-separated option currencies, or ALL (default)")
    parser.add_argument("--environment", choices=API_BASES, default="mainnet")
    parser.add_argument("--output", type=Path, default=Path("derive_all_options_surface.html"))
    parser.add_argument("--max-expiries", type=int, default=0, help="Nearest N expiries per currency; 0 keeps all")
    parser.add_argument("--timeout", type=float, default=20, help="Per-request timeout in seconds")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in the default browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or args.max_expiries < 0:
        print("error: timeout must be positive and max-expiries non-negative", file=sys.stderr)
        return 2

    base_url = API_BASES[args.environment]
    instruments = fetch_instruments(base_url, args.timeout)
    active = [item for item in instruments if item.get("is_active") and item.get("option_details")]
    available = sorted({item["base_currency"] for item in active})
    requested_text = args.currencies.strip().upper()
    requested = available if requested_text == "ALL" else sorted({value.strip() for value in requested_text.split(",") if value.strip()})
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise RuntimeError(f"No active options for {', '.join(unknown)}. Available: {', '.join(available)}")

    selected_instruments: list[dict[str, Any]] = []
    ticker_requests: list[tuple[str, str]] = []
    expiry_count = 0
    for currency in requested:
        currency_instruments = [item for item in active if item["base_currency"] == currency]
        expiries = sorted({int(item["option_details"]["expiry"]) for item in currency_instruments})
        if args.max_expiries:
            expiries = expiries[: args.max_expiries]
        expiry_set = set(expiries)
        selected_instruments.extend(item for item in currency_instruments if int(item["option_details"]["expiry"]) in expiry_set)
        ticker_requests.extend(
            (currency, datetime.fromtimestamp(expiry, timezone.utc).strftime("%Y%m%d")) for expiry in expiries
        )
        expiry_count += len(expiries)

    if not ticker_requests:
        raise RuntimeError("No active option expiries found")
    tickers = fetch_tickers(base_url, ticker_requests, args.timeout)
    records = build_records(selected_instruments, tickers)
    if not records:
        raise RuntimeError("No priced options found for the selected currencies")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(args.environment, records), encoding="utf-8")
    print(
        f"Wrote {args.output.resolve()} with {len(records)} options, "
        f"{len(requested)} currencies, and {expiry_count} currency/expiry groups."
    )
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
