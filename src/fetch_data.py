"""Fetch ~20y of daily data from the Yahoo Finance chart API (no key needed).

Saves one CSV per symbol (Date, AdjClose) plus a combined wide CSV of adjusted
closes into ../data. Uses a browser User-Agent; retries a couple of times.
"""
import json
import time
import urllib.request
import pathlib
import sys

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

# Long-history, liquid instruments. ^GSPC is the primary series for the study.
SYMBOLS = {
    "^GSPC": "SP500",
    "^IXIC": "NASDAQ",
    "^RUT": "RUSSELL2000",
    "^FTSE": "FTSE100",
    "^N225": "NIKKEI225",
}

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def fetch(symbol, rng="20y", interval="1d", retries=3):
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}?range={rng}&interval={interval}"
    )
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {a+1} failed for {symbol}: {e}", file=sys.stderr)
            time.sleep(2)
    return None


def parse(js):
    res = js["chart"]["result"][0]
    ts = res["timestamp"]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    close = res["indicators"]["quote"][0]["close"]
    series = adj if adj else close
    out = []
    import datetime as dt
    for t, c in zip(ts, series):
        if c is None:
            continue
        d = dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")
        out.append((d, c))
    return out


def main():
    import pandas as pd
    frames = {}
    for sym, name in SYMBOLS.items():
        print(f"Fetching {sym} ({name}) ...")
        js = fetch(sym)
        if js is None:
            print(f"  SKIP {sym}")
            continue
        rows = parse(js)
        df = pd.DataFrame(rows, columns=["Date", "AdjClose"]).drop_duplicates("Date")
        df.to_csv(DATA / f"{name}.csv", index=False)
        s = df.set_index("Date")["AdjClose"].astype(float)
        s.name = name
        frames[name] = s
        print(f"  {name}: {len(s)} rows, {s.index.min()} -> {s.index.max()}")
        time.sleep(1)
    if frames:
        wide = pd.concat(frames.values(), axis=1).sort_index()
        wide.to_csv(DATA / "prices_wide.csv")
        print(f"Combined: {wide.shape[0]} rows x {wide.shape[1]} cols -> data/prices_wide.csv")


if __name__ == "__main__":
    import urllib.parse  # noqa: E402
    main()
