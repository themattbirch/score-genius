# backend/data_pipeline/refresh_odds.py
"""
Refresh latest betting odds for MLB / NBA / NFL and upsert ONLY the odds fields into
their respective schedule tables.

Updated fields per sport:

MLB (table: mlb_game_schedule)
- moneyline, spread, total (raw dicts)
- moneyline_home_clean, moneyline_away_clean
- spread_home_line_clean, spread_home_price_clean, spread_away_price_clean
- total_line_clean, total_over_price_clean, total_under_price_clean

NBA (table: nba_game_schedule)
- moneyline, spread, total (raw dicts)
- moneyline_clean  ({"home": str|None, "away": str|None})
- spread_clean     ({"home":{"line":float|None,"price":str|None}, "away":{"line":float|None,"price":str|None}})
- total_clean      ({"line": float|None, "over_price": str|None, "under_price": str|None})

NFL (table: nfl_game_schedule)
- moneyline, spread, total (raw dicts)
- moneyline_clean, spread_clean, total_clean (same shapes as NBA)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import requests

# ──────────────────────────────────────────────────────────────────────────────
# Supabase client (service key required)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from supabase import create_client, Client
except ImportError as e:
    print(f"FATAL: supabase client missing: {e}")
    sys.exit(1)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ──────────────────────────────────────────────────────────────────────────────
# Env / Config
# ──────────────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

missing = [n for n, v in [("SUPABASE_URL", SUPABASE_URL),
                          ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
                          ("ODDS_API_KEY", ODDS_API_KEY)] if not v]
if missing:
    print(f"FATAL: missing env vars: {', '.join(missing)}")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

PREFERRED_BOOK = "draftkings"

# ──────────────────────────────────────────────────────────────────────────────
# Sport-specific config
# ──────────────────────────────────────────────────────────────────────────────
def _norm_generic(name: str) -> str:
    return " ".join((name or "").split()).strip().lower()

def _title(s: str) -> str:
    return " ".join(w.capitalize() for w in (s or "").split())

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _fmt_american(n: Optional[float]) -> Optional[str]:
    if n is None:
        return None
    v = int(round(n))
    return f"+{v}" if v > 0 else f"{v}"

def _norm_mlb(name: str) -> str:
    if not name:
        return ""
    s = name.strip()
    s = s.replace("St.Louis", "St Louis").replace("St.", "St").lower().strip()
    if s in {"ath", "athletics", "oakland athletics"}:
        return "athletics"
    if "st louis" in s:
        return "st. louis cardinals"
    mapping = {
        "bal":"baltimore orioles","bos":"boston red sox","nyy":"new york yankees",
        "tbr":"tampa bay rays","tor":"toronto blue jays","chw":"chicago white sox",
        "cle":"cleveland guardians","det":"detroit tigers","kcr":"kansas city royals",
        "min":"minnesota twins","hou":"houston astros","laa":"los angeles angels",
        "sea":"seattle mariners","tex":"texas rangers","atl":"atlanta braves",
        "mia":"miami marlins","nym":"new york mets","phi":"philadelphia phillies",
        "wsn":"washington nationals","chc":"chicago cubs","cin":"cincinnati reds",
        "mil":"milwaukee brewers","pit":"pittsburgh pirates","ari":"arizona diamondbacks",
        "col":"colorado rockies","lad":"los angeles dodgers","sdp":"san diego padres",
        "sfg":"san francisco giants","stl":"st. louis cardinals",
    }
    return mapping.get(s, s).strip()

def _norm_nfl(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    mapping = {
        "nyg":"new york giants","nyj":"new york jets","ne":"new england patriots",
        "gb":"green bay packers","kc":"kansas city chiefs","sf":"san francisco 49ers",
        "la":"los angeles rams","lac":"los angeles chargers","dal":"dallas cowboys",
        "phi":"philadelphia eagles","sea":"seattle seahawks","pit":"pittsburgh steelers",
        "den":"denver broncos","lv":"las vegas raiders","oak":"las vegas raiders",
        "ari":"arizona cardinals","min":"minnesota vikings","ten":"tennessee titans",
        "cin":"cincinnati bengals","bal":"baltimore ravens","hou":"houston texans",
        "jax":"jacksonville jaguars","atl":"atlanta falcons","car":"carolina panthers",
        "tb":"tampa bay buccaneers","no":"new orleans saints","chi":"chicago bears",
        "det":"detroit lions","ind":"indianapolis colts","wash":"washington commanders",
    }
    return mapping.get(s, s).strip()

SPORTS: Dict[str, Dict[str, Any]] = {
    "mlb": {
        "table": "mlb_game_schedule",
        "date_column": "game_date_et",
        "home_col": "home_team_name",
        "away_col": "away_team_name",
        "time_col": "scheduled_time_utc",
        "odds_api": "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
        "normalize": _norm_mlb,
        "raw_fields": ["moneyline", "spread", "total"],
        "clean_mode": "mlb_columns",  # individual columns
    },
    "nba": {
        "table": "nba_game_schedule",
        "date_column": "game_date",
        "home_col": "home_team",
        "away_col": "away_team",
        "time_col": "scheduled_time",
        "odds_api": "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
        "normalize": _norm_generic,
        "raw_fields": ["moneyline", "spread", "total"],
        "clean_mode": "nested_dicts",  # moneyline_clean/spread_clean/total_clean
    },
    "nfl": {
        "table": "nfl_game_schedule",
        "date_column": "game_date",
        "home_col": "home_team",
        "away_col": "away_team",
        "time_col": "scheduled_time",
        "odds_api": "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
        "normalize": _norm_nfl,
        "raw_fields": ["moneyline", "spread", "total"],
        "clean_mode": "nested_dicts",  # moneyline_clean/spread_clean/total_clean
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Odds API window + fetch
# ──────────────────────────────────────────────────────────────────────────────
def _odds_window(start_date_et: datetime, days: int) -> Tuple[str, str]:
    start_et = start_date_et.replace(hour=0, minute=0, second=0, microsecond=0)
    end_et = (start_et + timedelta(days=days - 1)).replace(hour=23, minute=59, second=59, microsecond=0)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (start_et.astimezone(UTC).strftime(fmt), end_et.astimezone(UTC).strftime(fmt))

def fetch_odds(odds_api_url: str, start_date_et: datetime, days: int) -> List[Dict[str, Any]]:
    c_from, c_to = _odds_window(start_date_et, days)
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "commenceTimeFrom": c_from,
        "commenceTimeTo": c_to,
    }
    print(f"[odds] GET {odds_api_url} {params}")
    try:
        r = requests.get(odds_api_url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        print(f"[odds] events: {len(data)}")
        return data
    except requests.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        body = getattr(getattr(e, "response", None), "text", "") or ""
        print(f"[odds] ERROR status={status} body={body[:180]}")
        return []

# ──────────────────────────────────────────────────────────────────────────────
# Matching + extraction
# ──────────────────────────────────────────────────────────────────────────────
def match_event(game_row: Dict[str, Any],
                odds_events: List[Dict[str, Any]],
                normalize_fn) -> Optional[Dict[str, Any]]:
    g_home = normalize_fn(game_row.get("_home_raw") or "")
    g_away = normalize_fn(game_row.get("_away_raw") or "")
    if not g_home or not g_away:
        return None

    # parse scheduled time (may be ET or UTC ISO with offset)
    g_dt = None
    try:
        g_dt = datetime.fromisoformat((game_row.get("_sched_iso") or "").replace("Z", "+00:00"))
    except Exception:
        pass

    for ev in odds_events:
        oh = normalize_fn(ev.get("home_team", "") or "")
        oa = normalize_fn(ev.get("away_team", "") or "")
        if oh != g_home or oa != g_away:
            continue
        try:
            edt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if g_dt:
                if edt.astimezone(ET).date() != g_dt.astimezone(ET).date():
                    continue
            return ev
        except Exception:
            continue
    return None

def extract_clean_payload(event: Optional[Dict[str, Any]],
                          clean_mode: str) -> Dict[str, Any]:
    """
    Returns a dict that includes:
      - raw dicts: moneyline, spread, total
      - and either:
        a) MLB individual clean columns, or
        b) nested dicts (NBA/NFL)
    """
    raw = {"moneyline": {}, "spread": {}, "total": {}}

    # default empty structures
    mlb_clean_cols = {
        "moneyline_home_clean": None,
        "moneyline_away_clean": None,
        "spread_home_line_clean": None,
        "spread_home_price_clean": None,
        "spread_away_price_clean": None,
        "total_line_clean": None,
        "total_over_price_clean": None,
        "total_under_price_clean": None,
    }
    nested = {
        "moneyline_clean": {"home": None, "away": None},
        "spread_clean": {"home": {"line": None, "price": None}, "away": {"line": None, "price": None}},
        "total_clean": {"line": None, "over_price": None, "under_price": None},
    }

    if not event:
        return {**raw, **(mlb_clean_cols if clean_mode == "mlb_columns" else nested)}

    # choose bookmaker
    bms = event.get("bookmakers", []) or []
    bm = next((b for b in bms if b.get("key") == PREFERRED_BOOK), bms[0]) if bms else {}
    markets = bm.get("markets", []) or []

    # team names as title for comparing outcomes
    home_title = _title(_norm_generic(event.get("home_team", "")))
    away_title = _title(_norm_generic(event.get("away_team", "")))

    # fill raw + clean
    for m in markets:
        key = m.get("key")
        outcomes = m.get("outcomes", []) or []

        if key == "h2h":
            # raw
            for o in outcomes:
                t = _title(o.get("name", ""))
                price = o.get("price")
                if t:
                    raw["moneyline"][t] = price
            # clean
            for o in outcomes:
                t = _title(o.get("name", ""))
                price = _fmt_american(_safe_float(o.get("price")))
                if clean_mode == "mlb_columns":
                    if t == home_title:
                        mlb_clean_cols["moneyline_home_clean"] = price
                    elif t == away_title:
                        mlb_clean_cols["moneyline_away_clean"] = price
                else:
                    if t == home_title:
                        nested["moneyline_clean"]["home"] = price
                    elif t == away_title:
                        nested["moneyline_clean"]["away"] = price

        elif key == "spreads":
            # raw
            for o in outcomes:
                t = _title(o.get("name", ""))
                if t and o.get("price") is not None and o.get("point") is not None:
                    raw["spread"][t] = {"price": o.get("price"), "point": o.get("point")}
            # clean
            for o in outcomes:
                t = _title(o.get("name", ""))
                price = _fmt_american(_safe_float(o.get("price")))
                point = _safe_float(o.get("point"))
                if clean_mode == "mlb_columns":
                    if t == home_title:
                        mlb_clean_cols["spread_home_line_clean"] = point
                        mlb_clean_cols["spread_home_price_clean"] = price
                    elif t == away_title:
                        mlb_clean_cols["spread_away_price_clean"] = price
                else:
                    if t == home_title:
                        nested["spread_clean"]["home"]["line"] = point
                        nested["spread_clean"]["home"]["price"] = price
                    elif t == away_title:
                        # away line sometimes absent in source payloads; keep None if missing
                        nested["spread_clean"]["away"]["price"] = price

        elif key == "totals":
            # raw
            for o in outcomes:
                nm = o.get("name", "")
                if nm in ("Over", "Under") and o.get("price") is not None and o.get("point") is not None:
                    raw["total"][nm] = {"price": o.get("price"), "point": o.get("point")}
            # clean
            first_point = None
            for o in outcomes:
                first_point = _safe_float(o.get("point"))
                if first_point is not None:
                    break
            if clean_mode == "mlb_columns":
                mlb_clean_cols["total_line_clean"] = first_point
            else:
                nested["total_clean"]["line"] = first_point
            for o in outcomes:
                nm = o.get("name")
                price = _fmt_american(_safe_float(o.get("price")))
                if clean_mode == "mlb_columns":
                    if nm == "Over":
                        mlb_clean_cols["total_over_price_clean"] = price
                    elif nm == "Under":
                        mlb_clean_cols["total_under_price_clean"] = price
                else:
                    if nm == "Over":
                        nested["total_clean"]["over_price"] = price
                    elif nm == "Under":
                        nested["total_clean"]["under_price"] = price

    return {**raw, **(mlb_clean_cols if clean_mode == "mlb_columns" else nested)}

# ──────────────────────────────────────────────────────────────────────────────
# Supabase reads / writes
# ──────────────────────────────────────────────────────────────────────────────
def _date_list(start_date_et: datetime, days: int) -> List[str]:
    base = start_date_et.date()
    return [(base + timedelta(days=i)).isoformat() for i in range(days)]

def load_schedule_rows(sport_cfg: Dict[str, Any], start_date_et: datetime, days: int) -> List[Dict[str, Any]]:
    table = sport_cfg["table"]
    date_col = sport_cfg["date_column"]
    home_col = sport_cfg["home_col"]
    away_col = sport_cfg["away_col"]
    time_col = sport_cfg["time_col"]

    dates = _date_list(start_date_et, days)

    print(f"[db] fetch {table} where {date_col} IN {dates}")
    # We only need key fields + existing odds to optionally skip
    cols = ",".join([
        "game_id", home_col, away_col, time_col,
        "moneyline", "spread", "total",
        # try to select clean fields if they exist; safe to include (Supabase ignores extras)
        "moneyline_clean", "spread_clean", "total_clean",
        "moneyline_home_clean","moneyline_away_clean",
        "spread_home_line_clean","spread_home_price_clean","spread_away_price_clean",
        "total_line_clean","total_over_price_clean","total_under_price_clean",
    ])
    resp = supabase.table(table).select(cols).in_(date_col, dates).execute()
    rows = resp.data or []

    # normalize helper keys for matching
    for r in rows:
        r["_home_raw"] = r.get(home_col)
        r["_away_raw"] = r.get(away_col)
        r["_sched_iso"] = r.get(time_col)
    print(f"[db] rows: {len(rows)}")
    return rows

def build_update_payload(sport: str,
                         sport_cfg: Dict[str, Any],
                         clean: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "moneyline": clean.get("moneyline") or {},
        "spread": clean.get("spread") or {},
        "total": clean.get("total") or {},
        "updated_at": datetime.now(UTC).isoformat(),
    }

    if sport_cfg["clean_mode"] == "mlb_columns":
        # add individual columns
        for k in [
            "moneyline_home_clean","moneyline_away_clean",
            "spread_home_line_clean","spread_home_price_clean","spread_away_price_clean",
            "total_line_clean","total_over_price_clean","total_under_price_clean",
        ]:
            payload[k] = clean.get(k)
    else:
        # nested dicts for NBA/NFL
        payload["moneyline_clean"] = clean.get("moneyline_clean")
        payload["spread_clean"] = clean.get("spread_clean")
        payload["total_clean"] = clean.get("total_clean")

    return payload

def perform_updates(sport: str,
                    rows: List[Dict[str, Any]],
                    odds_events: List[Dict[str, Any]],
                    sport_cfg: Dict[str, Any],
                    skip_existing: bool) -> Tuple[int, int, int]:
    normalize_fn = sport_cfg["normalize"]
    updated, skipped, unmatched = 0, 0, 0

    for r in rows:
        # Optional skip: if clean odds already present, avoid clobbering
        if skip_existing:
            if sport_cfg["clean_mode"] == "mlb_columns":
                has_clean = any(r.get(k) is not None for k in [
                    "moneyline_home_clean","moneyline_away_clean",
                    "spread_home_line_clean","spread_home_price_clean","spread_away_price_clean",
                    "total_line_clean","total_over_price_clean","total_under_price_clean",
                ])
            else:
                mc = r.get("moneyline_clean") or {}
                sc = r.get("spread_clean") or {}
                tc = r.get("total_clean") or {}
                has_clean = any([
                    mc.get("home") or mc.get("away"),
                    (sc.get("home") or {}).get("price") or (sc.get("home") or {}).get("line") or (sc.get("away") or {}).get("price"),
                    tc.get("line") or tc.get("over_price") or tc.get("under_price"),
                ])
            if has_clean:
                skipped += 1
                continue

        ev = match_event(r, odds_events, normalize_fn)
        if not ev:
            unmatched += 1
            continue

        clean = extract_clean_payload(ev, sport_cfg["clean_mode"])
        payload = build_update_payload(sport, sport_cfg, clean)

        try:
            supabase.table(sport_cfg["table"]).update(payload).eq("game_id", r.get("game_id")).execute()
            updated += 1
        except Exception as e:
            print(f"[db] update error game_id={r.get('game_id')}: {e}")

    return updated, skipped, unmatched

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Refresh odds for MLB/NBA/NFL schedule rows.")
    ap.add_argument("--sport", choices=["mlb","nba","nfl"], required=True, help="Which sport to refresh.")
    ap.add_argument("--date", help="ET start date (YYYY-MM-DD). Default: today ET.")
    ap.add_argument("--days", type=int, help="Days ahead (inclusive). Defaults: mlb=2, nba=2, nfl=8")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip rows that already have clean odds populated.")
    args = ap.parse_args()

    cfg = SPORTS[args.sport]
    default_days = 8 if args.sport == "nfl" else 2
    days = args.days or default_days

    if args.date:
        try:
            start_et = datetime.fromisoformat(args.date).replace(tzinfo=ET)
        except Exception:
            print("--date must be YYYY-MM-DD")
            sys.exit(2)
    else:
        start_et = datetime.now(ET)

    rows = load_schedule_rows(cfg, start_et, days)
    if not rows:
        print("No schedule rows found in the selected window. Nothing to do.")
        return

    odds = fetch_odds(cfg["odds_api"], start_et, days)
    if not odds:
        print("No odds events fetched. Exiting without updates.")
        return

    updated, skipped, unmatched = perform_updates(args.sport, rows, odds, cfg, args.skip_existing)
    print(f"Done. updated={updated}, skipped={skipped}, unmatched={unmatched}")

if __name__ == "__main__":
    main()
