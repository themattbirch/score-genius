# backend/mlb_features/make_mlb_snapshots.py
"""
Generate and upsert per‑game MLB feature snapshots for frontend display.

Key fixes:
- Robust, single-path date parsing (ET) for all game rows.
- Dedicated _compute_rest_days() with clamps and debug logging.
- Rest advantage calculated once, consistently, for pre-game snapshots.
- Validation step to catch/normalize any absurd rest numbers before headline build.
- Extra logging around Toronto (and any team) if rest > 10 days.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from supabase import create_client, Client

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Imports
# ──────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
PROJECT_DIR = BACKEND_DIR.parent
for p in (BACKEND_DIR, PROJECT_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from backend.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
except ImportError:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase URL/key missing in make_mlb_snapshots.py")

try:
    from backend.mlb_features.engine import run_mlb_feature_pipeline
    from backend.mlb_features.utils import (
        normalize_team_name,
        determine_season,
        DEFAULTS as MLB_DEFAULTS,
    )
    from backend.mlb_features.handedness_for_display import transform as handedness_transform
except ImportError as e:
    logger.error("Could not import MLB feature dependencies: %s", e)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Supabase
# ──────────────────────────────────────────────────────────────────────────────
sb_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Fetch helpers
# ──────────────────────────────────────────────────────────────────────────────
def fetch_table(
    table_name: str,
    match_criteria: Dict[str, Any] | None = None,
    select_cols: str = "*",
) -> pd.DataFrame:
    logger.debug("Fetching %s with %s", table_name, match_criteria)
    try:
        qb = sb_client.table(table_name).select(select_cols)
        if match_criteria:
            qb = qb.match(match_criteria)
        res = qb.execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        logger.error("Error fetching table %s: %s", table_name, e, exc_info=True)
        return pd.DataFrame()


def fetch_mlb_full_history() -> pd.DataFrame:
    """Full historical game stats (needed for form & rest)."""
    res = sb_client.table("mlb_historical_game_stats").select("*").execute()
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df

    # Normalize to ET date for comparisons
    if "game_date_time_utc" in df.columns:
        df["played_date_et"] = (
            pd.to_datetime(df["game_date_time_utc"], errors="coerce", utc=True)
            .dt.tz_convert("America/New_York")
            .dt.date
        )
    elif "game_date_et" in df.columns:
        df["played_date_et"] = pd.to_datetime(df["game_date_et"], errors="coerce").dt.date
    else:
        df["played_date_et"] = pd.NaT

    # String team ids for quick masks
    if "home_team_id" in df.columns:
        df["home_team_id_str"] = df["home_team_id"].astype(str)
    if "away_team_id" in df.columns:
        df["away_team_id_str"] = df["away_team_id"].astype(str)
    return df


def fetch_mlb_team_season_stats() -> pd.DataFrame:
    res = sb_client.table("mlb_historical_team_stats").select("*").execute()
    return pd.DataFrame(res.data or [])


def fetch_mlb_pitcher_splits_data(season_year: int) -> pd.DataFrame:
    res = (
        sb_client.table("mlb_historical_team_stats")
        .select("team_id, season, season_avg_runs_vs_lhp, season_avg_runs_vs_rhp")
        .eq("season", season_year)
        .execute()
    )
    return pd.DataFrame(res.data or [])


# ──────────────────────────────────────────────────────────────────────────────
# Utility: Team form string
# ──────────────────────────────────────────────────────────────────────────────
def _get_mlb_team_form_string(
    team_id_to_check: Union[str, int],
    current_game_date_et: Union[pd.Timestamp, pd.Timestamp, str],
    all_historical_games: pd.DataFrame,
    num_form_games: int = 5,
) -> str:
    team_id_str = str(team_id_to_check)
    current_date = pd.to_datetime(current_game_date_et, errors="coerce").date()

    if pd.isna(current_date) or all_historical_games.empty:
        return "N/A"

    if "parsed_date_for_form" not in all_historical_games.columns:
        if "game_date_et" in all_historical_games.columns:
            all_historical_games["parsed_date_for_form"] = (
                pd.to_datetime(all_historical_games["game_date_et"], errors="coerce").dt.date
            )
        elif "game_date_time_utc" in all_historical_games.columns:
            all_historical_games["parsed_date_for_form"] = (
                pd.to_datetime(all_historical_games["game_date_time_utc"], errors="coerce", utc=True)
                .dt.tz_convert("America/New_York")
                .dt.date
            )
        else:
            return "N/A"

    # Cast id columns to string once
    if "home_team_id_str" not in all_historical_games.columns and "home_team_id" in all_historical_games.columns:
        all_historical_games["home_team_id_str"] = all_historical_games["home_team_id"].astype(str)
    if "away_team_id_str" not in all_historical_games.columns and "away_team_id" in all_historical_games.columns:
        all_historical_games["away_team_id_str"] = all_historical_games["away_team_id"].astype(str)

    completed_mask = (
        (all_historical_games["parsed_date_for_form"] < current_date)
        & (
            all_historical_games.get("status_short", "").isin(["FT", "F"])
            | (all_historical_games.get("status_long", "") == "Finished")
        )
        & pd.notna(all_historical_games.get("home_score"))
        & pd.notna(all_historical_games.get("away_score"))
    )

    team_mask = (
        (all_historical_games["home_team_id_str"] == team_id_str)
        | (all_historical_games["away_team_id_str"] == team_id_str)
    )

    team_games = all_historical_games.loc[completed_mask & team_mask].copy()
    if team_games.empty:
        return "N/A"

    recent_games = team_games.sort_values("parsed_date_for_form", ascending=False).head(num_form_games)
    if recent_games.empty:
        return "N/A"

    results: List[str] = []
    for _, g in recent_games.sort_values("parsed_date_for_form").iterrows():
        is_home = g["home_team_id_str"] == team_id_str
        team_score = pd.to_numeric(g["home_score"] if is_home else g["away_score"], errors="coerce")
        opp_score = pd.to_numeric(g["away_score"] if is_home else g["home_score"], errors="coerce")
        if pd.isna(team_score) or pd.isna(opp_score):
            results.append("?")
        elif team_score > opp_score:
            results.append("W")
        elif team_score < opp_score:
            results.append("L")
        else:
            results.append("T")
    return "".join(results) if results else "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Rest calculation helpers
# ──────────────────────────────────────────────────────────────────────────────
MAX_REASONABLE_REST = 10  # days, anything above gets clamped/logged
MIN_REST = 0              # we allow 0 (same-day doubleheader edge case)


def _compute_rest_days(
    all_hist: pd.DataFrame, team_id: str, game_date_et: pd.Timestamp | pd.Timestamp | str
) -> int:
    """Return days since team_id last played before game_date_et.

    Ensures:
    - uses ET dates
    - returns at least MIN_REST
    - clamps large off-season diffs to 1 (or MIN_REST) to avoid 200+ values
    """
    gdate = pd.to_datetime(game_date_et, errors="coerce").date()
    if pd.isna(gdate):
        return 1

    mask = (
        (all_hist["played_date_et"] < gdate)
        & (
            (all_hist["home_team_id_str"] == str(team_id))
            | (all_hist["away_team_id_str"] == str(team_id))
        )
    )
    if not mask.any():
        return 1

    last_date = all_hist.loc[mask, "played_date_et"].max()
    if pd.isna(last_date):
        return 1

    diff = int((gdate - last_date).days)
    diff = max(MIN_REST, diff)

    if diff > MAX_REASONABLE_REST:
        logger.debug(
            "Rest > %d detected (team=%s, diff=%d, last=%s, game=%s). Clamping to 1.",
            MAX_REASONABLE_REST,
            team_id,
            diff,
            last_date,
            gdate,
        )
        return 1
    return diff


def _validate_and_fix_rest(row: Dict[str, Any]) -> None:
    """Normalize absurd values and log."""
    hr = float(row.get("rest_days_home", 1) or 1)
    ar = float(row.get("rest_days_away", 1) or 1)

    def clamp(val: float) -> float:
        if val < MIN_REST:
            return MIN_REST
        if val > MAX_REASONABLE_REST:
            return 1.0
        return val

    new_hr, new_ar = clamp(hr), clamp(ar)
    if (new_hr != hr) or (new_ar != ar):
        logger.debug(
            "Clamped rest days: home %.2f->%.2f, away %.2f->%.2f", hr, new_hr, ar, new_ar
        )

    row["rest_days_home"] = new_hr
    row["rest_days_away"] = new_ar
    row["rest_advantage"] = new_hr - new_ar


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot Generator
# ──────────────────────────────────────────────────────────────────────────────
def make_mlb_snapshot(
    game_id: Union[str, int],
    input_game_date_col: str = "game_date_et",
    input_game_date_utc_col: str = "game_date_time_utc",
    input_home_team_id_col: str = "home_team_id",
    input_away_team_id_col: str = "away_team_id",
    input_home_pitcher_hand_col: str = "home_starter_pitcher_handedness",
    input_away_pitcher_hand_col: str = "away_starter_pitcher_handedness",
    input_home_score_col: str = "home_score",
    input_away_score_col: str = "away_score",
):
    gid = str(game_id)
    logger.info("— Generating MLB Snapshot for game_id=%s —", gid)

    hist_cols = (
        "game_id,game_date_time_utc,status_short,status_long,"
        "home_team_id,away_team_id,home_score,away_score,"
        "h_inn_1,h_inn_2,h_inn_3,h_inn_4,h_inn_5,h_inn_6,h_inn_7,h_inn_8,h_inn_9,h_inn_extra,"
        "a_inn_1,a_inn_2,a_inn_3,a_inn_4,a_inn_5,a_inn_6,a_inn_7,a_inn_8,a_inn_9,a_inn_extra,"
        "home_starter_pitcher_handedness,away_starter_pitcher_handedness"
    )
    df_hist = fetch_table("mlb_historical_game_stats", {"game_id": gid}, hist_cols)
    df_sched = fetch_table(
        "mlb_game_schedule",
        {"game_id": gid},
        "game_id,home_team_id,away_team_id,game_date_et,home_probable_pitcher_handedness,away_probable_pitcher_handedness",
    )

    is_historical_game = not df_hist.empty

    # Build df_game
    if is_historical_game:
        df_game = df_hist.copy()
        if not df_sched.empty and "game_date_et" in df_sched.columns:
            df_game["game_date_et"] = df_sched["game_date_et"].iloc[0]
    else:
        if df_sched.empty:
            logger.error("No schedule row for game_id=%s. Abort.", gid)
            return
        df_game = df_sched.copy()
        # Stub cols that won't exist yet
        for c in [
            input_home_score_col,
            input_away_score_col,
            input_home_pitcher_hand_col,
            input_away_pitcher_hand_col,
            *[f"h_inn_{i}" for i in range(1, 10)],
            "h_inn_extra",
            *[f"a_inn_{i}" for i in range(1, 10)],
            "a_inn_extra",
        ]:
            if c not in df_game.columns:
                df_game[c] = pd.NA

    # Parse/current ET date
    if "game_date_et" in df_game.columns and pd.notna(df_game["game_date_et"].iloc[0]):
        current_game_date_et = pd.to_datetime(df_game["game_date_et"].iloc[0], errors="coerce").date()
    else:
        dt_utc = (
            pd.to_datetime(df_hist[input_game_date_utc_col].iloc[0], errors="coerce", utc=True)
            if not df_hist.empty
            else pd.NaT
        )
        current_game_date_et = (
            dt_utc.tz_convert("America/New_York").date() if pd.notna(dt_utc) else None
        )

    if current_game_date_et is None or pd.isna(current_game_date_et):
        logger.error("No valid game date for game_id=%s. Abort.", gid)
        return

    # Preload history/team stats once
    df_full_history = fetch_mlb_full_history()
    df_team_stats = fetch_mlb_team_season_stats()

    # Rest advantage (pre-game only)
    if not is_historical_game:
        home_id = str(df_game[input_home_team_id_col].iloc[0])
        away_id = str(df_game[input_away_team_id_col].iloc[0])

        home_rest = _compute_rest_days(df_full_history, home_id, current_game_date_et)
        away_rest = _compute_rest_days(df_full_history, away_id, current_game_date_et)

        df_game["rest_days_home"] = home_rest
        df_game["rest_days_away"] = away_rest
        df_game["rest_advantage"] = home_rest - away_rest
    else:
        df_game["rest_days_home"] = pd.NA
        df_game["rest_days_away"] = pd.NA
        df_game["rest_advantage"] = pd.NA

    df_game["is_historical_game"] = is_historical_game

    # Determine season
    season_year = determine_season(pd.Timestamp(current_game_date_et))

    # Form strings
    home_id = df_game[input_home_team_id_col].iloc[0]
    away_id = df_game[input_away_team_id_col].iloc[0]
    df_game["home_current_form"] = _get_mlb_team_form_string(home_id, current_game_date_et, df_full_history)
    df_game["away_current_form"] = _get_mlb_team_form_string(away_id, current_game_date_et, df_full_history)

    # Feature pipeline
    df_features = run_mlb_feature_pipeline(
        df=df_game.copy(),
        mlb_historical_games_df=df_full_history,
        mlb_historical_team_stats_df=df_team_stats,
        debug=True,
        keep_display_only_features=True,
    )
    if df_features.empty:
        logger.error("Feature pipeline returned empty for game_id=%s", gid)
        return
    if len(df_features) > 1:
        logger.warning("Expected 1 feature row, got %d; using first.", len(df_features))

    row = df_features.iloc[0].to_dict()

    # Finalize / validate rest values
    if is_historical_game:
        # If the pipeline produced rest fields, normalize them; else leave NA
        _validate_and_fix_rest(row)
    else:
        row["rest_days_home"] = int(df_game["rest_days_home"].iloc[0])
        row["rest_days_away"] = int(df_game["rest_days_away"].iloc[0])
        row["rest_advantage"] = row["rest_days_home"] - row["rest_days_away"]
        _validate_and_fix_rest(row)

    logger.info(
        "Rest (final): H=%s A=%s Adv=%s",
        row["rest_days_home"],
        row["rest_days_away"],
        row["rest_advantage"],
    )

    # Radar helpers
    h_for = row.get("home_season_runs_for_avg", 4.5)
    h_against = row.get("home_season_runs_against_avg", 4.5)
    a_for = row.get("away_season_runs_for_avg", 4.5)
    a_against = row.get("away_season_runs_against_avg", 4.5)

    row["home_run_differential"] = float(h_for) - float(h_against)
    row["away_run_differential"] = float(a_for) - float(a_against)
    row["home_pythag_win_pct"] = (float(h_for) ** 2) / (
        (float(h_for) ** 2) + (float(h_against) ** 2)
    )
    row["away_pythag_win_pct"] = (float(a_for) ** 2) / (
        (float(a_for) ** 2) + (float(a_against) ** 2)
    )

    # Handedness display features
    df_pitcher_splits = fetch_mlb_pitcher_splits_data(season_year)
    home_hand_col = (
        "home_starter_pitcher_handedness"
        if is_historical_game
        else "home_probable_pitcher_handedness"
    )
    away_hand_col = (
        "away_starter_pitcher_handedness"
        if is_historical_game
        else "away_probable_pitcher_handedness"
    )
    row[home_hand_col] = df_game.get(home_hand_col, pd.Series([pd.NA])).iloc[0]
    row[away_hand_col] = df_game.get(away_hand_col, pd.Series([pd.NA])).iloc[0]

    df_hand = handedness_transform(
        df=pd.DataFrame([row]),
        mlb_pitcher_splits_df=df_pitcher_splits,
        home_team_col_param="home_team_norm",
        away_team_col_param="away_team_norm",
        home_pitcher_hand_col=home_hand_col,
        away_pitcher_hand_col=away_hand_col,
        debug=True,
    )
    if not df_hand.empty:
        row.update(df_hand.iloc[0].to_dict())

    # Advanced RPC for bar chart
    rpc_adv = sb_client.rpc("get_mlb_advanced_team_stats_splits", {"p_season": season_year}).execute().data or []
    df_rpc_adv = pd.DataFrame(rpc_adv)
    if "team_id" in df_rpc_adv.columns:
        df_rpc_adv["team_norm"] = df_rpc_adv["team_id"].apply(normalize_team_name)

    home_norm = normalize_team_name(row.get(input_home_team_id_col))
    away_norm = normalize_team_name(row.get(input_away_team_id_col))

    def first_row(df: pd.DataFrame) -> Dict[str, Any]:
        return df.iloc[0].to_dict() if not df.empty else {}

    home_adv = first_row(df_rpc_adv[df_rpc_adv["team_norm"] == home_norm])
    away_adv = first_row(df_rpc_adv[df_rpc_adv["team_norm"] == away_norm])

    RUNS_FOR_DEFAULT = float(MLB_DEFAULTS.get("mlb_avg_runs_for", 4.5))
    RUNS_AGAINST_DEFAULT = float(MLB_DEFAULTS.get("mlb_avg_runs_against", 4.5))

    bar_chart_data = [
        {
            "category": "Avg Runs For",
            "Home": round(float(home_adv.get("runs_for_avg_overall", RUNS_FOR_DEFAULT)), 2),
            "Away": round(float(away_adv.get("runs_for_avg_overall", RUNS_FOR_DEFAULT)), 2),
        },
        {
            "category": "Avg Runs Against",
            "Home": round(float(home_adv.get("runs_against_avg_overall", RUNS_AGAINST_DEFAULT)), 2),
            "Away": round(float(away_adv.get("runs_against_avg_overall", RUNS_AGAINST_DEFAULT)), 2),
        },
    ]

    # Metric ranges RPC
    ranges_rows = sb_client.rpc("get_mlb_metric_ranges", {"p_season": season_year}).execute().data or []
    league_ranges: Dict[str, Dict[str, float | bool]] = {
        r["metric"]: {
            "min": float(r["min_value"]),
            "max": float(r["max_value"]),
            "invert": (r["metric"] == "Season Runs Against"),
        }
        for r in ranges_rows
    }

    # Fill missing ranges with computed values
    if "Pythagorean Win %" not in league_ranges and not df_team_stats.empty:
        pyth = (df_team_stats["runs_for_avg_all"] ** 2) / (
            (df_team_stats["runs_for_avg_all"] ** 2)
            + (df_team_stats["runs_against_avg_all"] ** 2)
        )
        league_ranges["Pythagorean Win %"] = {
            "min": float(pyth.min()),
            "max": float(pyth.max()),
            "invert": False,
        }

    if "Run Differential" not in league_ranges and not df_team_stats.empty:
        rd = df_team_stats["runs_for_avg_all"] - df_team_stats["runs_against_avg_all"]
        league_ranges["Run Differential"] = {
            "min": float(rd.min()),
            "max": float(rd.max()),
            "invert": False,
        }

    if not ranges_rows:
        league_ranges.update(
            {
                "Venue Win %": {
                    "min": float(df_team_stats["wins_home_percentage"].min()),
                    "max": float(df_team_stats["wins_home_percentage"].max()),
                    "invert": False,
                },
                "Season Runs Scored": {
                    "min": float(df_team_stats["runs_for_avg_all"].min()),
                    "max": float(df_team_stats["runs_for_avg_all"].max()),
                    "invert": False,
                },
                "Season Runs Against": {
                    "min": float(df_team_stats["runs_against_avg_all"].min()),
                    "max": float(df_team_stats["runs_against_avg_all"].max()),
                    "invert": True,
                },
                "Home/Away Win Advantage": {
                    "min": float(
                        (df_team_stats["wins_home_percentage"] - df_team_stats["wins_away_percentage"]).min()
                    ),
                    "max": float(
                        (df_team_stats["wins_home_percentage"] - df_team_stats["wins_away_percentage"]).max()
                    ),
                    "invert": False,
                },
            }
        )

    radar_metrics_map = {
        "Pythagorean W%": {
            "db_name": "Pythagorean Win %",
            "home_col": "home_pythag_win_pct",
            "away_col": "away_pythag_win_pct",
            "round": 3,
        },
        "Run Differential": {
            "db_name": "Run Differential",
            "home_col": "home_run_differential",
            "away_col": "away_run_differential",
            "round": 2,
        },
        "Venue W%": {
            "db_name": "Venue Win %",
            "home_col": "home_venue_win_pct_home",
            "away_col": "away_venue_win_pct_away",
            "round": 3,
        },
        "H/A Advantage": {
            "db_name": "Home/Away Win Advantage",
            "home_col": "home_venue_win_advantage",
            "away_col": "away_venue_win_advantage",
            "round": 3,
        },
    }

    def _scale(v: float, rng: Dict[str, Any]) -> float:
        if rng["max"] == rng["min"]:
            pct = 50.0
        else:
            pct = 100.0 * (v - rng["min"]) / (rng["max"] - rng["min"])
        return 100.0 - pct if rng.get("invert", False) else pct

    radar_payload: List[Dict[str, Any]] = []
    for disp, cfg in radar_metrics_map.items():
        rng = league_ranges[cfg["db_name"]]
        h_raw = float(row.get(cfg["home_col"], 0.0))
        a_raw = float(row.get(cfg["away_col"], 0.0))
        radar_payload.append(
            {
                "metric": disp,
                "home_raw": round(h_raw, cfg["round"]),
                "away_raw": round(a_raw, cfg["round"]),
                "home_idx": round(_scale(h_raw, rng), 1),
                "away_idx": round(_scale(a_raw, rng), 1),
            }
        )

    home_hand_val = float(row.get("h_team_off_avg_runs_vs_opp_hand", 0.0))
    away_hand_val = float(row.get("a_team_off_avg_runs_vs_opp_hand", 0.0))
    pie_payload = [
        {
            "category": f"Home Offense vs Starting Pitcher's Hand ({round(home_hand_val, 2)} Runs)",
            "value": home_hand_val,
            "color": "#60a5fa",
        },
        {
            "category": f"Away Offense vs Starting Pitcher's Hand ({round(away_hand_val, 2)} Runs)",
            "value": away_hand_val,
            "color": "#4ade80",
        },
    ]

    # Headlines
    headlines = [
        {"label": "Rest Advantage (Home)", "value": int(row["rest_advantage"])},
        {
            "label": "Form Win% Diff",
            "value": round(float(row.get("form_win_pct_diff", 0.0)), 2),
        },
        {
            "label": "Prev Season Win% Diff",
            "value": round(float(row.get("prev_season_win_pct_diff", 0.0)), 2),
        },
        {
            "label": f"H2H Home Win% (L{int(row.get('matchup_num_games', 0))})",
            "value": round(float(row.get("matchup_home_win_pct", 0.0)), 2),
        },
    ]

    snapshot_payload_final = {
        "game_id": gid,
        "game_date": pd.Timestamp(current_game_date_et).isoformat(),
        "season": str(season_year),
        "is_historical": is_historical_game,
        "headline_stats": headlines,
        "bar_chart_data": bar_chart_data,
        "radar_chart_data": radar_payload,
        "pie_chart_data": pie_payload,
        "last_updated": pd.Timestamp.utcnow().isoformat(),
    }

    logger.info("Upserting MLB snapshot for game_id=%s", gid)
    upsert_response = (
        sb_client.table("mlb_snapshots")
        .upsert(snapshot_payload_final, on_conflict="game_id")
        .execute()
    )

    if getattr(upsert_response, "error", None):
        logger.error("Snapshot upsert FAILED for %s: %s", gid, upsert_response.error)
    elif hasattr(upsert_response, "data") and not upsert_response.data and not getattr(
        upsert_response, "count", 0
    ):
        logger.warning("Snapshot upsert returned no data/count for %s: %s", gid, upsert_response)
    else:
        logger.info("✅ Snapshot upserted for game_id=%s", gid)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info(
            "Usage: python backend/mlb_features/make_mlb_snapshots.py <game_id1> [<game_id2> ...]"
        )
    else:
        for arg in sys.argv[1:]:
            try:
                make_mlb_snapshot(arg)
            except Exception as e_main:
                logger.error("CRITICAL ERROR for game_id %s: %s", arg, e_main, exc_info=True)
