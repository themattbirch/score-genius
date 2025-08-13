# backend/nfl_features/situational.py
from __future__ import annotations

import logging
from typing import Mapping, Optional, Tuple

import pandas as pd

from . import utils

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------------------------------------
# Division / conference map (canonicalized team shortnames)
# --------------------------------------------------------------------------
TEAM_DIVISIONS: Mapping[str, str] = {
    # NFC East
    "cowboys": "NFC East", "giants": "NFC East", "eagles": "NFC East", "commanders": "NFC East",
    # NFC North
    "bears": "NFC North", "lions": "NFC North", "packers": "NFC North", "vikings": "NFC North",
    # NFC South
    "falcons": "NFC South", "panthers": "NFC South", "saints": "NFC South", "buccaneers": "NFC South",
    # NFC West
    "cardinals": "NFC West", "rams": "NFC West", "49ers": "NFC West", "seahawks": "NFC West",
    # AFC East
    "bills": "AFC East", "dolphins": "AFC East", "patriots": "AFC East", "jets": "AFC East",
    # AFC North
    "ravens": "AFC North", "bengals": "AFC North", "browns": "AFC North", "steelers": "AFC North",
    # AFC South
    "texans": "AFC South", "colts": "AFC South", "jaguars": "AFC South", "titans": "AFC South",
    # AFC West
    "broncos": "AFC West", "chiefs": "AFC West", "raiders": "AFC West", "chargers": "AFC West",
}

# --------------------------------------------------------------------------
# Known international venues (historical + recent)
# --------------------------------------------------------------------------
INTERNATIONAL_VENUE_NAMES = {
    # UK
    "wembley stadium", "tottenham hotspur stadium", "twickenham stadium",
    # Germany
    "allianz arena", "deutsche bank park", "frankfurt stadium", "frankfurt arena",
    # Mexico
    "estadio azteca",
}
INTERNATIONAL_COUNTRIES = {"uk", "united kingdom", "england", "germany", "mexico"}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _et_timestamp(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Build kickoff timestamps:
      - ts_utc: prefer 'game_timestamp' (UTC) if present, else from game_date + game_time (naive ET).
      - ts_et : ts_utc converted to America/New_York.
    """
    # Prefer explicit UTC timestamp if present
    if "game_timestamp" in df.columns:
        ts_utc = pd.to_datetime(df["game_timestamp"], utc=True, errors="coerce")
    else:
        ts_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    # Fallback from date+time (assume ET scheduling)
    date_str = df.get("game_date", pd.Series("", index=df.index)).astype(str)
    time_str = df.get("game_time", pd.Series("00:00:00", index=df.index)).astype(str)
    dt_combo = (date_str + " " + time_str).str.strip()

    try:
        fallback_et = (
            pd.to_datetime(dt_combo, errors="coerce")
            .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        )
        fallback_utc = fallback_et.dt.tz_convert("UTC")
    except Exception:
        fallback_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    ts_utc = ts_utc.fillna(fallback_utc)
    ts_et = ts_utc.dt.tz_convert("America/New_York")
    return ts_utc, ts_et


def _is_thanksgiving(d: pd.Timestamp) -> bool:
    """NFL Thanksgiving: 4th Thursday of November."""
    if pd.isna(d):
        return False
    d = d.tz_convert("America/New_York") if d.tzinfo is not None else d.tz_localize("America/New_York")
    year = d.year
    # First day of November
    nov1 = pd.Timestamp(year=year, month=11, day=1, tz="America/New_York")
    # Day of week (Mon=0 ... Sun=6); Thursday=3
    offset = (3 - nov1.dayofweek) % 7
    first_thu = nov1 + pd.Timedelta(days=offset)
    fourth_thu = first_thu + pd.Timedelta(days=21)
    return d.normalize() == fourth_thu.normalize()


def _is_black_friday(d: pd.Timestamp) -> bool:
    """Day after Thanksgiving."""
    if pd.isna(d):
        return False
    d_et = d.tz_convert("America/New_York") if d.tzinfo is not None else d.tz_localize("America/New_York")
    return _is_thanksgiving(d_et - pd.Timedelta(days=1)) is True


def _holiday_flags(ts_et: pd.Series) -> pd.DataFrame:
    """Return boolean flags for holiday/special slates (ET)."""
    out = pd.DataFrame(index=ts_et.index)
    out["situational_is_thanksgiving"] = ts_et.apply(_is_thanksgiving).astype(int)
    out["situational_is_black_friday"] = ts_et.apply(_is_black_friday).astype(int)

    # Christmas / Christmas Eve
    month = ts_et.dt.month
    day = ts_et.dt.day
    out["situational_is_christmas"] = ((month == 12) & (day == 25)).astype(int)
    out["situational_is_christmas_eve"] = ((month == 12) & (day == 24)).astype(int)
    return out


def _slot_labels(hour_et: pd.Series, is_international: pd.Series) -> pd.Series:
    """
    Classify kickoff slot (ET):
      - intl_morning: international & hour < 13
      - early       : 12:00 <= hour < 16 (most 1:00p)
      - late        : 16:00 <= hour < 19 (4:05/4:25p)
      - night       : hour >= 19 (SNF/MNF/Thu/Sat night)
    """
    slot = pd.Series("early", index=hour_et.index, dtype="object")

    # International morning first
    slot = slot.mask(is_international & (hour_et < 13), "intl_morning")
    # Late window
    slot = slot.mask((hour_et >= 16) & (hour_et < 19), "late")
    # Night window
    slot = slot.mask(hour_et >= 19, "night")
    return slot


def _one_hot_slot(slot: pd.Series) -> pd.DataFrame:
    """One-hot numeric flags for model consumption."""
    out = pd.DataFrame(index=slot.index)
    out["situational_slot_early"] = (slot == "early").astype(int)
    out["situational_slot_late"] = (slot == "late").astype(int)
    out["situational_slot_night"] = (slot == "night").astype(int)
    out["situational_slot_intl_morning"] = (slot == "intl_morning").astype(int)
    return out


def _detect_international(df: pd.DataFrame) -> pd.Series:
    """
    Detect international games using venue hints:
      - venue_country (non-USA)
      - venue_name in known set
    Falls back to 0 if no hints available.
    """
    name = df.get("venue_name", pd.Series("", index=df.index)).astype(str).str.lower().str.strip()
    country = df.get("venue_country", pd.Series("", index=df.index)).astype(str).str.lower().str.strip()

    is_intl_by_name = name.isin(INTERNATIONAL_VENUE_NAMES)
    is_intl_by_country = country.isin(INTERNATIONAL_COUNTRIES) | ((country != "") & (~country.isin({"us", "usa", "united states", "u.s.", "u.s.a."})))
    return (is_intl_by_name | is_intl_by_country).astype(int)


def _detect_neutral(df: pd.DataFrame, is_international: pd.Series) -> pd.Series:
    """
    Neutral-site heuristic:
      - respect explicit 'is_neutral_site' if provided.
      - otherwise, treat international as neutral.
      - optionally, if venue_home_team provided and != home_team_norm -> neutral.
    """
    if "is_neutral_site" in df.columns:
        base = df["is_neutral_site"].fillna(0).astype(int)
    else:
        base = pd.Series(0, index=df.index, dtype=int)

    # If we know the venue's home team, compare:
    venue_home = df.get("venue_home_team_norm", pd.Series("", index=df.index)).astype(str).str.lower()
    home_team = df.get("home_team_norm", pd.Series("", index=df.index)).astype(str).str.lower()
    by_owner_mismatch = ((venue_home != "") & (home_team != "") & (venue_home != home_team)).astype(int)

    # Combine: explicit OR owner mismatch OR international
    neutral = ((base == 1) | (by_owner_mismatch == 1) | (is_international == 1)).astype(int)
    return neutral


def _late_season_mask(df: pd.DataFrame, ts_et: pd.Series, is_regular_season: pd.Series) -> pd.Series:
    """
    Late-season proxy:
      - If 'week' is present: week >= 15 during regular season.
      - Else: REG-season games in December or later.
    """
    if "week" in df.columns:
        wk = pd.to_numeric(df["week"], errors="coerce")
        return ((is_regular_season == 1) & (wk >= 15)).astype(int)
    # Fallback by calendar
    month = ts_et.dt.month
    return ((is_regular_season == 1) & (month >= 12)).astype(int)


def _division_conference(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    home = df.get("home_team_norm", pd.Series("", index=df.index)).apply(utils.normalize_team_name).str.lower()
    away = df.get("away_team_norm", pd.Series("", index=df.index)).apply(utils.normalize_team_name).str.lower()

    home_div = home.map(TEAM_DIVISIONS)
    away_div = away.map(TEAM_DIVISIONS)

    home_conf = home_div.str.split().str[0]
    away_conf = away_div.str.split().str[0]

    is_division_game = (home_div == away_div).astype(int)
    is_conference_game = ((home_conf == away_conf) & (is_division_game == 0)).astype(int)
    return is_division_game, is_conference_game, home_div, away_div


def _standalone_flag(ts_et: pd.Series, is_international: pd.Series, holiday_df: pd.DataFrame) -> pd.Series:
    """
    Standalone = unique kickoff within a date-time slot OR special designations:
      - International games (often single-slot in the morning ET)
      - Black Friday (typically a single afternoon game)
      - Night games frequently are single-slot (but we compute uniqueness robustly)
    """
    # Build a minute-precision slot key in ET
    slot_key = ts_et.dt.tz_convert("America/New_York")
    slot_key = slot_key.dt.strftime("%Y-%m-%d %H:%M")  # minute-level window
    counts = slot_key.value_counts()
    unique_slot = slot_key.map(counts).fillna(0) == 1

    is_black_friday = holiday_df["situational_is_black_friday"] == 1

    standalone = (unique_slot | (is_international == 1) | is_black_friday).astype(int)
    return standalone


# --------------------------------------------------------------------------
# Main transform
# --------------------------------------------------------------------------
def compute_situational_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive situational/context features for each game (one row per game_id).

    Emits *numeric* model-ready columns prefixed with `situational_` and a few
    core flags used elsewhere. All features are pre-game, deterministic, and
    leakage-safe.
    """
    if games_df.empty:
        return pd.DataFrame()

    df = games_df.copy()

    # Ensure game_id present
    if "game_id" not in df.columns:
        logger.warning("situational: 'game_id' not found; returning empty.")
        return pd.DataFrame()

    # 1) Time foundations (UTC & ET)
    ts_utc, ts_et = _et_timestamp(df)
    dow = ts_et.dt.dayofweek  # Mon=0 .. Sun=6
    hour = ts_et.dt.hour

    # 2) Stage flags
    stage = df.get("stage", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    is_regular_season = (stage == "REG").astype(int)
    is_playoffs = (stage == "PST").astype(int)

    # 3) Division / conference flags
    is_division_game, is_conference_game, home_div, away_div = _division_conference(df)

    # 4) International & Neutral
    is_international = _detect_international(df)
    is_neutral_site = _detect_neutral(df, is_international)

    # 5) Holiday / special slates
    holiday_df = _holiday_flags(ts_et)

    # 6) Slot classification (ET) + one-hot
    slot = _slot_labels(hour, is_international.astype(bool))
    slot_1h = _one_hot_slot(slot)

    # 7) Primetime (refined): any night kickoff (>=19 ET)
    is_primetime = (hour >= 19).astype(int)

    # 8) Weekend flag (Sat/Sun)
    is_weekend = dow.isin([5, 6]).astype(int)

    # 9) Late-season proxy
    is_late_season = _late_season_mask(df, ts_et, is_regular_season)

    # 10) Standalone window
    is_standalone = _standalone_flag(ts_et, is_international, holiday_df)

    # 11) Interaction terms (late-season × rivalry)
    division_late = ((is_division_game == 1) & (is_late_season == 1)).astype(int)
    conference_late = ((is_conference_game == 1) & (is_late_season == 1)).astype(int)

    # 12) Assemble output (numeric only)
    out = pd.DataFrame({
        "game_id": df["game_id"],
        # legacy/core flags kept for compatibility:
        "is_primetime": is_primetime.astype(int),
        "is_weekend": is_weekend.astype(int),
        "is_regular_season": is_regular_season.astype(int),
        "is_playoffs": is_playoffs.astype(int),
        "is_division_game": is_division_game.astype(int),
        "is_conference_game": is_conference_game.astype(int),
        # new situational context:
        "situational_is_standalone": is_standalone.astype(int),
        "situational_is_international": is_international.astype(int),
        "situational_is_neutral_site": is_neutral_site.astype(int),
        "situational_is_late_season": is_late_season.astype(int),
        "situational_is_thanksgiving": holiday_df["situational_is_thanksgiving"].astype(int),
        "situational_is_black_friday": holiday_df["situational_is_black_friday"].astype(int),
        "situational_is_christmas": holiday_df["situational_is_christmas"].astype(int),
        "situational_is_christmas_eve": holiday_df["situational_is_christmas_eve"].astype(int),
        # slot one-hots
        "situational_slot_early": slot_1h["situational_slot_early"].astype(int),
        "situational_slot_late": slot_1h["situational_slot_late"].astype(int),
        "situational_slot_night": slot_1h["situational_slot_night"].astype(int),
        "situational_slot_intl_morning": slot_1h["situational_slot_intl_morning"].astype(int),
        # interactions
        "situational_division_late_season": division_late.astype(int),
        "situational_conference_late_season": conference_late.astype(int),
    })

    # Guarantee one row per game_id
    if out["game_id"].duplicated().any():
        dup_ct = int(out["game_id"].duplicated().sum())
        logger.warning("situational: %d duplicate game_id rows; keeping last", dup_ct)
        out = out.sort_values("game_id").drop_duplicates("game_id", keep="last")

    return out
