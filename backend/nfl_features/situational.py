# backend/nfl_features/situational.py
from __future__ import annotations

import logging
from typing import Mapping, Optional, Tuple

import pandas as pd

from . import utils

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["compute_situational_features"]

# --------------------------------------------------------------------------
# Division / conference map (canonicalized team shortnames)
# (keys must match utils.normalize_team_name(...).lower())
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
    # Safe aliases (only if your normalizer might output these)
    "san francisco 49ers": "NFC West",
}

# --------------------------------------------------------------------------
# International venues (names: use substring tokens; countries list expanded)
# --------------------------------------------------------------------------
INTERNATIONAL_VENUE_TOKENS = {
    # UK
    "wembley", "tottenham", "twickenham",
    # Germany
    "allianz arena", "munich", "deutsche bank park", "frankfurt",
    # Mexico
    "estadio azteca",
    # Brazil (São Paulo)
    "sao paulo", "corinthians",
    # Spain (Madrid)
    "madrid", "bernabeu", "santiago bernabéu", "santiago bernabeu",
}
INTERNATIONAL_COUNTRIES = {
    "uk", "united kingdom", "england",
    "germany", "mexico",
    "brazil", "spain",
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _et_timestamp(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Build kickoff timestamps with robust fallbacks:
      1) kickoff_ts (assumed UTC or parseable to UTC)
      2) game_timestamp (assumed UTC or parseable to UTC)
      3) game_date + game_time as ET (then → UTC)
      4) game_date + 12:00 ET (then → UTC)

    Returns (ts_utc, ts_et), always tz-aware.
    """
    idx = df.index

    # 1) kickoff_ts
    if "kickoff_ts" in df.columns:
        try:
            ko_utc = pd.to_datetime(df["kickoff_ts"], errors="coerce", utc=True)
        except Exception:
            ko_utc = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    else:
        ko_utc = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")

    # 2) game_timestamp
    if "game_timestamp" in df.columns:
        try:
            gt_utc = pd.to_datetime(df["game_timestamp"], errors="coerce", utc=True)
        except Exception:
            gt_utc = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    else:
        gt_utc = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")

    # 3) game_date + game_time (assume ET)
    date_str = df.get("game_date", pd.Series("", index=idx)).astype(str)
    time_str = df.get("game_time", pd.Series("00:00:00", index=idx)).astype(str)
    dt_combo = (date_str + " " + time_str).str.strip()
    try:
        # parse naive → localize ET → to UTC
        combo_naive = pd.to_datetime(dt_combo, errors="coerce")
        combo_et = combo_naive.dt.tz_localize(
            "America/New_York", ambiguous="infer", nonexistent="shift_forward"
        )
        combo_utc = combo_et.dt.tz_convert("UTC")
    except Exception:
        combo_utc = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")

    # 4) game_date + 12:00 ET
    try:
        date_only = pd.to_datetime(df.get("game_date", pd.Series("", index=idx)), errors="coerce")
        midday_et = date_only.dt.tz_localize(
            "America/New_York", ambiguous="infer", nonexistent="shift_forward"
        ) + pd.Timedelta(hours=12)
        midday_utc = midday_et.dt.tz_convert("UTC")
    except Exception:
        midday_utc = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")

    ts_utc = ko_utc.fillna(gt_utc).fillna(combo_utc).fillna(midday_utc)
    ts_et = ts_utc.dt.tz_convert("America/New_York")
    return ts_utc, ts_et


def _is_thanksgiving(d: pd.Timestamp) -> bool:
    """NFL Thanksgiving: 4th Thursday of November (ET)."""
    if pd.isna(d):
        return False
    d_et = d.tz_convert("America/New_York") if d.tzinfo is not None else d.tz_localize("America/New_York")
    year = d_et.year
    nov1 = pd.Timestamp(year=year, month=11, day=1, tz="America/New_York")
    # Thursday = 3
    offset = (3 - nov1.dayofweek) % 7
    fourth_thu = nov1 + pd.Timedelta(days=offset + 21)
    return d_et.normalize() == fourth_thu.normalize()


def _is_black_friday(d: pd.Timestamp) -> bool:
    """Day after Thanksgiving (ET)."""
    if pd.isna(d):
        return False
    d_et = d.tz_convert("America/New_York") if d.tzinfo is not None else d.tz_localize("America/New_York")
    return _is_thanksgiving(d_et - pd.Timedelta(days=1))


def _holiday_flags(ts_et: pd.Series) -> pd.DataFrame:
    """Return boolean flags for holiday/special slates (ET)."""
    out = pd.DataFrame(index=ts_et.index)
    out["situational_is_thanksgiving"] = ts_et.apply(_is_thanksgiving).astype("int8")
    out["situational_is_black_friday"] = ts_et.apply(_is_black_friday).astype("int8")
    month = ts_et.dt.month
    day = ts_et.dt.day
    out["situational_is_christmas"] = ((month == 12) & (day == 25)).astype("int8")
    out["situational_is_christmas_eve"] = ((month == 12) & (day == 24)).astype("int8")
    out["situational_is_new_years_day"] = ((month == 1) & (day == 1)).astype("int8")
    return out


def _slot_labels(hour_et: pd.Series, is_international: pd.Series) -> pd.Series:
    """
    Classify kickoff slot (ET) with clear priority:
      intl_morning > night (>=19) > late (16–18) > early (else).
    """
    slot = pd.Series("early", index=hour_et.index, dtype="object")
    intl = is_international.astype(bool)
    slot = slot.mask(intl & (hour_et < 13), "intl_morning")
    slot = slot.mask((hour_et >= 19), "night")
    slot = slot.mask((hour_et >= 16) & (hour_et < 19), "late")
    return slot


def _one_hot_slot(slot: pd.Series) -> pd.DataFrame:
    """One-hot numeric flags for model consumption (mutually exclusive by design)."""
    out = pd.DataFrame(index=slot.index)
    out["situational_slot_early"] = (slot == "early").astype("int8")
    out["situational_slot_late"] = (slot == "late").astype("int8")
    out["situational_slot_night"] = (slot == "night").astype("int8")
    out["situational_slot_intl_morning"] = (slot == "intl_morning").astype("int8")
    return out


def _detect_international(df: pd.DataFrame) -> pd.Series:
    """
    Detect international games using venue hints:
      - venue_country (non-US)
      - venue_name contains token from INTERNATIONAL_VENUE_TOKENS
    """
    name = df.get("venue_name", pd.Series("", index=df.index)).astype(str).str.lower().str.strip()
    country = df.get("venue_country", pd.Series("", index=df.index)).astype(str).str.lower().str.strip()

    is_intl_by_country = (country != "") & (~country.isin({"us", "usa", "united states", "u.s.", "u.s.a."}))
    is_intl_by_country |= country.isin(INTERNATIONAL_COUNTRIES)

    # substring token match
    name_tokens = pd.Series(False, index=name.index)
    if not name.empty:
        for tok in INTERNATIONAL_VENUE_TOKENS:
            name_tokens = name_tokens | name.str.contains(tok, na=False)

    return ((is_intl_by_country | name_tokens).astype("int8"))


def _detect_neutral(df: pd.DataFrame, is_international: pd.Series) -> pd.Series:
    """
    Neutral-site heuristic:
      - explicit 'is_neutral_site' if provided (truthy)
      - OR international
      - OR venue_home_team_norm exists and != home_team_norm
      - OR (optional) venue_city_norm exists and differs from both home/away city norms
    """
    idx = df.index
    base = df.get("is_neutral_site", pd.Series(0, index=idx)).fillna(0).astype(int)

    venue_home = df.get("venue_home_team_norm", pd.Series("", index=idx)).astype(str).str.lower()
    home_team = df.get("home_team_norm", pd.Series("", index=idx)).astype(str).str.lower()

    by_owner_mismatch = ((venue_home != "") & (home_team != "") & (venue_home != home_team)).astype(int)

    # Optional city-based hint (only if columns exist)
    v_city = df.get("venue_city_norm", pd.Series("", index=idx)).astype(str).str.lower()
    h_city = df.get("home_city_norm", pd.Series("", index=idx)).astype(str).str.lower()
    a_city = df.get("away_city_norm", pd.Series("", index=idx)).astype(str).str.lower()
    by_city_mismatch = ((v_city != "") & (v_city != h_city) & (v_city != a_city)).astype(int)

    neutral = ((base == 1) | (by_owner_mismatch == 1) | (is_international == 1) | (by_city_mismatch == 1)).astype("int8")
    return neutral


def _late_season_mask(df: pd.DataFrame, ts_et: pd.Series, is_regular_season: pd.Series) -> pd.Series:
    """
    Late-season proxy:
      - If 'week' present: week >= 15 during regular season.
      - Else: REG-season games in December or later.
    """
    if "week" in df.columns:
        wk = pd.to_numeric(df["week"], errors="coerce")
        return ((is_regular_season.astype(bool)) & (wk >= 15)).fillna(False).astype("int8")
    month = ts_et.dt.month
    return ((is_regular_season.astype(bool)) & (month >= 12)).astype("int8")


def _division_conference(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    home = df.get("home_team_norm", pd.Series("", index=df.index)).apply(utils.normalize_team_name).str.lower()
    away = df.get("away_team_norm", pd.Series("", index=df.index)).apply(utils.normalize_team_name).str.lower()

    home_div = home.map(TEAM_DIVISIONS)
    away_div = away.map(TEAM_DIVISIONS)

    # Conf via division prefix (e.g., "NFC East" → "NFC")
    home_conf = home_div.str.split().str[0]
    away_conf = away_div.str.split().str[0]

    # Safe comparisons (require non-null)
    is_division_game = ((home_div.notna()) & (away_div.notna()) & (home_div == away_div)).astype("int8")
    is_conference_game = (
        (home_conf.notna()) & (away_conf.notna()) &
        (home_conf == away_conf) & (is_division_game == 0)
    ).astype("int8")

    return is_division_game, is_conference_game, home_div, away_div


def _standalone_flag(ts_et: pd.Series, is_international: pd.Series, holiday_df: pd.DataFrame) -> pd.Series:
    """
    Standalone = unique kickoff within a minute-precision slot (ET) OR special designations.
    """
    # Guard NaT rows as non-standalone
    valid = ts_et.notna()
    slot_key = pd.Series("", index=ts_et.index)
    slot_key.loc[valid] = ts_et.loc[valid].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M")

    counts = slot_key.value_counts()
    unique_slot = slot_key.map(counts).fillna(0).astype(int) == 1

    is_black_friday = (holiday_df["situational_is_black_friday"] == 1)
    standalone = (unique_slot | (is_international.astype(bool)) | is_black_friday).astype("int8")
    standalone = standalone.where(valid, 0).astype("int8")
    return standalone

# --------------------------------------------------------------------------
# Main transform
# --------------------------------------------------------------------------
def compute_situational_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive situational/context features for each game (one row per game_id).

    Emits numeric, model-ready columns:
      - legacy/core: is_primetime, is_weekend, is_regular_season, is_playoffs,
                     is_division_game, is_conference_game
      - situational_*: international, neutral, late-season, holiday flags,
                       kickoff slot one-hots, standalone,
                       interactions (division/conference × late-season)
      - helpers: day_of_week (0–6 ET), situational_kickoff_hour_et (0–23)
    """
    if games_df.empty:
        return pd.DataFrame()

    df = games_df.copy()

    if "game_id" not in df.columns:
        logger.warning("situational: 'game_id' not found; returning empty.")
        return pd.DataFrame()

    # 1) Time foundations (UTC & ET)
    ts_utc, ts_et = _et_timestamp(df)
    dow = ts_et.dt.dayofweek.astype("int8")  # 0=Mon .. 6=Sun
    hour = ts_et.dt.hour.astype("int8")

    # 2) Stage flags (REG/PST)
    stage = df.get("stage", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    is_regular_season = (stage == "REG").astype("int8")
    is_playoffs = (stage == "PST").astype("int8")

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

    # 7) Primetime (night window ≥ 19 ET)
    is_primetime = (hour >= 19).astype("int8")

    # 8) Weekend flag (Sat/Sun)
    is_weekend = dow.isin([5, 6]).astype("int8")

    # 9) Late-season proxy
    is_late_season = _late_season_mask(df, ts_et, is_regular_season)

    # 10) Standalone window
    is_standalone = _standalone_flag(ts_et, is_international, holiday_df)

    # 11) Interaction terms (late-season × rivalry)
    division_late = ((is_division_game == 1) & (is_late_season == 1)).astype("int8")
    conference_late = ((is_conference_game == 1) & (is_late_season == 1)).astype("int8")

    # 12) Assemble output (numeric only)
    out = pd.DataFrame({
        "game_id": df["game_id"],
        # helpers
        "day_of_week": dow,                                  # 0..6
        "situational_kickoff_hour_et": hour,                 # 0..23
        # legacy/core flags
        "is_primetime": is_primetime,
        "is_weekend": is_weekend,
        "is_regular_season": is_regular_season,
        "is_playoffs": is_playoffs,
        "is_division_game": is_division_game,
        "is_conference_game": is_conference_game,
        # situational context
        "situational_is_standalone": is_standalone,
        "situational_is_international": is_international,
        "situational_is_neutral_site": is_neutral_site,
        "situational_is_late_season": is_late_season,
        "situational_is_thanksgiving": holiday_df["situational_is_thanksgiving"].astype("int8"),
        "situational_is_black_friday": holiday_df["situational_is_black_friday"].astype("int8"),
        "situational_is_christmas": holiday_df["situational_is_christmas"].astype("int8"),
        "situational_is_christmas_eve": holiday_df["situational_is_christmas_eve"].astype("int8"),
        "situational_is_new_years_day": holiday_df["situational_is_new_years_day"].astype("int8"),
        # slot one-hots
        "situational_slot_early": slot_1h["situational_slot_early"].astype("int8"),
        "situational_slot_late": slot_1h["situational_slot_late"].astype("int8"),
        "situational_slot_night": slot_1h["situational_slot_night"].astype("int8"),
        "situational_slot_intl_morning": slot_1h["situational_slot_intl_morning"].astype("int8"),
        # interactions
        "situational_division_late_season": division_late,
        "situational_conference_late_season": conference_late,
    })

    # Guarantee one row per game_id
    if out["game_id"].duplicated().any():
        dup_ct = int(out["game_id"].duplicated().sum())
        logger.warning("situational: %d duplicate game_id rows; keeping last", dup_ct)
        out = out.sort_values("game_id").drop_duplicates("game_id", keep="last")

    return out
