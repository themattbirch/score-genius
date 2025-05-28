# backend/nba_features/make_nba_snapshots.py

import os
import sys
from pathlib import Path

# ─── Make project and backend importable ──────────────────────────
HERE    = Path(__file__).resolve().parent       # .../backend/nba_features
BACKEND = HERE.parent                           # .../backend
PROJECT = BACKEND.parent                        # project root
for p in (PROJECT, BACKEND):
    path = str(p)
    if path not in sys.path:
        sys.path.insert(0, path)

# ─── Load .env via backend.config ────────────────────────────────
try:
    from backend.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
    print("✔ Config imported from backend.config")
except ImportError:
    SUPABASE_URL         = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

# ─── Now import dependencies ─────────────────────────────────────
import pandas as pd
from supabase import create_client
from .advanced import transform as advanced_transform
from .form    import transform as form_transform
from .h2h     import transform as h2h_transform
from .rest    import transform as rest_transform
from .rolling import transform as rolling_transform
from .season  import transform as season_transform

# ─── Initialize Supabase ─────────────────────────────────────────
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─── Helpers ─────────────────────────────────────────────────────
def fetch_raw_game(game_id: int) -> pd.DataFrame:
    """
    Return one-row DataFrame: box-score if available, else schedule pre-game.
    """
    r = sb.table("nba_historical_game_stats").select("*")\
           .eq("game_id", game_id).execute()
    df = pd.DataFrame(r.data or [])
    if not df.empty:
        return df
    r2 = sb.table("nba_game_schedule")\
           .select("game_id, game_date, scheduled_time, home_team, away_team")\
           .eq("game_id", game_id).execute()
    df2 = pd.DataFrame(r2.data or [])
    if "scheduled_time" in df2.columns:
        df2 = df2.rename(columns={"scheduled_time": "game_time_utc"})
    return df2


def fetch_full_history() -> pd.DataFrame:
    """
    Entire historical game stats for H2H and rolling features.
    """
    r = sb.table("nba_historical_game_stats").select("*").execute()
    return pd.DataFrame(r.data or [])


def fetch_team_season_stats() -> pd.DataFrame:
    """
    All teams' season stats (season_transform will pick correct season).
    """
    r = sb.table("nba_historical_team_stats").select("*").execute()
    return pd.DataFrame(r.data or [])


def fetch_season_adv_stats(season_year: int) -> pd.DataFrame:
    """
    Season-to-date advanced metrics for each team via RPC.
    """
    resp = sb.rpc("get_nba_advanced_team_stats", {"p_season_year": season_year}).execute()
    return pd.DataFrame(resp.data or [])


# ─── Snapshot generator ──────────────────────────────────────────
def make_nba_snapshot(game_id: int):
    # 1) Load raw, history, team stats & season RPC
    df_game      = fetch_raw_game(game_id)
    df_history   = fetch_full_history()
    df_team      = fetch_team_season_stats()
    season_year  = pd.to_datetime(df_game['game_date'].iloc[0]).year
    adv_all      = fetch_season_adv_stats(season_year)

    # ensure adv_all has a 'team_name' column
    if 'team_name' not in adv_all.columns:
        for c in adv_all.columns:
            if 'team' in c.lower() and 'name' in c.lower():
                adv_all.rename(columns={c: 'team_name'}, inplace=True)
                break

    # 2) Compute feature transforms
    df_adv    = advanced_transform(df_game)
    df_form   = form_transform(df_adv)
    df_rest   = rest_transform(df_form)
    df_h2h    = h2h_transform(df_rest, historical_df=df_history)
    df_season = season_transform(df_h2h, team_stats_df=df_team)
    df_roll   = rolling_transform(df_season)

    # 3) Override pre-game advanced metrics from adv_all
    if 'home_score' not in df_game.columns:
        home_nm = df_game.at[0, 'home_team']
        away_nm = df_game.at[0, 'away_team']
        # match season RPC table
        home_row = adv_all.loc[adv_all['team_name'] == home_nm]
        away_row = adv_all.loc[adv_all['team_name'] == away_nm]
        if not home_row.empty and not away_row.empty:
            home_row = home_row.iloc[0]
            away_row = away_row.iloc[0]
            for metric, cols in [
                ('pace', ('home_pace', 'away_pace')),
                ('off_rtg', ('home_offensive_rating', 'away_offensive_rating')),
                ('def_rtg', ('home_defensive_rating', 'away_defensive_rating')),
                ('efg_pct', ('home_efg_pct', 'away_efg_pct')),
                ('tov_pct', ('home_tov_rate', 'away_tov_rate')),
                ('oreb_pct', ('home_oreb_pct', 'away_oreb_pct')),
            ]:
                df_adv.at[0, cols[0]] = float(home_row[metric])
                df_adv.at[0, cols[1]] = float(away_row[metric])
            # recompute nets
            df_adv.at[0, 'home_net_rating']         = df_adv.home_offensive_rating.iloc[0] - df_adv.home_defensive_rating.iloc[0]
            df_adv.at[0, 'away_net_rating']         = df_adv.away_offensive_rating.iloc[0] - df_adv.away_defensive_rating.iloc[0]
            df_adv.at[0, 'efficiency_differential'] = df_adv.home_net_rating.iloc[0]       - df_adv.away_net_rating.iloc[0]

    # 4) Build payloads
    # HEADLINE: top diffs & advantages
    headline = [
        {'label': 'rest_advantage',        'value': int(df_rest.rest_advantage.iloc[0])},
        {'label': 'momentum_diff',         'value': float(df_form.momentum_diff.iloc[0])},
        {'label': 'form_win_pct_diff',     'value': float(df_form.form_win_pct_diff.iloc[0])},
        {'label': 'efficiency_differential','value': float(df_adv.efficiency_differential.iloc[0])},
        {'label': 'season_win_pct_diff',   'value': float(df_season.season_win_pct_diff.iloc[0])},
    ]

    # BAR: home quarter points
    bar = []
    for i in (1, 2, 3, 4):
        col = f'home_q{i}'
        if col in df_game.columns and pd.notna(df_game[col].iloc[0]):
            val = int(df_game[col].iloc[0])
        else:
            val = 0
        bar.append({'name': f'Q{i}', 'value': val})

    # RADAR: advanced five-metric spider
    radar = [
        {'metric': 'Pace',   'value': float(df_adv.home_pace.iloc[0])},
        {'metric': 'OffRtg', 'value': float(df_adv.home_offensive_rating.iloc[0])},
        {'metric': 'DefRtg', 'value': float(df_adv.home_defensive_rating.iloc[0])},
        {'metric': 'eFG%',   'value': float(df_adv.home_efg_pct.iloc[0])},
        {'metric': 'TOV%',   'value': float(df_adv.home_tov_rate.iloc[0])},
    ]

    # PIE: shot distribution
    twop  = int((df_adv.home_fg_made.iloc[0] - df_adv.home_3pm.iloc[0]) or 0)
    three = int(df_adv.home_3pm.iloc[0] or 0)
    ft    = int(df_adv.home_ft_made.iloc[0] or 0)
    pie = [
        {'category': '2P', 'value': twop,  'color': '#4ade80'},
        {'category': '3P', 'value': three, 'color': '#60a5fa'},
        {'category': 'FT', 'value': ft,    'color': '#fbbf24'},
    ]

    # 5) Upsert into Supabase
    sb.table('nba_snapshots').upsert({
        'game_id':        game_id,
        'headline_stats': headline,
        'bar_data':       bar,
        'radar_data':     radar,
        'pie_data':       pie,
    }).execute()

    print(f"✅ snapshot for {game_id} done")

# ─── CLI entry ───────────────────────────────────────────────────
if __name__ == '__main__':
    ids = [int(x) for x in sys.argv[1:]]
    for gid in ids:
        make_nba_snapshot(gid)
