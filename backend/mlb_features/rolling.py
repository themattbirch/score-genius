# backend/mlb_features/rolling.py
from __future__ import annotations

"""
Calculates leakage-free rolling mean features for MLB games.
This version has been simplified for robustness and clarity.
-- MODIFIED to accept a pre-computed DataFrame of rolling features. --
"""
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def transform(
    df: pd.DataFrame,
    *,
    precomputed_rolling_features_df: Optional[pd.DataFrame] = None,
    window_sizes: List[int] = [15, 30, 60, 100],
    debug: bool = False,
    **kwargs,
) -> pd.DataFrame:
    if precomputed_rolling_features_df is not None and not precomputed_rolling_features_df.empty:
        out = df.copy()
        feats = precomputed_rolling_features_df.copy()

        # Convert team_id once, downcast to smallest int
        feats['team_id'] = pd.to_numeric(feats['team_id'], errors='coerce').dropna().astype('int16')
        feat_cols = [c for c in feats.columns if c.startswith('rolling_')]
        feats = feats[['team_id', *feat_cols]].drop_duplicates('team_id').set_index('team_id')

        # Filter to only the teams present in this chunk
        teams = pd.to_numeric(out['home_team_id'], errors='coerce').dropna().astype('int16')
        teams = teams.append(pd.to_numeric(out['away_team_id'], errors='coerce').dropna().astype('int16'))
        feats = feats.loc[feats.index.intersection(teams.unique())]

        # Build home and away feature tables once
        home_feats = feats.add_prefix('home_')
        away_feats = feats.add_prefix('away_')

        # Ensure keys are ints
        out['home_team_id'] = pd.to_numeric(out['home_team_id'], errors='coerce').astype('int16')
        out['away_team_id'] = pd.to_numeric(out['away_team_id'], errors='coerce').astype('int16')

        # Two fast index-joins instead of two merges
        out = out.join(home_feats, on='home_team_id')
        out = out.join(away_feats, on='away_team_id')

        return out
        # --- END of new logic ---

    # --- ORIGINAL LOGIC (now serves as a fallback) ---
    else:
        logger.info("No pre-computed rolling features provided. Calculating from scratch...")
        
        out = df.copy()
        # The 'historical_df' alias is for backward compatibility if other scripts call it this way
        if 'historical_df' in kwargs and kwargs['historical_df'] is not None:
             out = pd.concat([kwargs['historical_df'], out], ignore_index=True)

        stat_map = {
            "runs_scored": ("home_score", "away_score"),
            "runs_allowed": ("away_score", "home_score"),
            "hits_for": ("home_hits", "away_hits"),
        }

        # Ensure columns exist
        for home_col, away_col in stat_map.values():
            if home_col not in out.columns: out[home_col] = pd.NA
            if away_col not in out.columns: out[away_col] = pd.NA

        # Build long format
        home_log = out[["game_id","game_date_et","home_team_norm","home_score","away_score","home_hits","away_hits"]].rename(columns={"home_team_norm":"team_norm"})
        home_log["is_home"] = True
        away_log = out[["game_id","game_date_et","away_team_norm","home_score","away_score","home_hits","away_hits"]].rename(columns={"away_team_norm":"team_norm"})
        away_log["is_home"] = False
        long_df = pd.concat([home_log, away_log], ignore_index=True)

        # Map stats
        for stat, (h_col, a_col) in stat_map.items():
            long_df[stat] = pd.to_numeric(
                np.where(long_df["is_home"], long_df[h_col], long_df[a_col]),
                errors="coerce",
            )

        long_df.sort_values(by=["team_norm","game_date_et","game_id"], inplace=True)

        # Rolling means
        all_roll = []
        for stat in stat_map:
            for w in window_sizes:
                col = f"rolling_{stat}_mean_{w}"
                all_roll.append(col)
                # Ensure the column is numeric before rolling
                long_df[stat] = pd.to_numeric(long_df[stat], errors='coerce')
                long_df[col] = (
                    long_df.groupby("team_norm")[stat]
                    .shift(1)
                    .rolling(window=w, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True) # Avoid index alignment issues
                )

        # Pivot back to one row per game
        pivoted = long_df.pivot_table(index="game_id", columns="team_norm", values=all_roll)
        if not pivoted.empty:
            pivoted.columns = ["_".join(str(c) for c in col).strip() for col in pivoted.columns.values]
            result = pd.merge(df, pivoted, on="game_id", how="left")
        else:
            result = df.copy() # If pivot is empty, just return the original df

        # Apply home/away prefixes
        final_cols_to_add = {}
        if not pivoted.empty:
            for idx, row in result.iterrows():
                home, away = row.get("home_team_norm"), row.get("away_team_norm")
                if pd.isna(home) or pd.isna(away): continue
                
                for col in pivoted.columns:
                    if f"_{home}" in col:
                        new_col_name = f"home_{col.replace(f'_{home}', '')}"
                        if new_col_name not in final_cols_to_add: final_cols_to_add[new_col_name] = [np.nan] * len(result)
                        final_cols_to_add[new_col_name][idx] = row[col]

                    if f"_{away}" in col:
                        new_col_name = f"away_{col.replace(f'_{away}', '')}"
                        if new_col_name not in final_cols_to_add: final_cols_to_add[new_col_name] = [np.nan] * len(result)
                        final_cols_to_add[new_col_name][idx] = row[col]
            
            for new_col, data in final_cols_to_add.items():
                result[new_col] = data

            result.drop(columns=pivoted.columns, inplace=True, errors="ignore")

        return result