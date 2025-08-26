# backend/tests/nfl_features/test_advanced.py
from __future__ import annotations
"""
Unit tests for backend.nfl_features.advanced.compute_advanced_metrics
"""

import numpy as np
import pandas as pd
import pytest

from backend.nfl_features import advanced


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_box_scores_df() -> pd.DataFrame:
    """Four team‑game box‑scores exercising normal, zero‑denominator and NaN paths."""
    rows = [
        # 1 Normal
        dict(
            game_id=1,
            team_id=10,
            team_score=28,
            plays_total=60,
            yards_total=400,
            total_drives=10,
            turnovers_total=1,
            third_down_made=5,
            third_down_attempts=10,
            fourth_down_made=1,
            fourth_down_attempts=1,
            red_zone_made=3,
            red_zone_att=4,
            penalties=5,
            penalty_yards=45,
            possession_time="32:00",
        ),
        # 2 Zero attempts (third‑down etc.)
        dict(
            game_id=2,
            team_id=20,
            team_score=10,
            plays_total=50,
            yards_total=200,
            total_drives=8,
            turnovers_total=0,
            third_down_made=0,
            third_down_attempts=0,
            fourth_down_made=0,
            fourth_down_attempts=0,
            red_zone_made=0,
            red_zone_att=0,
            penalties=0,
            penalty_yards=0,
            possession_time="28:00",
        ),
        # 3 NaNs
        dict(
            game_id=3,
            team_id=30,
            team_score=21,
            plays_total=55,
            yards_total=300,
            total_drives=np.nan,
            turnovers_total=2,
            third_down_made=4,
            third_down_attempts=12,
            fourth_down_made=0,
            fourth_down_attempts=1,
            red_zone_made=2,
            red_zone_att=3,
            penalties=3,
            penalty_yards=30,
            possession_time="30:00",
        ),
        # 4 Missing score (check optional column handling)
        dict(
            game_id=4,
            team_id=40,
            team_score=np.nan,
            plays_total=65,
            yards_total=350,
            total_drives=11,
            turnovers_total=3,
            third_down_made=6,
            third_down_attempts=15,
            fourth_down_made=1,
            fourth_down_attempts=2,
            red_zone_made=2,
            red_zone_att=5,
            penalties=8,
            penalty_yards=70,
            possession_time="35:00",
        ),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_happy_path_calcs(sample_box_scores_df):
    res = advanced.compute_advanced_metrics(sample_box_scores_df)
    r1 = res.loc[res["game_id"] == 1].iloc[0]

    # Core columns present
    for col in (
        "adv_third_down_pct",
        "adv_fourth_down_pct",
        "adv_red_zone_pct",
        "adv_plays_per_minute",
        "adv_yards_per_drive",
        "adv_turnover_rate_per_play",
        "adv_yards_per_penalty",
        "adv_points_per_100_yards",
        "possession_seconds",
    ):
        assert col in res.columns

    # Numerical checks
    assert np.isclose(r1["possession_seconds"], 32 * 60)
    assert np.isclose(r1["adv_third_down_pct"], 0.5)
    assert np.isclose(r1["adv_fourth_down_pct"], 1.0)
    assert np.isclose(r1["adv_red_zone_pct"], 0.75)
    assert np.isclose(r1["adv_plays_per_minute"], 60 / 32.0)
    assert np.isclose(r1["adv_yards_per_drive"], 40.0)
    assert np.isclose(r1["adv_turnover_rate_per_play"], 1 / 60)
    assert np.isclose(r1["adv_yards_per_penalty"], 45 / 5)
    assert np.isclose(r1["adv_points_per_100_yards"], (28 / 400) * 100)


def test_zero_denominator_defaults(sample_box_scores_df):
    res = advanced.compute_advanced_metrics(sample_box_scores_df)
    r2 = res.loc[res["game_id"] == 2].iloc[0]

    assert r2["adv_third_down_pct"] == 0.0
    assert r2["adv_red_zone_pct"] == 0.0
    assert r2["adv_yards_per_penalty"] == 0.0


def test_nan_inputs_default(sample_box_scores_df):
    res = advanced.compute_advanced_metrics(sample_box_scores_df)
    r3 = res.loc[res["game_id"] == 3].iloc[0]
    # NaN total_drives → yards_per_drive default 0
    assert r3["adv_yards_per_drive"] == 0.0


def test_optional_points_per_yard_omitted(sample_box_scores_df):
    df_no_score = sample_box_scores_df.drop(columns=["team_score"])
    res = advanced.compute_advanced_metrics(df_no_score)
    assert "adv_points_per_100_yards" not in res.columns


def test_empty_input():
    assert advanced.compute_advanced_metrics(pd.DataFrame()).empty
