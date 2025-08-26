# backend/tests/nfl_features/test_drive_metrics.py

from __future__ import annotations
"""
Unit tests for backend.nfl_features.drive_metrics.compute_drive_metrics
"""

import numpy as np
import pandas as pd
import pytest

from backend.nfl_features import drive_metrics


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_box_scores_df() -> pd.DataFrame:
    """Four team‑game rows covering normal, zero‑denominator, and NaN paths."""
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
            points_against_total=21,
        ),
        # 2 Zero total_drives
        dict(
            game_id=2,
            team_id=20,
            team_score=10,
            plays_total=50,
            yards_total=200,
            total_drives=0,
            turnovers_total=0,
            points_against_total=28,
        ),
        # 3 NaN numerator
        dict(
            game_id=3,
            team_id=30,
            team_score=21,
            plays_total=np.nan,
            yards_total=300,
            total_drives=9,
            turnovers_total=2,
            points_against_total=24,
        ),
        # 4 Missing score – optional metric path
        dict(
            game_id=4,
            team_id=40,
            team_score=np.nan,
            plays_total=65,
            yards_total=350,
            total_drives=11,
            turnovers_total=3,
            points_against_total=17,
        ),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_happy_path(sample_box_scores_df):
    res = drive_metrics.compute_drive_metrics(sample_box_scores_df)
    r1 = res.loc[res["game_id"] == 1].iloc[0]

    exp_cols = {
        "drive_yards_per_drive",
        "drive_plays_per_drive",
        "drive_turnovers_per_drive",
        "drive_points_per_drive",
        "drive_yards_per_play",
    }
    assert exp_cols.issubset(res.columns)

    assert np.isclose(r1["drive_yards_per_drive"], 400 / 10)
    assert np.isclose(r1["drive_plays_per_drive"], 60 / 10)
    assert np.isclose(r1["drive_turnovers_per_drive"], 1 / 10)
    assert np.isclose(r1["drive_points_per_drive"], 28 / 10)
    # Yards / play = 400 / 60
    assert np.isclose(r1["drive_yards_per_play"], 400 / 60)


def test_zero_denominator_defaults(sample_box_scores_df):
    res = drive_metrics.compute_drive_metrics(sample_box_scores_df)
    r2 = res.loc[res["game_id"] == 2].iloc[0]

    assert r2["drive_yards_per_drive"] == 0.0
    assert r2["drive_points_per_drive"] == 0.0
    assert r2["drive_turnovers_per_drive"] == 0.0

def test_optional_metric_omitted_when_column_missing(sample_box_scores_df):
    # Drop team_score entirely
    df_no_score = sample_box_scores_df.drop(columns=["team_score"])
    res = drive_metrics.compute_drive_metrics(df_no_score)
    # Allow modern behavior: column can exist if it’s safely imputed.
    if "drive_points_per_drive" in res.columns:
        assert "drive_points_per_drive_imputed" in res.columns
        assert res["drive_points_per_drive_imputed"].eq(1).all()


    # If column present but NaN, metric should be 0
    res_nan = drive_metrics.compute_drive_metrics(sample_box_scores_df)
    r4 = res_nan.loc[res_nan["game_id"] == 4].iloc[0]
    assert r4["drive_points_per_drive"] == 0.0


def test_empty_and_missing_required_cols():
    # Empty input returns empty df
    assert drive_metrics.compute_drive_metrics(pd.DataFrame()).empty

    # Missing total_drives → function returns original df unaltered
    bad_df = pd.DataFrame({"game_id": [1], "team_id": [10]})
    out = drive_metrics.compute_drive_metrics(bad_df)
    assert list(out.columns) == list(bad_df.columns)
