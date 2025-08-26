# backend/tests/nba_features/test_advanced.py
import numpy as np
import pandas as pd
import pytest

import os
import sys

# ensure project root is on PYTHONPATH so "backend" can be imported
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    ),
)

from backend.nba_features import advanced, utils
from backend.nba_features.utils import normalize_team_name
print(f"Normalized 'Team Alpha': '{normalize_team_name('Team Alpha')}'") # Should be 'teamalpha'
print(f"Normalized 'Team Beta': '{normalize_team_name('Team Beta')}'")   # Should be 'teambeta'


# -------------------------------------------------------------
# Test Configuration / Expected Data
# -------------------------------------------------------------

MOCK_TEAMS_DATA = {
    "teamalpha": {"pace": 95.0, "off_rtg": 110.0, "def_rtg": 105.0, "net_rtg": 5.0,
                  "efg_pct": 0.52, "tov_pct": 0.14, "oreb_pct": 0.28, "ft_rate": 0.22},
    "teambeta":  {"pace": 93.0, "off_rtg": 108.0, "def_rtg": 107.0, "net_rtg": 1.0,
                  "efg_pct": 0.50, "tov_pct": 0.15, "oreb_pct": 0.26, "ft_rate": 0.20}
}
HOME_STAT_ADJUSTMENT = 0.01
AWAY_STAT_ADJUSTMENT = -0.01 # Note: The original fixture uses + AWAY_STAT_ADJUSTMENT.
                             # If it should be subtraction, this constant is fine, but fixture applies it as addition.
                             # Corrected in mock_seasonal_stats_df fixture below for clarity.
TEST_SEASON = "2023"

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

ADV_COLS_SEASONAL = []
if hasattr(advanced, 'EXPECTED_STATS'):
    for stat_base in advanced.EXPECTED_STATS:
        ADV_COLS_SEASONAL.append(f"h_{stat_base}_home")
        ADV_COLS_SEASONAL.append(f"a_{stat_base}_away")
        ADV_COLS_SEASONAL.append(f"hist_{stat_base}_split_diff")
        ADV_COLS_SEASONAL.append(f"h_{stat_base}_home_imputed")
        ADV_COLS_SEASONAL.append(f"a_{stat_base}_away_imputed")
else:
    print("Warning: advanced.EXPECTED_STATS not found, ADV_COLS_SEASONAL may be incomplete.")


def _assert_adv_cols(df):
    if not ADV_COLS_SEASONAL:
        pytest.fail("ADV_COLS_SEASONAL is empty. Check advanced.EXPECTED_STATS loading.")
    missing_cols = [c for c in ADV_COLS_SEASONAL if c not in df.columns]
    assert not missing_cols, f"Missing expected advanced columns: {missing_cols}"

# -------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------

@pytest.fixture
def sample_game_df():
    """Input game DataFrame, now including 'adv_stats_lookup_season'."""
    df = pd.DataFrame({
        "game_id": [1],
        "home_team": ["Team Alpha"],
        "away_team": ["Team Beta"],
        "home_score": [100], "away_score": [90],
        "home_fg_made": [40], "home_fg_attempted": [80],
        "away_fg_made": [35], "away_fg_attempted": [78],
        "home_3pm": [10], "home_3pa": [25],
        "away_3pm": [8], "away_3pa": [22],
        "home_ft_made": [10], "home_ft_attempted": [12],
        "away_ft_made": [12], "away_ft_attempted": [15],
        "home_off_reb": [10], "home_def_reb": [25], "home_total_reb": [35],
        "away_off_reb": [12], "away_def_reb": [23], "away_total_reb": [35],
        "home_turnovers": [15], "away_turnovers": [12],
        "home_ot": [0], "away_ot": [0],
    })
    df["adv_stats_lookup_season"] = TEST_SEASON # Crucial for advanced.transform
    return df

@pytest.fixture
def mock_seasonal_stats_df():
    """Mock seasonal stats with complete data for 'all_historical_splits_df'."""
    data_rows = []
    for norm_name_key, stats_dict in MOCK_TEAMS_DATA.items():
        row = {
            'season': TEST_SEASON, # Season the stats pertain to
            'team_name': norm_name_key.capitalize(),
            'team_norm': norm_name_key # Matches normalize_team_name output
        }
        for stat_base_name in advanced.EXPECTED_STATS:
            base_val = stats_dict.get(stat_base_name, utils.DEFAULTS.get(stat_base_name, 0.0))
            row[f"{stat_base_name}_home"] = base_val + HOME_STAT_ADJUSTMENT
            # Applying AWAY_STAT_ADJUSTMENT, e.g. base_val - 0.01
            row[f"{stat_base_name}_away"] = base_val + AWAY_STAT_ADJUSTMENT
        data_rows.append(row)
    return pd.DataFrame(data_rows)

@pytest.fixture
def mock_seasonal_stats_df_missing_efg():
    """Mock seasonal stats where Team Alpha is missing efg_pct data."""
    data_rows = []
    for norm_name_key, stats_dict in MOCK_TEAMS_DATA.items():
        row = {
            'season': TEST_SEASON,
            'team_name': norm_name_key.capitalize(),
            'team_norm': norm_name_key
        }
        for stat_base_name in advanced.EXPECTED_STATS:
            base_val = stats_dict.get(stat_base_name, utils.DEFAULTS.get(stat_base_name, 0.0))
            if norm_name_key == "teamalpha" and stat_base_name == "efg_pct":
                row[f"{stat_base_name}_home"] = None # Will trigger imputation
                row[f"{stat_base_name}_away"] = None # Will trigger imputation
            else:
                row[f"{stat_base_name}_home"] = base_val + HOME_STAT_ADJUSTMENT
                row[f"{stat_base_name}_away"] = base_val + AWAY_STAT_ADJUSTMENT
        data_rows.append(row)
    return pd.DataFrame(data_rows)

# -------------------------------------------------------------
# Tests
# -------------------------------------------------------------

def test_advanced_metrics_math(sample_game_df, mock_seasonal_stats_df):
    """Tests that historical seasonal stats are correctly merged and differentials calculated."""
    # CORRECTED CALL to advanced.transform:
    out = advanced.transform(
        sample_game_df.copy(), # sample_game_df now includes 'adv_stats_lookup_season'
        all_historical_splits_df=mock_seasonal_stats_df, # Changed name, removed 'season' arg
        flag_imputations=True,
        debug=False
    )
    
    _assert_adv_cols(out)
    assert not out.empty, "Output DataFrame should not be empty"
    r = out.iloc[0]

    for stat_base in ["efg_pct", "pace", "off_rtg"]:
        expected_h_stat_home = MOCK_TEAMS_DATA["teamalpha"][stat_base] + HOME_STAT_ADJUSTMENT
        assert np.isclose(r[f"h_{stat_base}_home"], expected_h_stat_home), \
            f"Mismatch for h_{stat_base}_home. Expected {expected_h_stat_home}, got {r[f'h_{stat_base}_home']}"
        assert not r[f"h_{stat_base}_home_imputed"], \
            f"h_{stat_base}_home_imputed should be False for {stat_base}"

        expected_a_stat_away = MOCK_TEAMS_DATA["teambeta"][stat_base] + AWAY_STAT_ADJUSTMENT
        assert np.isclose(r[f"a_{stat_base}_away"], expected_a_stat_away), \
            f"Mismatch for a_{stat_base}_away. Expected {expected_a_stat_away}, got {r[f'a_{stat_base}_away']}"
        assert not r[f"a_{stat_base}_away_imputed"], \
            f"a_{stat_base}_away_imputed should be False for {stat_base}"
            
        expected_diff = expected_h_stat_home - expected_a_stat_away
        assert np.isclose(r[f"hist_{stat_base}_split_diff"], expected_diff), \
            f"Mismatch for hist_{stat_base}_split_diff. Expected {expected_diff}, got {r[f'hist_{stat_base}_split_diff']}"

    h_efg_home = MOCK_TEAMS_DATA["teamalpha"]["efg_pct"] + HOME_STAT_ADJUSTMENT
    a_efg_away = MOCK_TEAMS_DATA["teambeta"]["efg_pct"] + AWAY_STAT_ADJUSTMENT
    expected_efg_diff = h_efg_home - a_efg_away
    assert np.isclose(r["hist_efg_pct_split_diff"], expected_efg_diff)


def test_default_value_fallback(sample_game_df, mock_seasonal_stats_df_missing_efg):
    """
    Tests fallback to default value and imputation flag setting for missing/invalid stats.
    """
    # CORRECTED CALL to advanced.transform:
    out = advanced.transform(
        sample_game_df.copy(), # sample_game_df now includes 'adv_stats_lookup_season'
        all_historical_splits_df=mock_seasonal_stats_df_missing_efg, # Changed name, removed 'season' arg
        flag_imputations=True,
        debug=False
    )

    _assert_adv_cols(out)
    assert not out.empty, "Output DataFrame should not be empty"
    r = out.iloc[0]

    default_efg = utils.DEFAULTS.get("efg_pct", 0.0)
    assert np.isclose(r["h_efg_pct_home"], default_efg), \
        f"h_efg_pct_home should be default. Expected {default_efg}, got {r['h_efg_pct_home']}"
    assert r["h_efg_pct_home_imputed"], \
        "h_efg_pct_home_imputed should be True when default is used"

    expected_a_efg_away = MOCK_TEAMS_DATA["teambeta"]["efg_pct"] + AWAY_STAT_ADJUSTMENT
    assert np.isclose(r["a_efg_pct_away"], expected_a_efg_away), \
        f"a_efg_pct_away for Team Beta. Expected {expected_a_efg_away}, got {r['a_efg_pct_away']}"
    assert not r["a_efg_pct_away_imputed"], \
        "a_efg_pct_away_imputed should be False for Team Beta"
        
    expected_diff_with_default = default_efg - expected_a_efg_away
    assert np.isclose(r["hist_efg_pct_split_diff"], expected_diff_with_default), \
        f"hist_efg_pct_split_diff with default. Expected {expected_diff_with_default}, got {r['hist_efg_pct_split_diff']}"


def test_missing_columns_graceful(mock_seasonal_stats_df):
    """
    Tests transform with minimal input game_df (but with required lookup keys).
    """
    # df_minimal now includes 'adv_stats_lookup_season'
    df_minimal = pd.DataFrame({
        "game_id": [3],
        "home_team": ["Team Alpha"],
        "away_team": ["Team Beta"],
        "adv_stats_lookup_season": [TEST_SEASON] # Crucial for advanced.transform
    })

    # CORRECTED CALL to advanced.transform:
    out = advanced.transform(
        df_minimal.copy(),
        all_historical_splits_df=mock_seasonal_stats_df, # Changed name, removed 'season' arg
        flag_imputations=True,
        debug=False
    )

    _assert_adv_cols(out)
    assert not out.empty, "Output DataFrame should not be empty"
    r = out.iloc[0]

    expected_h_pace_home = MOCK_TEAMS_DATA["teamalpha"]["pace"] + HOME_STAT_ADJUSTMENT
    assert np.isclose(r["h_pace_home"], expected_h_pace_home)
    assert not r["h_pace_home_imputed"]

    expected_a_net_rtg_away = MOCK_TEAMS_DATA["teambeta"]["net_rtg"] + AWAY_STAT_ADJUSTMENT
    assert np.isclose(r["a_net_rtg_away"], expected_a_net_rtg_away)
    assert not r["a_net_rtg_away_imputed"]

    h_tov_home = MOCK_TEAMS_DATA["teamalpha"]["tov_pct"] + HOME_STAT_ADJUSTMENT
    a_tov_away = MOCK_TEAMS_DATA["teambeta"]["tov_pct"] + AWAY_STAT_ADJUSTMENT
    expected_tov_diff = h_tov_home - a_tov_away
    assert np.isclose(r["hist_tov_pct_split_diff"], expected_tov_diff)