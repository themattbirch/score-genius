# Supabase RPC Specification for ScoreGenius

This document outlines the complete set of Remote Procedure Calls (RPCs) created for the ScoreGenius backend. These functions provide a secure and controlled API layer for accessing sports data.

## ✅ NBA RPCs

### `rpc_get_nba_seasonal_splits`

- **Purpose**: Fetches seasonal advanced stats for a **single** NBA team for a specific season.
- **Use Case**: Frontend display when viewing a single team's season-long performance.
- **Data Source**: `public.nba_team_seasonal_advanced_splits`

**Parameters**

| Name          | Type      | Description                                   |
| ------------- | --------- | --------------------------------------------- |
| `p_team_norm` | `text`    | The normalized team identifier (e.g., 'ATL'). |
| `p_season`    | `integer` | The starting year of the season (e.g., 2023). |

**Returns**

A table row containing detailed home/away advanced stats, including `pace_home`, `off_rtg_home`, `def_rtg_away`, etc.

**Example Call (Python)**

```python
# Fetches seasonal splits for the Atlanta Hawks for the 2023-24 season
splits = supabase.rpc(
    'rpc_get_nba_seasonal_splits',
    {'p_team_norm': 'ATL', 'p_season': 2023}
).execute()
```

### `rpc_get_nba_all_seasonal_splits`

- **Purpose**: Fetches seasonal advanced stats for all NBA teams for a specific season.
- **Use Case**: Backend batch processing (e.g., prediction.py) where stats for all teams are needed at once.
- **Data Source**: `public.nba_team_seasonal_advanced_splits`

**Parameters**

| Name       | Type      | Description                                   |
| ---------- | --------- | --------------------------------------------- |
| `p_season` | `integer` | The starting year of the season (e.g., 2023). |

**Returns**

A set of table rows, one for each team, containing detailed home/away advanced stats.

**Example Call (Python)**

```python
# Fetches all team seasonal splits for the 2023-24 season
all_splits = supabase.rpc(
    'rpc_get_nba_all_seasonal_splits',
    {'p_season': 2023}
).execute()
```

### `rpc_get_nba_rolling_20_features`

- **Purpose**: Fetches the most recent 20-game rolling features for a single NBA team on or before a specific date.
- **Use Case**: Frontend display for a team's recent form or for a specific game preview.
- **Data Source**: `public.nba_team_rolling_20_features`

**Parameters**

| Name          | Type   | Description                                      |
| ------------- | ------ | ------------------------------------------------ |
| `p_team_norm` | `text` | The normalized team identifier (e.g., 'BOS').    |
| `p_game_date` | `date` | The date to look back from (e.g., '2024-01-15'). |

**Returns**

A single table row with the most recent rolling feature set for that team.

**Example Call (Python)**

```python
# Fetches latest rolling features for the Celtics as of Jan 15, 2024
features = supabase.rpc(
    'rpc_get_nba_rolling_20_features',
    {'p_team_norm': 'BOS', 'p_game_date': '2024-01-15'}
).execute()
```

### `rpc_get_nba_rolling_features_for_games`

- **Purpose**: Fetches the most recent 20-game rolling features for multiple teams involved in upcoming games.
- **Use Case**: Backend batch processing (prediction.py) to efficiently gather features for all scheduled games.
- **Data Source**: `public.nba_team_rolling_20_features`

**Parameters**

| Name     | Type                   | Description                                                                  |
| -------- | ---------------------- | ---------------------------------------------------------------------------- |
| `p_keys` | `game_team_date_key[]` | An array of composite keys, each containing game_id, team_id, and game_date. |

**Returns**

A set of table rows containing the latest rolling features for each team in the input array.

**Example Call (Python)**

```python
# p_keys format: ["(game_id_1,team_id_1,game_date_1)", "(game_id_2,team_id_2,game_date_2)"]
keys = ["(123,ATL,2024-02-10)", "(124,BOS,2024-02-11)"]
bulk_features = supabase.rpc(
    'rpc_get_nba_rolling_features_for_games',
    {'p_keys': keys}
).execute()
```

### `rpc_get_nba_snapshot`

- **Purpose**: Fetches the complete pre-generated snapshot for a single NBA game.
- **Use Case**: Frontend display for a specific game preview page.
- **Data Source**: `public.nba_snapshots`

**Parameters**

| Name        | Type   | Description                         |
| ----------- | ------ | ----------------------------------- |
| `p_game_id` | `text` | The unique identifier for the game. |

**Returns**

A single row containing all snapshot data (headline stats, chart data, etc.) as JSONB objects.

**Example Call (Python)**

```python
# Fetches the snapshot for game with ID '12345'
snapshot = supabase.rpc('rpc_get_nba_snapshot', {'p_game_id': '12345'}).execute()
```

## ✅ MLB RPCs

### `rpc_get_mlb_seasonal_splits`

- **Purpose**: Fetches overall seasonal stats for a single MLB team.
- **Use Case**: Frontend display when viewing a single team's season performance.
- **Data Source**: `public.mlb_team_seasonal_advanced_splits`

**Parameters**

| Name        | Type      | Description                          |
| ----------- | --------- | ------------------------------------ |
| `p_team_id` | `text`    | The unique team identifier.          |
| `p_season`  | `integer` | The year of the season (e.g., 2023). |

**Returns**

A single row with overall stats like gp, win_pct, pythag_win_pct, run_differential, etc.

**Example Call (Python)**

```python
stats = supabase.rpc(
    'rpc_get_mlb_seasonal_splits',
    {'p_team_id': '147', 'p_season': 2023}
).execute()
```

### `rpc_get_mlb_rolling_10_features`

- **Purpose**: Fetches the most recent 10-game rolling features for a single MLB team on or before a specific date.
- **Use Case**: Frontend display for a team's recent form or for a specific game preview.
- **Data Source**: `public.mlb_team_rolling_10_features`

**Parameters**

| Name          | Type      | Description                                      |
| ------------- | --------- | ------------------------------------------------ |
| `p_team_id`   | `integer` | The unique team identifier.                      |
| `p_game_date` | `date`    | The date to look back from (e.g., '2024-08-10'). |

**Returns**

A single table row with the most recent rolling feature set for that team.

**Example Call (Python)**

```python
features = supabase.rpc(
    'rpc_get_mlb_rolling_10_features',
    {'p_team_id': 147, 'p_game_date': '2024-08-10'}
).execute()
```

### `rpc_get_mlb_rolling_features_for_games`

- **Purpose**: Fetches the most recent 10-game rolling features for multiple teams involved in upcoming games.
- **Use Case**: Backend batch processing (prediction.py) to efficiently gather features for all scheduled games.
- **Data Source**: `public.mlb_team_rolling_10_features`

**Parameters**

| Name     | Type                       | Description                                                                  |
| -------- | -------------------------- | ---------------------------------------------------------------------------- |
| `p_keys` | `mlb_game_team_date_key[]` | An array of composite keys, each containing game_id, team_id, and game_date. |

**Returns**

A set of table rows containing the latest rolling features for each team in the input array.

**Example Call (Python)**

```python
# p_keys format: ["(game_id_1,team_id_1,game_date_1)", "(game_id_2,team_id_2,game_date_2)"]
keys = ["(abc,147,2024-08-10)", "(def,121,2024-08-11)"]
bulk_features = supabase.rpc(
    'rpc_get_mlb_rolling_features_for_games',
    {'p_keys': keys}
).execute()
```

## ✅ NFL RPCs

### `rpc_get_nfl_season_stats`

- **Purpose**: Fetches aggregated season stats for a single NFL team.
- **Use Case**: Frontend display for a single team's season overview.
- **Data Source**: `public.nfl_season_stats`

**Parameters**

| Name        | Type      | Description                          |
| ----------- | --------- | ------------------------------------ |
| `p_team_id` | `integer` | The unique team identifier.          |
| `p_season`  | `integer` | The year of the season (e.g., 2023). |

**Returns**

A single row with stats like wins_all_percentage, points_for_avg_all, etc.

**Example Call (Python)**

```python
stats = supabase.rpc(
    'rpc_get_nfl_season_stats',
    {'p_team_id': 1, 'p_season': 2023}
).execute()
```

### `rpc_get_nfl_all_season_stats`

- **Purpose**: Fetches aggregated season stats for all NFL teams for a specific season.
- **Use Case**: Backend processing where context for all teams in a season is needed.
- **Data Source**: `public.nfl_season_stats`

**Parameters**

| Name       | Type      | Description                          |
| ---------- | --------- | ------------------------------------ |
| `p_season` | `integer` | The year of the season (e.g., 2023). |

**Returns**

A set of rows containing season stats for every team in that season.

**Example Call (Python)**

```python
all_stats = supabase.rpc(
    'rpc_get_nfl_all_season_stats',
    {'p_season': 2023}
).execute()
```

### `rpc_get_nfl_recent_form`

- **Purpose**: Fetches the most recent "form" data for a single NFL team.
- **Use Case**: Frontend display for a team's current momentum.
- **Data Source**: `public.nfl_recent_form`

**Parameters**

| Name        | Type      | Description                 |
| ----------- | --------- | --------------------------- |
| `p_team_id` | `integer` | The unique team identifier. |

**Returns**

A single row containing the most recent rolling averages for that team.

**Example Call (Python)**

```python
form = supabase.rpc('rpc_get_nfl_recent_form', {'p_team_id': 1}).execute()
```

### `rpc_get_nfl_all_recent_form`

- **Purpose**: Fetches the most recent "form" data for all NFL teams.
- **Use Case**: Backend batch processing (engine.py) to provide recent form context for all teams.
- **Data Source**: `public.nfl_recent_form`

**Parameters**

None.

**Returns**

A set of rows containing the latest recent form data for every team.

**Example Call (Python)**

```python
all_form_data = supabase.rpc('rpc_get_nfl_all_recent_form', {}).execute()
```

### `rpc_get_nfl_advanced_stats`

- **Purpose**: Fetches advanced seasonal stats for a single NFL team.
- **Use Case**: Frontend display for a deep-dive on a single team.
- **Data Source**: `public.nfl_advanced_team_stats`

**Parameters**

| Name        | Type      | Description                          |
| ----------- | --------- | ------------------------------------ |
| `p_team_id` | `integer` | The unique team identifier.          |
| `p_season`  | `integer` | The year of the season (e.g., 2023). |

**Returns**

A single row with stats like avg_third_down_pct, pythagorean_win_pct, etc.

**Example Call (Python)**

```python
adv_stats = supabase.rpc(
    'rpc_get_nfl_advanced_stats',
    {'p_team_id': 1, 'p_season': 2023}
).execute()
```

### `rpc_get_nfl_all_advanced_stats`

- **Purpose**: Fetches advanced seasonal stats for all NFL teams for a specific season.
- **Use Case**: Backend processing where advanced context for all teams is needed.
- **Data Source**: `public.nfl_advanced_team_stats`

**Parameters**

| Name       | Type      | Description                          |
| ---------- | --------- | ------------------------------------ |
| `p_season` | `integer` | The year of the season (e.g., 2023). |

**Returns**

A set of rows containing advanced stats for every team in that season.

**Example Call (Python)**

```python
all_adv_stats = supabase.rpc(
    'rpc_get_nfl_all_advanced_stats',
    {'p_season': 2023}
).execute()
```

### `rpc_get_nfl_snapshot`

- **Purpose**: Fetches the complete pre-generated snapshot for a single NFL game.
- **Use Case**: Frontend display for a specific game preview page.
- **Data Source**: `public.nfl_snapshots`

**Parameters**

| Name        | Type   | Description                         |
| ----------- | ------ | ----------------------------------- |
| `p_game_id` | `text` | The unique identifier for the game. |

**Returns**

A single row containing all snapshot data (headline stats, chart data, etc.) as JSONB objects.

**Example Call (Python)**

```python
snapshot = supabase.rpc('rpc_get_nfl_snapshot', {'p_game_id': '2023_01_ATL_CAR'}).execute()
```
