# Repository Overview & Next Steps

This document outlines the current security setup, documentation status, future development plan, and next actions for the NFL stats repository.

---

## 🔐 Security Configuration

This project follows a read-only access pattern for consumers via a dedicated database role.

- **Role**: `nfl_stats_ro` (NOLOGIN)
- **Grants**:
  - `CONNECT` on the database
  - `USAGE` on the `public` schema
  - `SELECT` on the view `v_nfl_team_season_full`
- **Membership**: `GRANT nfl_stats_ro TO anon, authenticated;`
- **Restrictions**: All permissions on raw NFL data tables have been revoked from `anon` and `authenticated`.

> **Note on Row-Level Security (RLS)**
> The raw data tables are functionally private. If you ever re-expose them to service roles, be sure to implement RLS policies for security.

---

## 📊 Metric Dictionary

The metric dictionary is currently a work in progress.

- **Location**: `docs/nfl_team_agg_metrics_dict.md`
- **Recommendation**: Embed a markdown table in the file with the following columns: `col_name`, `type`, `agg_formula`, `notes`, and `source_col(s)`.

---

## 🚀 Future Enhancements (Phase 3)

The following table outlines planned features and where they will be implemented in the repository structure.

| Enhancement                   | Target Directory      | Notes                                                     |
| :---------------------------- | :-------------------- | :-------------------------------------------------------- |
| Strength of Schedule (SRS)    | `analytics/features/` | Requires a new materialized view.                         |
| Rolling Splits (Last X Games) | `db/views/`           | Best implemented as a parameterizable SQL function.       |
| Predictive Power Ratings      | `analytics/models/`   | Model outputs will be stored in `nfl_team_power_ratings`. |
| Advanced Stats API            | `apps/api/routes/`    | New endpoints will read from the new views.               |

---

## ✅ Next Actions

1.  **Commit this file**:
    ```bash
    git add REPO_STRUCTURE.md && git commit -m 'docs: add repo structure'
    ```
2.  **Generate Metric Dictionary**: Create and commit the `docs/nfl_team_agg_metrics_dict.md` file.
3.  **Tag Release**: After the changelog is updated, tag release `v1.3.0`.

---

## ❓ Key Questions

1.  Should I proceed with generating the **Metric Dictionary** (`nfl_team_agg_metrics_dict.md`), pre-filled with all columns and their corresponding formulas?
2.  Do you want a minimal **GitHub Actions workflow** that runs a SQL validation script nightly and fails the build on any data mismatch?
3.  Which Phase 3 feature should be prioritized: **Strength of Schedule (SRS)**, **rolling splits**, or **predictive power ratings**?
