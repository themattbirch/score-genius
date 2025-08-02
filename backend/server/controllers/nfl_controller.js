// backend/server/controllers/nfl_controller.js

import {
  fetchNflTeamSeasonFull,
  fetchNflTeamSeasonRegOnly,
  fetchNflScheduleData,
  fetchNflSnapshotData,
  fetchNflSnapshotsByIds,
  buildCacheHeader,
  fetchNflDashboardCards,
  fetchNflSos,
  fetchNflSrs,
  checkCronHealth,
  validateTeamAgg,
  fetchNflAdvancedStats,
  fetchNflSeasonStats,
} from "../services/nfl_service.js";
import {
  NFL_ALLOWED_CONFERENCES,
  NFL_ALLOWED_DIVISIONS,
  parseSeasonParam,
  parseCsvInts,
  normConf,
  normDiv,
  badParam,
} from "../utils/nfl_validation.js";
// ---------------------------------------------------------------------------
// GET /api/v1/nfl/teams/:season/full
// ---------------------------------------------------------------------------
export async function getNflTeamSeasonFull(req, res, next) {
  try {
    const season = parseSeasonParam(req.params.season);
    if (season == null) return badParam(res, "Invalid season; use YYYY.");

    const teamIds = parseCsvInts(req.query.teamId);
    const conf = normConf(req.query.conference);
    if (req.query.conference && !conf)
      return badParam(res, "Invalid conference. Use AFC or NFC.");
    const div = normDiv(req.query.division);
    if (req.query.division && !div)
      return badParam(
        res,
        "Invalid division. Use East, West, North, or South."
      );

    const includeRaw =
      req.query.includeRaw === "1" || req.query.includeRaw === "true";

    const data = await fetchNflTeamSeasonFull({
      season,
      teamIds,
      conference: conf,
      division: div,
      includeRaw,
    });

    res.set(buildCacheHeader());
    return res.status(200).json(data);
  } catch (err) {
    next(err);
  }
}

// ---------------------------------------------------------------------------
// GET /api/v1/nfl/teams/:season/regonly
// ---------------------------------------------------------------------------
export async function getNflTeamSeasonRegOnly(req, res, next) {
  try {
    const season = parseSeasonParam(req.params.season);
    if (season == null) return badParam(res, "Invalid season; use YYYY.");

    const teamIds = parseCsvInts(req.query.teamId);
    const conf = normConf(req.query.conference);
    if (req.query.conference && !conf)
      return badParam(res, "Invalid conference. Use AFC or NFC.");
    const div = normDiv(req.query.division);
    if (req.query.division && !div)
      return badParam(
        res,
        "Invalid division. Use East, West, North, or South."
      );

    const includeRaw =
      req.query.includeRaw === "1" || req.query.includeRaw === "true";

    const data = await fetchNflTeamSeasonRegOnly({
      season,
      teamIds,
      conference: conf,
      division: div,
      includeRaw,
    });

    res.set(buildCacheHeader());
    return res.status(200).json(data);
  } catch (err) {
    next(err);
  }
}
// GET /api/v1/nfl/schedule?date=YYYY-MM-DD
export async function getNflSchedule(req, res, next) {
  console.log("✅ --- Reached the REAL NFL Controller! --- ✅"); // <-- ADD THIS LINE

  try {
    const { date } = req.query;
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      return res
        .status(400)
        .json({ message: "Invalid or missing date (YYYY-MM-DD)." });
    }

    const rows = await fetchNflScheduleData(date);

    // Ensure predicted fields (and timestamp) are always present
    const data = rows.map((g) => ({
      ...g,
      predictedHomeScore: g.predictedHomeScore ?? null,
      predictedAwayScore: g.predictedAwayScore ?? null,
      predictionUtc: g.predictionUtc ?? g.prediction_utc ?? null, // handle snake/camel just in case
    }));

    res.set(buildCacheHeader());
    return res.status(200).json({
      message: `NFL schedule for ${date}`,
      retrieved: data.length,
      data,
    });
  } catch (err) {
    next(err);
  }
}

// GET /api/v1/nfl/snapshots/:gameId
export async function getNflSnapshot(req, res, next) {
  const { gameId } = req.params;
  try {
    const snapshot = await fetchNflSnapshotData(gameId);
    return res.json(snapshot);
  } catch (err) {
    if (err.status === 404) {
      return res.status(404).json({ message: err.message });
    }
    next(err);
  }
}

// GET /api/v1/nfl/snapshots?gameIds=1,2,3
export async function getNflSnapshots(req, res, next) {
  try {
    const ids = (req.query.gameIds || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!ids.length) {
      return res.status(400).json({ message: "No gameIds provided." });
    }
    const snapshots = await fetchNflSnapshotsByIds(ids);
    return res.json(snapshots);
  } catch (err) {
    next(err);
  }
}
// GET /api/v1/nfl/teams/:season/dashboard
export async function getNflDashboard(req, res, next) {
  try {
    const season = parseSeasonParam(req.params.season);
    if (season == null) return badParam(res, "Invalid season; use YYYY.");

    const teamIds = parseCsvInts(req.query.teamId);
    const conf = normConf(req.query.conference);
    if (req.query.conference && !conf)
      return badParam(res, "Invalid conference.");
    const div = normDiv(req.query.division);
    if (req.query.division && !div) return badParam(res, "Invalid division.");

    const data = await fetchNflDashboardCards({
      season,
      teamIds,
      conference: conf,
      division: div,
    });

    res.set(buildCacheHeader());
    return res.status(200).json({ season, retrieved: data.length, data });
  } catch (err) {
    next(err);
  }
}
// GET /api/v1/nfl/teams/:season/sos
export async function getNflSos(req, res, next) {
  try {
    const season = parseSeasonParam(req.params.season);
    if (season == null) return badParam(res, "Invalid season; use YYYY.");

    const teamIds = parseCsvInts(req.query.teamId);
    const conf = normConf(req.query.conference);
    if (req.query.conference && !conf)
      return badParam(res, "Invalid conference.");
    const div = normDiv(req.query.division);
    if (req.query.division && !div) return badParam(res, "Invalid division.");

    const data = await fetchNflSos({
      season,
      teamIds,
      conference: conf,
      division: div,
    });
    res.set(buildCacheHeader());
    return res.status(200).json({ season, retrieved: data.length, data });
  } catch (err) {
    next(err);
  }
}

// GET /api/v1/nfl/teams/:season/srs
export async function getNflSrs(req, res, next) {
  try {
    const season = parseSeasonParam(req.params.season);
    if (season == null) return badParam(res, "Invalid season; use YYYY.");

    const teamIds = parseCsvInts(req.query.teamId);
    const conf = normConf(req.query.conference);
    if (req.query.conference && !conf)
      return badParam(res, "Invalid conference.");
    const div = normDiv(req.query.division);
    if (req.query.division && !div) return badParam(res, "Invalid division.");

    const data = await fetchNflSrs({
      season,
      teamIds,
      conference: conf,
      division: div,
    });
    res.set(buildCacheHeader());
    return res.status(200).json({ season, retrieved: data.length, data });
  } catch (err) {
    next(err);
  }
}
// GET /api/v1/nfl/health/cron
export async function getNflCronHealth(req, res, next) {
  try {
    const result = await checkCronHealth();
    res.set(buildCacheHeader());
    return res.status(200).json(result);
  } catch (err) {
    next(err);
  }
}

// GET /api/v1/nfl/health/validate
export async function getNflValidation(req, res, next) {
  try {
    const result = await validateTeamAgg();
    res.set(buildCacheHeader());
    return res.status(200).json(result);
  } catch (err) {
    next(err);
  }
}
// GET /api/v1/nfl/games/:id
export async function getNflGameById(req, res, next) {
  try {
    const gameId = Number.parseInt(req.params.id, 10);
    if (Number.isNaN(gameId)) return badParam(res, "Invalid game id.");

    const game = await fetchNflGameById(gameId); // implement in nfl_service.js

    if (!game) return res.status(404).json({ message: "Game not found." });

    const payload = {
      ...game,
      predictedHomeScore: game.predictedHomeScore ?? null,
      predictedAwayScore: game.predictedAwayScore ?? null,
      predictionUtc: game.predictionUtc ?? game.prediction_utc ?? null,
    };

    res.set(buildCacheHeader());
    return res.status(200).json(payload);
  } catch (err) {
    next(err);
  }
}

// GET /api/v1/nfl/team-stats/advanced?season=YYYY
export async function getNflAdvancedStats(req, res, next) {
  try {
    // 1. Validate the season from the query parameter
    const season = parseSeasonParam(req.query.season);
    if (season == null) {
      return badParam(res, "Invalid or missing season parameter; use YYYY.");
    }

    // 2. Call the service layer to fetch the data
    const data = await fetchNflAdvancedStats({ season });

    // 3. Set cache headers and send the successful response
    res.set(buildCacheHeader());
    return res.status(200).json({
      season,
      retrieved: data.length,
      data,
    });
  } catch (err) {
    // 4. Pass any errors to the error-handling middleware
    next(err);
  }
}
// GET /api/v1/nfl/team-stats/summary?season=YYYY
// GET /api/v1/nfl/team-stats/summary?season=YYYY
export async function getNflTeamStatsSummary(req, res, next) {
  try {
    const season = parseSeasonParam(req.query.season);
    if (season == null)
      return badParam(res, "Invalid or missing season parameter; use YYYY.");

    // Parallel fetches of the components
    const [advancedRaw, srsRaw, sosRaw, seasonStats] = await Promise.all([
      fetchNflAdvancedStats({ season }),
      fetchNflSrs({ season }),
      fetchNflSos({ season }),
      fetchNflSeasonStats({
        season,
        teamIds: [],
        conference: null,
        division: null,
      }),
    ]);

    // Logging for visibility (optional)
    console.log("Summary source lengths:", {
      advanced: advancedRaw.length,
      srs: srsRaw.length,
      sos: sosRaw.length,
      season: seasonStats.length,
    });

    const indexed = {};

    const upsert = (row, source) => {
      const teamId = row.team_id ?? row.teamId;
      if (!teamId) return;
      if (!indexed[teamId]) {
        indexed[teamId] = {
          team_id: teamId,
          team_name: row.team_name ?? row.teamName,
        };
      }
      if (source === "advanced") {
        Object.assign(indexed[teamId], row);
      } else if (source === "srs") {
        indexed[teamId].srs = row.srs_lite ?? row.srs;
      } else if (source === "sos") {
        indexed[teamId].sos = row.sos;
        if (row.sos_rank !== undefined) {
          indexed[teamId].sosRank = row.sos_rank;
        }
      }
    };

    advancedRaw.forEach((r) => upsert(r, "advanced"));
    srsRaw.forEach((r) => upsert(r, "srs"));
    sosRaw.forEach((r) => upsert(r, "sos"));

    // Augment with winPct from season stats
    Object.entries(indexed).forEach(([teamId, existing]) => {
      const seasonRow = seasonStats.find(
        (r) => String(r.team_id) === String(teamId)
      );
      if (seasonRow) {
        existing.winPct = seasonRow.win_pct ?? seasonRow.wins_all_percentage;
      }
    });

    const data = Object.values(indexed);

    res.set(buildCacheHeader());
    return res.status(200).json({
      season,
      retrieved: data.length,
      data,
    });
  } catch (err) {
    next(err);
  }
}
