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
  try {
    const { date } = req.query;
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      return res
        .status(400)
        .json({ message: "Invalid or missing date (YYYY-MM-DD)." });
    }

    const data = await fetchNflScheduleData(date);
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
