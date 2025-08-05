// frontend/src/screens/stats_screen.tsx

/* -------------------------------------------------------------------------- */
/*  Stats Screen – polished UI/UX, responsive tables, sticky toolbar          */
/* -------------------------------------------------------------------------- */

import React, { useMemo, useState, useEffect } from "react";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";
import { useNetworkStatus } from "@/hooks/use_network_status";
import { useOnline } from "@/contexts/online_context";
import OfflineBanner from "@/components/offline_banner";

import { useTeamStats } from "@/api/use_team_stats";
import { usePlayerStats } from "@/api/use_player_stats";
import { useAdvancedStats as useNbaAdvancedStats } from "@/api/use_nba_advanced_stats";
import { useMlbAdvancedStats } from "@/api/use_mlb_advanced_stats";
import { useNflAdvancedStats } from "@/api/use_nfl_advanced_stats";
import { useNflTeamSummary } from "@/api/use_nfl_team_summary";

import {
  UnifiedPlayerStats,
  UnifiedTeamStats,
  NbaAdvancedTeamStats,
  MlbAdvancedTeamStats,
  NflAdvancedTeamStats,
} from "@/types";

import SkeletonBox from "@/components/ui/skeleton_box";
import { ChevronsUpDown, Calendar as CalendarIcon } from "lucide-react";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import type { Sport } from "@/types";

/* -------------------------------------------------------------------------- */
/*  Helper sets for number formatting                                         */
/* -------------------------------------------------------------------------- */

const pctKeys = new Set([
  "three_pct",
  "ft_pct",
  "wins_all_percentage",
  "winPct",
  "efg_pct",
  "tov_pct",
  "oreb_pct",
  "pythagoreanWinPct",
  "avgRedZonePct",
  "avgThirdDownPct",
  "pythagorean_win_pct",
  "avg_red_zone_pct",
  "avg_third_down_pct",
]);

const oneDecimalKeys = new Set([
  "points_for_avg_all",
  "points_against_avg_all",
  "runs_for_avg_all",
  "runs_against_avg_all",
  "pace",
  "off_rtg",
  "def_rtg",
  "run_differential_avg",
  "home_away_run_diff_avg_split",
  "avgYardsPerDrive",
  "avgTurnoversPerGame",
  "avg_yards_per_drive",
  "avg_turnovers_per_game",
  "srs",
  "sos",
]);

const zeroDecimalKeys = new Set([
  "games_played",
  "points",
  "rebounds",
  "assists",
  "minutes",
  "run_differential",
  "expected_wins",
  "luck_factor",
  "wins",
  "runs_for",
  "runs_against",
  "gp",
]);

type SortDir = "asc" | "desc";
type TeamKey = keyof UnifiedTeamStats;
type PlayerKey = keyof UnifiedPlayerStats;
type NbaAdvKey = keyof NbaAdvancedTeamStats;
type MlbAdvKey = keyof MlbAdvancedTeamStats;

/* -------------------------------------------------------------------------- */
/*  Header cell (sortable)                                                    */
/* -------------------------------------------------------------------------- */
interface HeaderProps {
  label: string;
  active: boolean;
  dir?: SortDir;
  onClick: () => void;
  align?: "left" | "right";
  className?: string;
  [k: string]: any;
}
const HeaderCell: React.FC<HeaderProps> = ({
  label,
  active,
  dir,
  onClick,
  align = "right",
  className = "",
  ...rest
}) => (
  <th
    {...rest}
    scope="col"
    onClick={onClick}
    title="Click to sort"
    className={`select-none cursor-pointer py-2 px-3 text-sm font-medium group
                bg-gray-50 dark:bg-[var(--color-panel)]
                text-slate-600 dark:text-text-secondary
                hover:bg-gray-200 dark:hover:bg-gray-700/30
                ${align === "left" ? "text-left" : "text-center"}
                ${className}`}
    aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
  >
    <span className="inline-flex items-center">
      {label}
      <ChevronsUpDown
        size={12}
        className={`ml-1 transition-transform ${
          active
            ? dir === "asc"
              ? "opacity-80 -translate-y-0.5"
              : "opacity-80 translate-y-0.5 rotate-180"
            : "opacity-25 group-hover:opacity-60"
        }`}
      />
    </span>
  </th>
);

/* -------------------------------------------------------------------------- */
/*  Utility: format cell values                                               */
/* -------------------------------------------------------------------------- */
const formatCell = (val: any, key: string): string => {
  if (val == null || val === -Infinity) return "–";
  if (pctKeys.has(key))
    return typeof val === "number" ? `${(val * 100).toFixed(1)}%` : String(val);
  if (oneDecimalKeys.has(key))
    return typeof val === "number" ? val.toFixed(1) : String(val);
  if (zeroDecimalKeys.has(key))
    return typeof val === "number" ? val.toFixed(0) : String(val);
  return typeof val === "number" ? val.toFixed(1) : String(val);
};

/* -------------------------------------------------------------------------- */
/*  Main component                                                            */
/* -------------------------------------------------------------------------- */
const StatsScreen: React.FC = () => {
  /* ───── context & basic state ────────────────────────────────────────── */
  const { sport } = useSport();
  const { date } = useDate();
  const online = useNetworkStatus();
  if (!online) {
    return <OfflineBanner message="You’re offline — stats unavailable" />;
  }

  /* ───── season logic ────────────────────────────────────────────────── */
  const defaultSeason = useMemo(() => {
    if (sport === "MLB") return date.getUTCFullYear();
    const m = date.getUTCMonth() + 1;
    const d = date.getUTCDate();
    return m > 10 || (m === 10 && d >= 22)
      ? date.getUTCFullYear()
      : date.getUTCFullYear() - 1;
  }, [sport, date]);

  const [season, setSeason] = useState<number>(defaultSeason);
  useEffect(() => setSeason(defaultSeason), [defaultSeason]);

  /* ───── sub‑tabs ────────────────────────────────────────────────────── */
  const [tab, setTab] = useState<"teams" | "players" | "advanced">("teams");
  useEffect(() => setTab("teams"), [sport]);

  /* ───── search (NBA players) ────────────────────────────────────────── */
  const [playerSearch, setPlayerSearch] = useState("");

  /* ───── fetch flags ─────────────────────────────────────────────────── */
  const canFetchSeason = useMemo(
    () => (sport === "MLB" ? true : season <= defaultSeason),
    [sport, season, defaultSeason]
  );

  /* ---------------- NFL – Teams tab ---------------- */
  const nflSummaryHeaders = [
    { label: "Team", key: "teamName" as const },
    { label: "Win %", key: "winPct" as const },
    { label: "Yds / Drive", key: "avgYardsPerDrive" as const },
    { label: "Red Zone %", key: "avgRedZonePct" as const },
    { label: "TOP", key: "avgTimeOfPossession" as const },
    { label: "TOs / Game", key: "avgTurnoversPerGame" as const },
  ];

  /* ---------------- NFL – Advanced tab ------------- */
  const nflAdvHeaders = [
    { label: "Team", key: "team_name" as const }, // keeps names aligned
    { label: "Pythag W %", key: "pythagorean_win_pct" as const },
    { label: "3rd %", key: "avg_third_down_pct" as const },
    { label: "SRS", key: "srs" as const },
    { label: "SoS", key: "sos" as const },
  ];

  // new sort state override for NFL
  useEffect(() => {
    setTeamSort({
      key:
        sport === "NFL"
          ? tab === "advanced"
            ? ("srs" as TeamKey)
            : ("winPct" as TeamKey)
          : ("wins_all_percentage" as TeamKey),
      dir: "desc",
    });
  }, [sport, tab]);

  const teamOpts = useMemo(
    () => ({
      sport,
      season,
      enabled: online && canFetchSeason && tab === "teams",
    }),
    [sport, season, online, canFetchSeason, tab]
  );

  const playerOpts = useMemo(
    () => ({
      sport: "NBA" as Sport,
      season,
      search: playerSearch,
      enabled: online && canFetchSeason && sport === "NBA" && tab === "players",
    }),
    [season, playerSearch, online, canFetchSeason, sport, tab]
  );

  const nbaAdvOpts = useMemo(
    () => ({
      sport: "NBA" as Sport,
      season,
      enabled:
        online && canFetchSeason && sport === "NBA" && tab === "advanced",
    }),
    [season, online, canFetchSeason, sport, tab]
  );

  const mlbAdvOpts = useMemo(
    () => ({
      sport: "MLB" as Sport,
      season,
      enabled:
        online && canFetchSeason && sport === "MLB" && tab === "advanced",
    }),
    [season, online, canFetchSeason, sport, tab]
  );

  /* ───── queries ─────────────────────────────────────────────────────── */
  // NFL summary / team stats (single source)
  const isNflTeamSummaryActive = sport === "NFL" && tab === "teams";
  const {
    data: teamSummary, // used for merging SRS/SoS into advanced
    isLoading: teamLoading,
    error: teamErr,
  } = useNflTeamSummary({
    season,
    sport: "NFL",
    enabled: online && canFetchSeason && sport === "NFL",
  });

  const {
    data: teamData,
    isLoading: teamStatsLoading,
    error: teamStatsErr,
  } = useTeamStats(teamOpts);

  const {
    data: nflAdv,
    isLoading: nflAdvLoading,
    error: nflAdvErr,
  } = useNflAdvancedStats({
    season,
    sport: "NFL",
    enabled: online && canFetchSeason && sport === "NFL" && tab === "advanced",
  });

  const {
    data: playerData,
    isLoading: playerLoading,
    error: playerErr,
  } = usePlayerStats(playerOpts);

  const {
    data: nbaAdv,
    isLoading: nbaAdvLoading,
    error: nbaAdvErr,
  } = useNbaAdvancedStats(nbaAdvOpts);

  const {
    data: mlbAdv,
    isLoading: mlbAdvLoading,
    error: mlbAdvErr,
  } = useMlbAdvancedStats(mlbAdvOpts);

  /* ------------------------------------------------------------------ */
  /*  Helpers                                                           */
  /* ------------------------------------------------------------------ */

  const canonical = (s?: string) => (s ?? "").replace(/\s+/g, "").toLowerCase();
  const toNum = (v: any) =>
    v === null || v === undefined || v === "" ? null : Number(v);

  /*  Type guard: does this value look like the wrapper shape? */
  interface AdvWrapper {
    advanced: any[];
    srs?: any[];
    sos?: any[];
  }
  const isWrapper = (val: unknown): val is AdvWrapper =>
    !!val && typeof val === "object" && "advanced" in val;

  /* ------------------------------------------------------------------ */
  /*  NFL ADVANCED NORMALISER (single source of truth)                  */
  /* ------------------------------------------------------------------ */

  type NflAdvRow = NflAdvancedTeamStats & Record<string, any>;

  const nflAdvData = useMemo<NflAdvRow[] | undefined>(() => {
    if (!nflAdv && !teamSummary) return undefined;

    /* ---- 1. unify row source ---------------------------------------- */
    let advRows: any[] = [];

    // coerce to unknown so the guard can discriminate without “never” bleed
    const maybeNflAdv = nflAdv as unknown;
    const maybeTeamSummary = teamSummary as unknown;

    if (Array.isArray(nflAdv)) {
      advRows = nflAdv;
    } else if (isWrapper(maybeNflAdv)) {
      advRows = (maybeNflAdv as AdvWrapper).advanced;
    } else if (isWrapper(maybeTeamSummary)) {
      advRows = (maybeTeamSummary as AdvWrapper).advanced;
    } else if (Array.isArray(teamSummary)) {
      advRows = teamSummary;
    }

    /* ---- 2. build SRS / SoS lookup tables --------------------------- */
    const srsIndex: Record<string, number | null> = {};
    const sosIndex: Record<string, number | null> = {};

    const ingestSrs = (rows: any[] | undefined) => {
      (rows ?? []).forEach((r) => {
        const key = canonical(r.team_name ?? r.teamName);
        srsIndex[key] = toNum(r.srs ?? r.srs_rating ?? r.srs_lite ?? r.srsLite);
      });
    };
    const ingestSos = (rows: any[] | undefined) => {
      (rows ?? []).forEach((r) => {
        const key = canonical(r.team_name ?? r.teamName);
        sosIndex[key] = toNum(r.sos ?? r.sos_rating ?? r.strength_of_schedule);
      });
    };

    if (isWrapper(teamSummary)) {
      ingestSrs(teamSummary.srs);
      ingestSos(teamSummary.sos);
    }
    if (isWrapper(nflAdv)) {
      ingestSrs(nflAdv.srs);
      ingestSos(nflAdv.sos);
    }

    if (Array.isArray(teamSummary)) {
      // also pull any embedded srs/sos from flat summary rows
      (teamSummary as any[]).forEach((r) => {
        const key = canonical(r.team_name ?? r.teamName);
        if (srsIndex[key] == null) {
          srsIndex[key] = toNum(
            r.srs ?? r.srs_rating ?? r.srs_lite ?? r.srsLite
          );
        }
        if (sosIndex[key] == null) {
          sosIndex[key] = toNum(
            r.sos ?? r.sos_rating ?? r.strength_of_schedule
          );
        }
      });
    }

    /* ---- 3. merge into canonical rows ------------------------------ */
    return advRows.map((row: any) => {
      const key = canonical(row.team_name ?? row.teamName);
      return {
        ...row,
        srs:
          toNum(row.srs ?? row.srs_rating ?? row.srs_lite ?? row.srsLite) ??
          srsIndex[key] ??
          null,
        sos:
          toNum(row.sos ?? row.sos_rating ?? row.strength_of_schedule) ??
          sosIndex[key] ??
          null,
      };
    });
  }, [nflAdv, teamSummary]);

  /* ───── sort state helpers ──────────────────────────────────────────── */
  const [teamSort, setTeamSort] = useState<{ key: TeamKey; dir: SortDir }>({
    key:
      sport === "NFL"
        ? ("winPct" as TeamKey)
        : ("wins_all_percentage" as TeamKey),
    dir: "desc",
  });
  useEffect(() => {
    setTeamSort({ key: "wins_all_percentage", dir: "desc" });
  }, [sport]);

  // NFL override
  useEffect(() => {
    if (sport === "NFL") {
      setTeamSort({ key: "srs" as any, dir: "desc" });
    }
  }, [sport]);

  const [playerSort, setPlayerSort] = useState<{
    key: PlayerKey;
    dir: SortDir;
  } | null>(null);

  const [nbaAdvSort, setNbaAdvSort] = useState<{
    key: NbaAdvKey;
    dir: SortDir;
  }>({ key: "off_rtg", dir: "desc" });

  const [mlbAdvSort, setMlbAdvSort] = useState<{
    key: MlbAdvKey;
    dir: SortDir;
  }>({ key: "run_differential", dir: "desc" });

  type NflAdvKey = keyof NflAdvancedTeamStats;
  const [nflAdvSort, setNflAdvSort] = useState<{
    key: NflAdvKey;
    dir: SortDir;
  }>({
    key: "srs",
    dir: "desc",
  });
  const toggleNflAdvSort = (k: NflAdvKey) =>
    setNflAdvSort((p) =>
      p.key === k
        ? { key: k, dir: p.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  /* ─── explicit toggle helpers (typed) ──────────────────────────────── */
  const toggleTeamSort = (k: TeamKey) =>
    setTeamSort((p) =>
      p.key === k
        ? { key: k, dir: p.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  const togglePlayerSort = (k: PlayerKey) =>
    setPlayerSort((p) =>
      p?.key === k
        ? { key: k, dir: p.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  const toggleNbaAdvSort = (k: NbaAdvKey) =>
    setNbaAdvSort((p) =>
      p.key === k
        ? { key: k, dir: p.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  const toggleMlbAdvSort = (k: MlbAdvKey) =>
    setMlbAdvSort((p) =>
      p.key === k
        ? { key: k, dir: p.dir === "asc" ? "desc" : "asc" }
        : { key: k, dir: "asc" }
    );

  /* ───── sorted data memo – ─────────────────────────── */
  const effectiveTeamData = teamData;

  /* pick the correct raw dataset before sorting */
  const rawTeamData =
    sport === "NFL" && tab === "teams" ? teamSummary : teamData;

  const sortedTeams = useMemo(() => {
    if (!rawTeamData) return [];
    const out = [...rawTeamData];
    const { key, dir } = teamSort;

    out.sort((a: any, b: any) => {
      const av = a[key] ?? null;
      const bv = b[key] ?? null;

      let diff;
      if (typeof av === "number" && typeof bv === "number") diff = av - bv;
      else if (av === null && bv === null) diff = 0;
      else if (av === null) diff = dir === "asc" ? -1 : 1;
      else if (bv === null) diff = dir === "asc" ? 1 : -1;
      else diff = String(av).localeCompare(String(bv));

      /* tie-breaker for stability */
      if (diff === 0) {
        const aName = String(a.teamName ?? a.team_name ?? "");
        const bName = String(b.teamName ?? b.team_name ?? "");
        diff = aName.localeCompare(bName);
      }
      return dir === "asc" ? diff : -diff;
    });

    return out;
  }, [rawTeamData, teamSort]);

  const sortedPlayers = useMemo(() => {
    if (!playerData) return [];
    if (!playerSort) return playerData;
    const { key, dir } = playerSort;
    return [...playerData].sort((a, b) => {
      const av = a[key] ?? -Infinity;
      const bv = b[key] ?? -Infinity;
      const diff =
        typeof av === "number" && typeof bv === "number"
          ? av - bv
          : String(av).localeCompare(String(bv));
      return dir === "asc" ? diff : -diff;
    });
  }, [playerData, playerSort]);

  const sortedNbaAdv = useMemo(() => {
    if (!nbaAdv) return [];
    const { key, dir } = nbaAdvSort;
    return [...nbaAdv].sort((a, b) => {
      const diff =
        (a[key] ?? -Infinity) > (b[key] ?? -Infinity)
          ? 1
          : (a[key] ?? -Infinity) < (b[key] ?? -Infinity)
          ? -1
          : 0;
      return dir === "asc" ? diff : -diff;
    });
  }, [nbaAdv, nbaAdvSort]);

  const sortedMlbAdv = useMemo(() => {
    if (!mlbAdv) return [];
    const { key, dir } = mlbAdvSort;
    return [...mlbAdv].sort((a, b) => {
      const diff =
        (a[key] ?? -Infinity) > (b[key] ?? -Infinity)
          ? 1
          : (a[key] ?? -Infinity) < (b[key] ?? -Infinity)
          ? -1
          : 0;
      return dir === "asc" ? diff : -diff;
    });
  }, [mlbAdv, mlbAdvSort]);

  console.log("outer", nflAdvData?.[0]);

  /* ===== SORTED NFL ADVANCED USING NORMALISED DATA ===== */
  const sortedNflAdv = useMemo(() => {
    if (!nflAdvData) return [];
    const { key, dir } = nflAdvSort;
    return [...nflAdvData].sort((a: any, b: any) => {
      const av = a[key] ?? -Infinity;
      const bv = b[key] ?? -Infinity;
      let diff;
      if (typeof av === "number" && typeof bv === "number") {
        diff = av - bv;
      } else {
        diff = String(av).localeCompare(String(bv));
      }
      if (diff === 0) {
        const aName = String(a.team_name);
        const bName = String(b.team_name);
        diff = aName.localeCompare(bName);
      }
      return dir === "asc" ? diff : -diff;
    });
  }, [nflAdvData, nflAdvSort]);

  /* -------------------------------------------------------------------- */
  /*  Header configs                                                      */
  /* -------------------------------------------------------------------- */
  const teamHeaders = useMemo(
    () =>
      sport === "MLB"
        ? [
            { label: "Team", key: "team_name" as TeamKey },
            { label: "Win %", key: "wins_all_percentage" as TeamKey },
            { label: "Runs For", key: "runs_for_avg_all" as TeamKey },
            { label: "Runs Vs", key: "runs_against_avg_all" as TeamKey },
            { label: "Streak", key: "current_form" as TeamKey },
          ]
        : [
            { label: "Team", key: "team_name" as TeamKey },
            { label: "Win %", key: "wins_all_percentage" as TeamKey },
            { label: "Off Pts", key: "points_for_avg_all" as TeamKey },
            { label: "Def Pts", key: "points_against_avg_all" as TeamKey },
            { label: "Streak", key: "current_form" as TeamKey },
          ],
    [sport]
  );

  const nbaAdvHeaders = [
    { label: "Team", key: "team_name" as NbaAdvKey },
    { label: "Pace", key: "pace" as NbaAdvKey },
    { label: "OffRtg", key: "off_rtg" as NbaAdvKey },
    { label: "DefRtg", key: "def_rtg" as NbaAdvKey },
    { label: "eFG%", key: "efg_pct" as NbaAdvKey },
    { label: "TOV%", key: "tov_pct" as NbaAdvKey },
    { label: "ORB%", key: "oreb_pct" as NbaAdvKey },
    { label: "GP", key: "games_played" as NbaAdvKey },
  ];

  const mlbAdvHeaders = [
    { label: "Team", key: "team_name" as MlbAdvKey },
    { label: "Win %", key: "win_pct" as MlbAdvKey },
    { label: "Pyth W%", key: "pythag_win_pct" as MlbAdvKey },
    { label: "Run Diff", key: "run_differential" as MlbAdvKey },
    { label: "Run Diff Avg", key: "run_differential_avg" as MlbAdvKey },
    { label: "Luck", key: "luck_factor" as MlbAdvKey },
    { label: "GP", key: "gp" as MlbAdvKey },
  ];

  const playerHeaders = [
    { label: "Player", key: "player_name" as PlayerKey },
    { label: "Team", key: "team_name" as PlayerKey },
    { label: "Pts", key: "points" as PlayerKey },
    { label: "Reb", key: "rebounds" as PlayerKey },
    { label: "Ast", key: "assists" as PlayerKey },
    { label: "3P%", key: "three_pct" as PlayerKey },
    { label: "FT%", key: "ft_pct" as PlayerKey },
    { label: "Min", key: "minutes" as PlayerKey },
    { label: "GP", key: "games_played" as PlayerKey },
  ];

  /* -------------------------------------------------------------------- */
  /*  Render helpers                                                      */
  /* -------------------------------------------------------------------- */

  const renderSkeletonList = (count: number) => (
    <div className="p-4 space-y-2">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonBox key={i} className="h-8 w-full rounded" />
      ))}
    </div>
  );

  const renderEmpty = (msg: string) => (
    <div className="p-4 text-center text-text-secondary">{msg}</div>
  );

  const tableWrapper =
    "overflow-x-auto rounded-xl shadow-[var(--shadow-card)] " +
    "border border-[var(--color-border-subtle)]";

  const tableBase =
    "min-w-full text-xs sm:text-sm " + "bg-white dark:bg-[var(--color-panel)]";

  const theadBase =
    "sticky top-0 z-10 bg-gray-50 dark:bg-[var(--color-panel)] " +
    "border-b border-[var(--color-border-subtle)]";

  const tbodyBase =
    "divide-y divide-[var(--color-border-subtle)] " +
    "bg-white dark:bg-[var(--color-panel)]";

  /* ===== TEAM TABLE ==================================================== */
  /* ===== TEAM TABLE ==================================================== */
  const TeamTable = () => {
    const usingNflSummary = sport === "NFL" && tab === "teams";

    /* pick correct dataset + status flags for Teams tab */
    const teamsLoading = usingNflSummary ? teamLoading : teamStatsLoading;
    const teamsError = usingNflSummary ? teamErr : teamStatsErr;

    const headers = usingNflSummary ? nflSummaryHeaders : teamHeaders;
    const dataToRender = sortedTeams;

    if (teamsLoading) return renderSkeletonList(8);
    if (teamsError) return renderEmpty("Failed to load team stats.");
    if (!dataToRender.length)
      return renderEmpty("No team stats for this season.");

    return (
      <div className={tableWrapper}>
        <table className={tableBase}>
          <caption className="sr-only">Team statistics</caption>
          <thead className={theadBase}>
            <tr>
              {headers.map(({ label, key }, idx) => (
                <HeaderCell
                  key={`${String(key)}-${idx}`}
                  data-tour={
                    key === "wins_all_percentage"
                      ? "stats-column-winpct"
                      : undefined
                  }
                  label={label}
                  active={teamSort.key === key}
                  dir={teamSort.dir}
                  onClick={() => toggleTeamSort(key as any)}
                  align={idx === 0 ? "left" : "right"}
                  className={
                    idx > 0
                      ? "border-l border-[var(--color-border-subtle)]"
                      : ""
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody className={tbodyBase}>
            {dataToRender.map((t) => (
              <tr
                key={String(t.team_id ?? t.teamId)}
                className="odd:bg-gray-50 dark:odd:bg-slate-800/40 hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {headers.map(({ key }, idx) => (
                  <td
                    key={`${String(key)}-${idx}`}
                    className={`py-1.5 sm:py-2 px-2 sm:px-3 whitespace-nowrap ${
                      idx === 0
                        ? "text-left font-medium"
                        : "text-center border-l border-[var(--color-border-subtle)]"
                    }`}
                  >
                    {formatCell(t[key as string], key as string)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  /* ===== PLAYER TABLE (NBA) =========================================== */
  const PlayerTable = () => {
    if (playerLoading) return renderSkeletonList(12);
    if (playerErr) return renderEmpty("Failed loading player stats.");
    if (!sortedPlayers.length)
      return renderEmpty(
        playerSearch ? `No results for “${playerSearch}”` : "No data."
      );

    return (
      <div className={tableWrapper}>
        <table className={tableBase}>
          <caption className="sr-only">NBA player statistics</caption>
          <thead className={theadBase}>
            <tr>
              {playerHeaders.map(({ label, key }, idx) => (
                <HeaderCell
                  key={key as string}
                  label={label}
                  active={playerSort?.key === key}
                  dir={playerSort?.dir}
                  onClick={() => togglePlayerSort(key)}
                  align={idx < 2 ? "left" : "right"}
                  className={
                    idx > 0
                      ? "border-l border-[var(--color-border-subtle)]"
                      : ""
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody className={tbodyBase}>
            {sortedPlayers.map((p, rowIdx) => (
              <tr
                key={`${p.player_id}-${rowIdx}`}
                className="odd:bg-gray-50 dark:odd:bg-slate-800/40 hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {playerHeaders.map(({ key }, idx) => (
                  <td
                    key={key as string}
                    className={`py-1.5 sm:py-2 px-2 sm:px-3 whitespace-nowrap ${
                      idx === 0
                        ? "text-left font-medium"
                        : idx === 1
                        ? "text-left border-l border-[var(--color-border-subtle)]"
                        : "text-center border-l border-[var(--color-border-subtle)]"
                    }`}
                  >
                    {formatCell(p[key], key as string)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  /* ===== NBA ADVANCED TABLE =========================================== */
  const NbaAdvTable = () => {
    if (nbaAdvLoading) return renderSkeletonList(8);
    if (nbaAdvErr) return renderEmpty("Problem loading advanced stats.");
    if (!sortedNbaAdv.length)
      return renderEmpty("No advanced stats for this season.");

    return (
      <div className={tableWrapper}>
        <table className={tableBase}>
          <caption className="sr-only">NBA advanced statistics</caption>
          <thead className={theadBase}>
            <tr>
              {nbaAdvHeaders.map(({ label, key }, idx) => (
                <HeaderCell
                  key={key as string}
                  label={label}
                  active={nbaAdvSort.key === key}
                  dir={nbaAdvSort.dir}
                  onClick={() => toggleNbaAdvSort(key)}
                  align={idx === 0 ? "left" : "right"}
                  className={
                    idx > 0
                      ? "border-l border-[var(--color-border-subtle)]"
                      : ""
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody className={tbodyBase}>
            {sortedNbaAdv.map((t) => (
              <tr
                key={t.team_id ?? t.team_name}
                className="odd:bg-gray-50 dark:odd:bg-slate-800/40 hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {nbaAdvHeaders.map(({ key }, idx) => (
                  <td
                    key={key as string}
                    className={`py-1.5 sm:py-2 px-2 sm:px-3 whitespace-nowrap ${
                      idx === 0
                        ? "text-left font-medium"
                        : "text-center border-l border-[var(--color-border-subtle)]"
                    }`}
                  >
                    {formatCell(t[key], key as string)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  /* ===== MLB ADVANCED TABLE =========================================== */
  const MlbAdvTable = () => {
    if (mlbAdvLoading) return renderSkeletonList(10);
    if (mlbAdvErr) return renderEmpty("Problem loading MLB advanced stats.");
    if (!sortedMlbAdv.length)
      return renderEmpty("No advanced stats for this season.");

    return (
      <div className={tableWrapper}>
        <table className={tableBase}>
          <caption className="sr-only">MLB advanced statistics</caption>
          <thead className={theadBase}>
            <tr>
              {mlbAdvHeaders.map(({ label, key }, idx) => (
                <HeaderCell
                  key={key as string}
                  label={label}
                  active={mlbAdvSort.key === key}
                  dir={mlbAdvSort.dir}
                  onClick={() => toggleMlbAdvSort(key)}
                  align={idx === 0 ? "left" : "right"}
                  className={
                    idx > 0
                      ? "border-l border-[var(--color-border-subtle)]"
                      : ""
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody className={tbodyBase}>
            {sortedMlbAdv.map((t) => (
              <tr
                key={t.team_id ?? t.team_name}
                className="odd:bg-gray-50 dark:odd:bg-slate-800/40 hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {mlbAdvHeaders.map(({ key }, idx) => (
                  <td
                    key={key as string}
                    className={`py-1.5 sm:py-2 px-2 sm:px-3 whitespace-nowrap ${
                      idx === 0
                        ? "text-left font-medium"
                        : "text-center border-l border-[var(--color-border-subtle)]"
                    }`}
                  >
                    {formatCell(t[key], key as string)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  /* ===== NFL ADVANCED TABLE =========================================== */

  const NflAdvTable = () => {
    if (nflAdvLoading) return renderSkeletonList(8);
    if (nflAdvErr) return renderEmpty("Problem loading NFL advanced stats.");
    if (!sortedNflAdv.length)
      return renderEmpty("No advanced stats for this season.");

    return (
      <div className={tableWrapper}>
        <table className={tableBase}>
          <caption className="sr-only">NFL advanced statistics</caption>
          <thead className={theadBase}>
            <tr>
              {nflAdvHeaders.map(({ label, key }, idx) => (
                <HeaderCell
                  key={`${String(key)}-${idx}`}
                  label={label}
                  active={nflAdvSort.key === key}
                  dir={nflAdvSort.dir}
                  onClick={() => toggleNflAdvSort(key as NflAdvKey)}
                  align={idx === 0 ? "left" : "right"}
                  className={
                    idx > 0
                      ? "border-l border-[var(--color-border-subtle)]"
                      : ""
                  }
                />
              ))}
            </tr>
          </thead>
          <tbody className={tbodyBase}>
            {sortedNflAdv.map((t) => (
              <tr
                key={String(t.team_name)}
                className="odd:bg-gray-50 dark:odd:bg-slate-800/40 hover:bg-gray-50 dark:hover:bg-gray-800/60"
              >
                {nflAdvHeaders.map(({ key }, idx) => (
                  <td
                    key={`${String(key)}-${idx}`}
                    className={`py-1.5 sm:py-2 px-2 sm:px-3 whitespace-nowrap ${
                      idx === 0
                        ? "text-left font-medium"
                        : "text-center border-l border-[var(--color-border-subtle)]"
                    }`}
                  >
                    {formatCell(t[key as string], key as string)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  /* -------------------------------------------------------------------- */
  /*  Season selector options                                             */
  /* -------------------------------------------------------------------- */
  const seasonOpts = useMemo(() => {
    return Array.from({ length: 5 }).map((_, i) => {
      const yr = defaultSeason - i;
      return {
        value: yr,
        label:
          sport === "NBA" ? `${yr}-${String(yr + 1).slice(-2)}` : String(yr),
        disabled: sport === "NFL" && yr > defaultSeason, // disable future for NFL
      };
    });
  }, [defaultSeason, sport]);

  /* -------------------------------------------------------------------- */
  /*  Render                                                              */
  /* -------------------------------------------------------------------- */
  const heading =
    tab === "teams"
      ? `${sport} Team Rankings`
      : tab === "players"
      ? "NBA Player Statistics"
      : `${sport} Advanced Statistics`;

  return (
    <section className="p-6 md:px-8 lg:px-12 space-y-6">
      {/* Sticky filters bar */}
      <div
        className="filters-bar
      -mx-6 md:-mx-8 lg:-mx-12
      px-6  md:px-8  lg:px-12
      "
      >
        {/* Sub‑tabs pill */}
        <div className="flex gap-1 rounded-lg bg-gray-200 dark:bg-[var(--color-surface-hover)] p-1 text-xs sm:text-sm">
          {(sport === "NBA"
            ? (["teams", "players", "advanced"] as const)
            : (["teams", "advanced"] as const)
          ).map((t) => {
            const tourAttr =
              t === "advanced" ? { "data-tour": "stats-subtab-advanced" } : {};
            return (
              <button
                key={t}
                aria-pressed={tab === t}
                onClick={() => setTab(t)}
                className={`rounded-md px-4 py-2 transition-colors ${
                  tab === t
                    ? "bg-green-600 text-white shadow-sm"
                    : "text-gray-700 dark:text-text-secondary hover:bg-gray-300 dark:hover:bg-gray-700/50"
                }`}
                {...tourAttr}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            );
          })}
        </div>

        {/* Season picker */}
        <Popover>
          <PopoverTrigger asChild>
            <button className="inline-flex items-center flex-nowrap whitespace-nowrap gap-1 rounded-lg border px-2 py-1 text-[10px] sm:text-xs border-[var(--color-border-subtle)] bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-gray-50 dark:hover:bg-slate-700">
              <CalendarIcon size={16} strokeWidth={1.8} />
              {sport === "NBA"
                ? `${season}-${String(season + 1).slice(-2)}`
                : season}
            </button>
          </PopoverTrigger>
          <PopoverContent
            side="bottom"
            align="end"
            sideOffset={8}
            className="bg-white dark:bg-[var(--color-panel)] rounded-lg shadow-lg p-2 w-36"
          >
            <div className="space-y-1">
              {seasonOpts.map(({ value, label, disabled }) => (
                <button
                  key={value}
                  onClick={() => !disabled && setSeason(value)}
                  disabled={disabled}
                  aria-disabled={disabled}
                  className={`w-full text-left px-3 py-2 rounded text-sm transition-colors
      ${
        value === season
          ? "font-semibold text-green-600 dark:text-green-500 bg-gray-100 dark:bg-slate-700"
          : disabled
          ? "text-gray-400 cursor-not-allowed"
          : "hover:bg-gray-100 dark:hover:bg-slate-700"
      }`}
                  title={
                    disabled ? "Season not yet available for NFL" : undefined
                  }
                >
                  {label}
                  {disabled && (
                    <span className="ml-1 text-xs opacity-60">(upcoming)</span>
                  )}
                </button>
              ))}
            </div>
          </PopoverContent>
        </Popover>
      </div>

      {/* Heading */}
      <h1 className="text-xl font-semibold">{heading}</h1>

      {/* Player search */}
      {sport === "NBA" && tab === "players" && (
        <input
          type="text"
          placeholder="Search player…"
          value={playerSearch}
          onChange={(e) => setPlayerSearch(e.target.value)}
          className="w-full sm:w-64 rounded-lg border border-[var(--color-border-subtle)] bg-white dark:bg-[var(--color-panel)] px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-green-600"
        />
      )}

      {/* Content */}
      {!online ? (
        <div className="p-4 border border-[var(--color-border-subtle)] rounded-lg bg-yellow-50 dark:bg-yellow-900/20 text-sm text-yellow-800 dark:text-yellow-300">
          Live stats require internet. Reconnect to view current data.
        </div>
      ) : sport === "NBA" ? (
        tab === "teams" ? (
          <TeamTable />
        ) : tab === "players" ? (
          <PlayerTable />
        ) : (
          <NbaAdvTable />
        )
      ) : sport === "NFL" ? (
        tab === "teams" ? (
          <TeamTable />
        ) : tab === "advanced" ? (
          <NflAdvTable />
        ) : null
      ) : tab === "teams" ? (
        <TeamTable />
      ) : (
        <MlbAdvTable />
      )}
    </section>
  );
};

export default StatsScreen;
