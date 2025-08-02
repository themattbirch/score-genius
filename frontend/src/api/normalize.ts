// frontend/src/api/normalize.ts

import type { NflTeamSummary } from "@/types";

/**
 * Shallow snake_case → camelCase key mapper.
 * Only converts top-level keys.
 */
export function snakeToCamel(s: string): string {
  return s.replace(/_([a-z])/g, (_, char) => char.toUpperCase());
}

/**
 * Normalize a raw NFL summary row coming from the backend into canonical camelCase,
 * with renamed fields (srs_lite → srs, win_pct → winPct) and defaults.
 */
export function normalizeNflTeamSummaryRow(raw: Record<string, any>) {
  const out: Record<string, any> = {};

  // Camel-case everything
  Object.entries(raw).forEach(([k, v]) => {
    const camel = k.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
    out[camel] = v;
  });

  // Ensure required canonical fields
  if (out.team_id !== undefined && out.teamId === undefined) {
    out.teamId = String(out.team_id);
  }
  if (out.team_name !== undefined && out.teamName === undefined) {
    out.teamName = out.team_name;
  }

  // season: coerce to number if present
  if (out.season !== undefined) {
    out.season = Number(out.season);
  }

  // Aliases / fallbacks
  if (out.srsLite !== undefined && out.srs === undefined) {
    out.srs = out.srsLite;
  }

  if (
    out.pythagorean_win_pct !== undefined &&
    out.pythagoreanWinPct === undefined
  ) {
    out.pythagoreanWinPct = out.pythagorean_win_pct;
  }

  // keep sosRank if present (no change)
  // winPct should already be in camel if backend provided win_pct
  if (out.win_pct !== undefined && out.winPct === undefined) {
    out.winPct = out.win_pct;
  }

  return out as unknown as NflTeamSummary;
}
