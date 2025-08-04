// frontend/src/utils/game.ts
import type { UnifiedGame } from "@/types";

/** Helper: are two dates the same local calendar day? */
const isSameLocalDay = (a: Date, b: Date) =>
  a.getFullYear() === b.getFullYear() &&
  a.getMonth() === b.getMonth() &&
  a.getDate() === b.getDate();

/**
 * Returns true if the game should be considered stale/archived:
 * - only for games happening today: 3.5 hours have elapsed since scheduled UTC start
 * - fallback: historical game with final scores (for today only—past days remain)
 */
export function isGameStale(game: UnifiedGame, now = Date.now()): boolean {
  const bufferMs = 3.5 * 60 * 60 * 1000; // 3.5h
  const ts = new Date(game.gameTimeUTC ?? "").getTime();
  if (Number.isNaN(ts)) return false; // can't judge, let higher-level logic handle finals

  const gameDate = new Date(game.gameTimeUTC ?? game.game_date);
  const nowDate = new Date(now);

  // Only consider stale if it's the same local day as today
  if (!isSameLocalDay(nowDate, gameDate)) {
    return false;
  }

  // Standard stale window
  if (now >= ts + bufferMs) {
    return true;
  }

  // fallback: if it's a historical final from today (should be rare because historical likely isn't today)
  if (
    game.dataType === "historical" &&
    game.home_final_score != null &&
    game.away_final_score != null &&
    isSameLocalDay(nowDate, gameDate)
  ) {
    return true;
  }

  return false;
}
