// frontend/src/screens/game_detail_screen.tsx
import React, { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { useNBASchedule } from "@/api/use_nba_schedule";
import { useInjuries } from "@/api/use_injuries";
import SkeletonBox from "@/components/ui/skeleton_box";
import type { UnifiedGame } from "@/types";

/* ────────────────────────────────────────────────────────────── */
/* Helper: Prediction badge                                       */
/* ────────────────────────────────────────────────────────────── */
const PredBadge: React.FC<{ away: number; home: number }> = ({
  away,
  home,
}) => (
  <span className="pred-badge px-3 py-1">
    {away.toFixed(1)} – {home.toFixed(1)}
    <span className="ml-1">pred.</span>
  </span>
);

/* ────────────────────────────────────────────────────────────── */
/* Main component                                                 */
/* ────────────────────────────────────────────────────────────── */
const TABS = ["Overview", "H2H", "Weather", "Odds", "Snapshots"] as const;
type Tab = (typeof TABS)[number];

const GameDetailScreen: React.FC = () => {
  const { gameId = "" } = useParams<{ gameId?: string }>();
  const isoDate = new Date().toISOString().slice(0, 10);

  /* Schedule row */
  const {
    data: games = [],
    isLoading: loadingGames,
    error: gamesError,
  } = useNBASchedule(isoDate);

  const thisGame: UnifiedGame | undefined = games.find((g) => g.id === gameId);

  /* Injuries */
  const {
    data: injuries = [],
    isLoading: loadingInjuries,
    error: injuriesError,
  } = useInjuries("NBA", isoDate);

  /* Early states */
  if (loadingGames || (loadingInjuries && thisGame))
    return <SkeletonBox className="h-screen w-full p-4" />;

  if (gamesError)
    return (
      <p className="p-4 text-red-500">Error loading game schedule data.</p>
    );

  if (!thisGame)
    return (
      <p className="p-4 text-orange-500">
        Game {gameId} not found for {isoDate}.
      </p>
    );

  /* Tab state */
  const [tab, setTab] = useState<Tab>("Overview");

  /* Memo helpers */
  const headerTime = useMemo(() => {
    if (!thisGame.gameTimeUTC) return thisGame.game_date;
    return new Date(thisGame.gameTimeUTC).toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
  }, [thisGame.gameTimeUTC, thisGame.game_date]);

  /* ────────────────────────── render ───────────────────────── */
  return (
    <div className="px-4 pb-10 space-y-6">
      {/* Game summary card */}
      <div className="app-card p-6 space-y-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h2 className="text-xl sm:text-2xl font-semibold">
              {thisGame.awayTeamName} @ {thisGame.homeTeamName}
            </h2>
            <p className="mt-1 text-sm text-text-secondary">
              {thisGame.game_date} / {headerTime}
            </p>
            {(thisGame.spreadLine != null || thisGame.totalLine != null) && (
              <p className="mt-1 text-xs text-text-secondary">
                Spread {thisGame.spreadLine ?? "N/A"}, Total{" "}
                {thisGame.totalLine ?? "N/A"}
              </p>
            )}
          </div>

          <div className="text-right">
            {thisGame.home_final_score != null &&
            thisGame.away_final_score != null ? (
              <p className="text-lg font-semibold">
                {thisGame.away_final_score} – {thisGame.home_final_score}
                <span className="block text-[10px] font-normal text-text-secondary">
                  final
                </span>
              </p>
            ) : thisGame.predictionHome != null &&
              thisGame.predictionAway != null ? (
              <PredBadge
                away={thisGame.predictionAway}
                home={thisGame.predictionHome}
              />
            ) : (
              <span className="text-sm text-text-secondary">—</span>
            )}
          </div>
        </div>

        {/* Tab bar */}
        <nav className="tab-bar">
          {TABS.map((t) => (
            <button
              key={t}
              className={`tab ${tab === t ? "tab-active" : ""}`}
              onClick={() => setTab(t)}
              aria-selected={tab === t}
            >
              {t}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab panels */}
      <div className="space-y-6">
        {tab === "Overview" && (
          <div className="app-card p-4">
            <p className="text-sm text-text-secondary">
              Quick overview will go here (team form, pace, recent scores,
              etc.). Replace this placeholder with real content.
            </p>
          </div>
        )}

        {tab === "H2H" && (
          <div className="app-card p-4">
            <p className="text-sm text-text-secondary">
              Head‑to‑head stats placeholder.
            </p>
          </div>
        )}

        {tab === "Weather" && (
          <div className="app-card p-4">
            <p className="text-sm text-text-secondary">
              Game‑time weather details (wind, temp, humidity)… coming soon.
            </p>
          </div>
        )}

        {tab === "Odds" && (
          <div className="app-card p-4">
            <p className="text-sm text-text-secondary">
              Live odds / line movement placeholder.
            </p>
          </div>
        )}

        {tab === "Snapshots" && (
          <div className="app-card p-4">
            <p className="text-sm text-text-secondary">
              Snapshot images / charts placeholder.
            </p>
          </div>
        )}

        {/* Injury panel always visible under its tab */}
        {tab === "Overview" && injuries.length > 0 && (
          <div className="app-card p-4">
            <h3 className="mb-2 font-semibold">Injury Report</h3>
            <ul className="space-y-1 text-sm">
              {injuries.map((inj) => (
                <li key={inj.id} className="flex justify-between">
                  <span>
                    {inj.player}
                    {inj.injury_type && ` (${inj.injury_type})`} —{" "}
                    <em>{inj.team_display_name}</em>
                  </span>
                  <span className="font-medium">{inj.status}</span>
                </li>
              ))}
            </ul>
            {injuriesError && (
              <p className="mt-2 text-xs text-red-500">
                Could not load injury details.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default GameDetailScreen;
