// frontend/src/components/games/game_card.tsx
import React, { useState, useEffect, useMemo } from "react";
import { UnifiedGame, Sport } from "@/types";
import { useSport } from "@/contexts/sport_context";

import { useWeather } from "@/hooks/use_weather";
import WeatherBadge from "./weather_badge";
import WeatherModal from "./weather_modal";
import SnapshotButton from "./snapshot_button";
import SnapshotModal from "./snapshot_modal";

/* ------------------------------------------------------------ */
/* Helpers                                                      */
/* ------------------------------------------------------------ */
const formatTime = (iso?: string | null, fallback?: string | null): string =>
  iso
    ? new Date(iso).toLocaleTimeString([], {
        hour: "numeric",
        minute: "2-digit",
      })
    : fallback ?? "--";

const needsStatusSuffix = (statusState?: string | null): boolean =>
  statusState != null &&
  !["final", "sched", "pre"].some((s) => statusState.toLowerCase().includes(s));

const useIsDesktop = (bp = 1024): boolean => {
  const [isDesk, setIsDesk] = useState<boolean>(
    typeof window === "undefined" ? true : window.innerWidth >= bp
  );
  useEffect(() => {
    const handler = () => setIsDesk(window.innerWidth >= bp);
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, [bp]);
  return isDesk;
};

const PredBadge: React.FC<{ away: number; home: number }> = ({
  away,
  home,
}) => (
  <span className="pred-badge">
    {away.toFixed(1)} – {home.toFixed(1)}
    <span className="ml-1">pred.</span>
  </span>
);

/* ------------------------------------------------------------ */
/* Component                                                    */
/* ------------------------------------------------------------ */
interface GameCardProps {
  game: UnifiedGame;
  forceCompact?: boolean;
}

const GameCard: React.FC<GameCardProps> = ({ game, forceCompact }) => {
  const { sport: contextSport } = useSport();
  const isDesktop = useIsDesktop();
  const compactDefault = forceCompact ?? !isDesktop;

  const [expanded, setExpanded] = useState<boolean>(!compactDefault);
  const [snapshotOpen, setSnapshotOpen] = useState<boolean>(false);
  const [weatherOpen, setWeatherOpen] = useState<boolean>(false);

  // Log when modal states change
  useEffect(() => {
    console.log("snapshotOpen state:", snapshotOpen);
  }, [snapshotOpen]);
  useEffect(() => {
    console.log("weatherOpen state:", weatherOpen);
  }, [weatherOpen]);

  useEffect(() => {
    setExpanded(!compactDefault);
  }, [compactDefault]);

  const {
    id: gameId,
    awayTeamName,
    homeTeamName,
    gameTimeUTC,
    statusState,
    game_date,
    dataType,
    sport,
    away_final_score,
    home_final_score,
    predictionAway,
    predictionHome,
    predicted_away_runs,
    predicted_home_runs,
    awayPitcher,
    awayPitcherHand,
    homePitcher,
    homePitcherHand,
  } = game;

  const supportsWeather = sport === "MLB" || sport === "NFL";
  const locationParam: string | undefined =
    supportsWeather && homeTeamName ? homeTeamName : undefined;

  const {
    data: weatherData,
    isLoading: isWeatherLoading,
    isError: isWeatherError,
  } = useWeather(supportsWeather ? sport : undefined, locationParam);
  const isIndoor = weatherData?.isIndoor === true;

  const timeLine = useMemo(() => {
    const t = formatTime(gameTimeUTC ?? undefined, game_date ?? undefined);
    const suffix = needsStatusSuffix(statusState) ? ` (${statusState!})` : "";
    return `${t}${suffix}`;
  }, [gameTimeUTC, game_date, statusState]);

  const isFinal =
    dataType === "historical" &&
    away_final_score != null &&
    home_final_score != null;

  const predAway =
    contextSport === "NBA"
      ? predictionAway
      : predicted_away_runs ?? predictionAway;
  const predHome =
    contextSport === "NBA"
      ? predictionHome
      : predicted_home_runs ?? predictionHome;
  const hasPrediction =
    dataType === "schedule" && predAway != null && predHome != null;

  const toggleExpanded = () => {
    console.log("toggleExpanded fired, previous expanded:", expanded);
    setExpanded((prev) => !prev);
  };

  // Ignore clicks that originate inside elements marked data-action
  const handleCardClick = (e: React.MouseEvent<HTMLElement>) => {
    console.log("CARD click", { compactDefault, target: e.target });
    if (!compactDefault) return; // desktop: do nothing
    const target = e.target as HTMLElement;
    if (target.closest("[data-action]")) {
      console.log("CARD click ignored due to data-action");
      return; // let the chip handle it
    }
    toggleExpanded();
  };

  return (
    <article
      className={`app-card ripple
${compactDefault ? "app-card--compact" : ""}
${expanded && !isDesktop ? "md:col-span-2" : ""}`}
      aria-expanded={expanded}
      onClick={handleCardClick}
    >
      {/* Header */}
      <header
        className="flex items-start gap-4 cursor-pointer"
        role={compactDefault ? "button" : undefined}
        tabIndex={compactDefault ? 0 : -1}
        onKeyDown={
          compactDefault
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  console.log("HEADER key press toggled expanded");
                  e.preventDefault();
                  toggleExpanded();
                }
              }
            : undefined
        }
      >
        <div className="min-w-0 flex-1">
          <p className="font-semibold text-sm sm:text-base leading-tight break-words">
            {awayTeamName}
          </p>
          <p className="font-semibold text-sm sm:text-base leading-tight break-words">
            {homeTeamName}
          </p>
          <p className="text-xs text-text-secondary mt-1">{timeLine}</p>
        </div>

        <div className="flex flex-col items-end text-right gap-1 ml-auto">
          {isFinal ? (
            <p className="font-semibold text-lg whitespace-nowrap leading-tight">
              {away_final_score} – {home_final_score}
              <span className="block text-[10px] font-normal text-text-secondary mt-0.5">
                final
              </span>
            </p>
          ) : hasPrediction ? (
            <PredBadge away={predAway as number} home={predHome as number} />
          ) : (
            <span className="text-sm text-text-secondary">—</span>
          )}

          {compactDefault && (
            <span
              className={`mt-0.5 inline-block text-text-secondary transition-transform ${
                expanded ? "rotate-180" : ""
              }`}
            >
              ▾
            </span>
          )}
        </div>
      </header>

      {/* Expanded Content */}
      {expanded && (
        <div className="mt-4 flex items-center justify-between">
          {/* Pitchers */}
          {sport === "MLB" && (
            <div className="flex flex-col justify-center text-xs text-text-secondary leading-tight">
              <p>
                {awayPitcher ?? "TBD"}{" "}
                {awayPitcherHand && `(${awayPitcherHand})`}
              </p>
              <p>
                {homePitcher ?? "TBD"}{" "}
                {homePitcherHand && `(${homePitcherHand})`}
              </p>
            </div>
          )}

          {/* Action Chips */}
          <div className="flex flex-col gap-2">
            <SnapshotButton
              data-action
              onMouseDown={(e) => {
                console.log("▶️ Snapshot onMouseDown");
                e.stopPropagation();
              }}
              onClick={(e) => {
                console.log("✅ Snapshot onClick");
                e.preventDefault();
                e.stopPropagation();
                setSnapshotOpen(true);
              }}
            />

            {supportsWeather && (
              <WeatherBadge
                data-action
                isLoading={isWeatherLoading}
                isError={isWeatherError}
                data={weatherData}
                isIndoor={isIndoor}
                onMouseDown={(e) => {
                  console.log("▶️ Weather onMouseDown");
                  e.stopPropagation();
                }}
                onClick={(e) => {
                  console.log("✅ Weather onClick");
                  e.preventDefault();
                  e.stopPropagation();
                  setWeatherOpen(true);
                }}
              />
            )}
          </div>
        </div>
      )}

      <SnapshotModal
        gameId={gameId}
        sport={sport as Sport}
        isOpen={snapshotOpen}
        onClose={() => setSnapshotOpen(false)}
      />

      <WeatherModal
        isOpen={weatherOpen}
        onClose={() => setWeatherOpen(false)}
        weatherData={weatherData}
        isIndoor={isIndoor}
      />
    </article>
  );
};

export default GameCard;
