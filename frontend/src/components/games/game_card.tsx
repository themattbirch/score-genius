// frontend/src/components/games/game_card.tsx
import React, {
  useState,
  useEffect,
  useMemo,
  useCallback,
  lazy,
  Suspense,
  memo,
} from "react";
import { UnifiedGame, Sport } from "@/types";
import { useSport } from "@/contexts/sport_context";

import { useWeather } from "@/hooks/use_weather";
import WeatherBadge from "./weather_badge";
const WeatherModal = lazy(() => import("./weather_modal"));
import SnapshotButton from "./snapshot_button";
const SnapshotModal = lazy(() => import("./snapshot_modal"));

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

const GameCardComponent: React.FC<GameCardProps> = ({ game, forceCompact }) => {
  const { sport: contextSport } = useSport();
  const isDesktop = useIsDesktop();
  const compactDefault = forceCompact ?? !isDesktop;

  const [expanded, setExpanded] = useState<boolean>(!compactDefault);
  const [snapshotOpen, setSnapshotOpen] = useState<boolean>(false);
  const [weatherOpen, setWeatherOpen] = useState<boolean>(false);

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

  // 1. Differentiate between sports that need a weather API call vs. just UI
  const shouldFetchWeather = sport === "MLB" || (sport === "NFL" && expanded);
  const isNBA = sport === "NBA";
  const showWeatherUI = shouldFetchWeather || isNBA;

  const {
    data: weatherData,
    isLoading: isWeatherLoading,
    isError: isWeatherError,
  } = useWeather(
    shouldFetchWeather ? sport : undefined,
    shouldFetchWeather ? homeTeamName : undefined
  );
  const isEffectivelyIndoor = isNBA || weatherData?.isIndoor === true;

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

  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  // Ignore clicks that originate inside elements marked data-action
  const handleCardClick = useCallback(
    (e: React.MouseEvent<HTMLElement>) => {
      if (!compactDefault) return; // desktop: do nothing
      const target = e.target as HTMLElement;
      if (target.closest("[data-action]")) {
        return; // let the chip handle it
      }
      toggleExpanded();
    },
    [compactDefault, toggleExpanded]
  );
  return (
    <article
      data-tour="game-card"
      className={`app-card ripple contain-layout
      ${compactDefault ? "app-card--compact" : ""}
      ${expanded && !isDesktop ? "md:col-span-2" : ""}`}
      aria-expanded={expanded}
      onClick={handleCardClick}
    >
      {/* Header */}
      <header
        className="flex items-start justify-between gap-4 cursor-pointer"
        role={compactDefault ? "button" : undefined}
        tabIndex={compactDefault ? 0 : -1}
        onKeyDown={
          compactDefault
            ? (e: React.KeyboardEvent<HTMLElement>) => {
                if (e.key === "Enter" || e.key === " ") {
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

        <div className="flex flex-col items-end text-right gap-1">
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

          {/* Centered chevron row (mobile only) */}
          {compactDefault && (
            <div className="mt-2 flex w-full justify-center">
              <span
                className={`card-chevron transition-transform ${
                  expanded ? "rotate-180" : ""
                }`}
              >
                ▾
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Expanded Content */}
      {expanded &&
        (() => {
          const hasPitchers =
            sport === "MLB" &&
            ((awayPitcher && awayPitcher.trim() !== "") ||
              (homePitcher && homePitcher.trim() !== ""));

          return (
            <div
              className={`mt-4 flex items-center gap-2 ${
                hasPitchers ? "justify-between" : "justify-start"
              }`}
            >
              {/* Pitchers — render only when at least one name exists */}
              {hasPitchers && (
                <div className="flex flex-col justify-center text-xs text-text-secondary leading-tight">
                  {awayPitcher && (
                    <p>
                      {awayPitcher} {awayPitcherHand && `(${awayPitcherHand})`}
                    </p>
                  )}
                  {homePitcher && (
                    <p>
                      {homePitcher} {homePitcherHand && `(${homePitcherHand})`}
                    </p>
                  )}
                </div>
              )}

              {/* Action Chips */}
              <div className="flex flex-col gap-2">
                <SnapshotButton
                  data-action
                  onClick={(e) => {
                    e.stopPropagation();
                    setSnapshotOpen(true);
                  }}
                />
                {showWeatherUI && (
                  <WeatherBadge
                    data-action
                    data-tour="weather-badge"
                    // The following props were made conditional for NBA
                    isIndoor={isEffectivelyIndoor}
                    isLoading={!isNBA && isWeatherLoading}
                    isError={!isNBA && isWeatherError}
                    data={weatherData}
                    onClick={(e) => {
                      e.stopPropagation();
                      setWeatherOpen(true);
                    }}
                  />
                )}
              </div>
            </div>
          );
        })()}
      <Suspense fallback={null}>
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
          // And use the new variable here as well
          isIndoor={isEffectivelyIndoor}
        />
      </Suspense>
    </article>
  );
};
const GameCard = memo(GameCardComponent);
export default GameCard;
