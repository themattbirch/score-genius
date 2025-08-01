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
import { useTour } from "@/contexts/tour_context";

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

  const { currentStepIndex, run: isTourRunning } = useTour();

  const [expanded, setExpanded] = useState<boolean>(!compactDefault);
  const [snapshotOpen, setSnapshotOpen] = useState<boolean>(false);
  const [weatherOpen, setWeatherOpen] = useState<boolean>(false);

  const H2HButton = () => (
    <SnapshotButton
      data-action
      data-tour="snapshot-button"
      label="H2H Stats"
      onClick={(e) => {
        e.stopPropagation();
        setSnapshotOpen(true);
      }}
    />
  );

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
    predictionAway, // NBA
    predictionHome, // NBA
    predicted_away_runs, // MLB
    predicted_home_runs, // MLB
    predicted_away_score, // NFL
    predicted_home_score, // NFL
    awayPitcher,
    awayPitcherHand,
    homePitcher,
    homePitcherHand,
  } = game;

  // Make the card "tour-aware"
  useEffect(() => {
    // These are the tour step indices that target elements inside the card.
    // Adjust these numbers if you change the order of your tour steps.
    const TOUR_STEPS_REQUIRING_EXPANSION = [2, 3, 4];

    const needsExpansion =
      isTourRunning &&
      sport !== "NFL" && // Only for sports with a collapsible section
      TOUR_STEPS_REQUIRING_EXPANSION.includes(currentStepIndex);

    if (needsExpansion) {
      setExpanded(true);
    }
  }, [isTourRunning, currentStepIndex, sport]);

  // 1. Differentiate between sports that need a weather API call vs. just UI
  const shouldFetchWeather = sport === "MLB" || sport === "NFL";
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

  // 👇 **MODIFIED LOGIC**
  // This logic now correctly handles the different prediction fields
  // for each sport based on your UnifiedGame type.
  let predAway, predHome;
  switch (contextSport) {
    case "NBA":
      predAway = predictionAway;
      predHome = predictionHome;
      break;
    case "MLB":
      predAway = predicted_away_runs;
      predHome = predicted_home_runs;
      break;
    case "NFL":
      predAway = predicted_away_score;
      predHome = predicted_home_score;
      break;
  }
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

        {/* We conditionally render a different layout for the right side based on the sport */}
        {sport === "NFL" ? (
          // --- This is the new layout for NFL cards ---
          <div className="flex flex-col items-end gap-2 shrink-0">
            <H2HButton />
            <WeatherBadge
              data-action
              data-tour="weather-badge"
              isIndoor={isEffectivelyIndoor}
              // 👇 This is the fix. The conditions are removed.
              isLoading={isWeatherLoading}
              isError={isWeatherError}
              data={weatherData}
              onClick={(e) => {
                e.stopPropagation();
                setWeatherOpen(true);
              }}
            />
          </div>
        ) : (
          // --- This is the original layout for MLB/NBA cards ---
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
        )}
      </header>

      {/* Expanded Content */}
      {sport !== "NFL" &&
        expanded &&
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
