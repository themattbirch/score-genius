// frontend/src/components/games/game_card.tsx
import React, {
  useState,
  useEffect,
  useMemo,
  useCallback,
  useRef,
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
import PredBadge from "@/components/games/pred_badge";
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

/* ------------------------------------------------------------ */
/* Component                                                    */
/* ------------------------------------------------------------ */
interface GameCardProps {
  game: UnifiedGame;
  forceCompact?: boolean;
}

const isSameLocalDay = (a: Date, b: Date) =>
  a.getFullYear() === b.getFullYear() &&
  a.getMonth() === b.getMonth() &&
  a.getDate() === b.getDate();

const GameCardComponent: React.FC<GameCardProps> = ({ game, forceCompact }) => {
  const { sport: contextSport } = useSport();
  const isDesktop = useIsDesktop();
  const compactDefault = forceCompact ?? !isDesktop;
  const lastClickRef = useRef<number>(0);
  const arrowRef = useRef<HTMLSpanElement | null>(null);

  const { currentStepIndex, run: isTourRunning } = useTour();

  const [expanded, setExpanded] = useState(!compactDefault);
  const [snapshotOpen, setSnapshotOpen] = useState<boolean>(false);
  const [weatherOpen, setWeatherOpen] = useState<boolean>(false);

  // determine if this game is for "today" in user's local timezone based on its UTC time
  const isTodayGame = useMemo(() => {
    if (!game.gameTimeUTC) return false;
    const now = new Date();
    const gameDate = new Date(game.gameTimeUTC);
    return isSameLocalDay(now, gameDate);
  }, [game.gameTimeUTC]);

  /* ------------------------------------------------------------------ */
  /* 🔔 “View details” tooltip — exactly once per browser-tab session    */
  /* ------------------------------------------------------------------ */
  const TOOLTIP_SESSION_KEY = "sg_viewDetailsTooltipSeen";

  /**
   * Show only when:
   *   • the card is in compact (collapsed) mode
   *   • the tooltip hasn’t appeared yet this session
   */
  const [showTooltip, setShowTooltip] = useState(() => {
    if (typeof window === "undefined") return false;
    return (
      compactDefault &&
      !expanded &&
      !sessionStorage.getItem(TOOLTIP_SESSION_KEY)
    );
  });

  const markSeen = () => sessionStorage.setItem(TOOLTIP_SESSION_KEY, "1");

  const dismissTooltip = useCallback(() => {
    if (!showTooltip) return;
    setShowTooltip(false);
    markSeen();
  }, [showTooltip]);

  useEffect(() => {
    if (!isTodayGame) {
      setShowTooltip(false);
      return;
    }

    const eligible =
      compactDefault &&
      !expanded &&
      !showTooltip &&
      !sessionStorage.getItem(TOOLTIP_SESSION_KEY);

    if (eligible) {
      setShowTooltip(true);
    }
  }, [compactDefault, expanded, showTooltip, isTodayGame]);

  useEffect(() => {
    if (showTooltip && isTodayGame) {
      triggerTouchGlow(arrowRef.current);
      markSeen();
    }
  }, [showTooltip, isTodayGame]);

  /* Surface tooltip if the card later becomes eligible */
  useEffect(() => {
    const eligible =
      compactDefault &&
      !expanded &&
      !showTooltip &&
      !sessionStorage.getItem(TOOLTIP_SESSION_KEY);

    if (eligible) setShowTooltip(true);
  }, [compactDefault, expanded, showTooltip]);

  /* When tooltip becomes visible, trigger the arrow glow and record it */
  useEffect(() => {
    if (showTooltip) {
      triggerTouchGlow(arrowRef.current);
      markSeen();
    }
  }, [showTooltip]);

  // expansion toggles
  const toggleExpandedImmediate = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);
  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => !prev);
    dismissTooltip();
  }, [dismissTooltip]);

  const handleArrowInteraction = useCallback(
    (e: React.MouseEvent | React.KeyboardEvent) => {
      e.stopPropagation();
      triggerTouchGlow(arrowRef.current);
      lastClickRef.current = Date.now();
      toggleExpanded(); // toggles + dismisses
    },
    [toggleExpanded]
  );

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

  // debug: track when tooltip state becomes true
  useEffect(() => {
    if (showTooltip) {
      console.log("🛠 tooltip state true for game", gameId);
    }
  }, [showTooltip, gameId]);

  useEffect(() => {
    if (showTooltip) {
      triggerTouchGlow(arrowRef.current);
    }
  }, [showTooltip]);

  // keep expanded in sync with compact vs desktop mode
  useEffect(() => {
    setExpanded((prev) => {
      if (compactDefault && prev) return false; // switched to compact → collapse
      if (!compactDefault && !prev) return true; // switched to desktop → expand
      return prev;
    });
  }, [compactDefault]);

  // tour-aware expansion
  useEffect(() => {
    const TOUR_STEPS_REQUIRING_EXPANSION = [2, 3, 4];
    const needsExpansion =
      isTourRunning &&
      sport !== "NFL" &&
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

  const triggerTouchGlow = (node: HTMLElement | null) => {
    if (!node) return;
    node.classList.add("card-chevron--touch-glow");
    setTimeout(() => {
      node.classList.remove("card-chevron--touch-glow");
    }, 300); // matches transition duration
  };

  return (
    <article
      data-tour="game-card"
      className={`app-card ripple contain-layout
      ${compactDefault ? "app-card--compact" : ""}
      ${compactDefault && !expanded ? "edge-gradient" : ""}
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
              <div className="flex flex-col items-center w-full">
                <p className="font-semibold text-lg leading-tight text-center">
                  {away_final_score} – {home_final_score}
                </p>
                <span className="text-[10px] font-normal text-text-secondary mt-0.5 text-center">
                  final
                </span>
              </div>
            ) : hasPrediction ? (
              <PredBadge away={predAway as number} home={predHome as number} />
            ) : (
              <span className="text-sm text-text-secondary">—</span>
            )}

            {compactDefault && isTodayGame && (
              <div className="mt-2 flex w-full justify-center">
                <div className="relative">
                  <span
                    ref={arrowRef}
                    className={`card-chevron transition-transform ${
                      expanded ? "rotate-180" : ""
                    }`}
                    onMouseEnter={() => {
                      if (
                        !showTooltip &&
                        compactDefault &&
                        !expanded &&
                        !sessionStorage.getItem(TOOLTIP_SESSION_KEY)
                      ) {
                        setShowTooltip(true);
                      }
                    }}
                    onFocus={(e) => {
                      if (Date.now() - lastClickRef.current < 200) return;
                      if (
                        !showTooltip &&
                        compactDefault &&
                        !expanded &&
                        !sessionStorage.getItem(TOOLTIP_SESSION_KEY)
                      ) {
                        setShowTooltip(true);
                      }
                    }}
                    onClick={handleArrowInteraction}
                    aria-describedby={
                      showTooltip ? "gamecard-tooltip" : undefined
                    }
                    role="button"
                  >
                    ▾
                  </span>
                  {showTooltip && isTodayGame && (
                    <span
                      id="gamecard-tooltip"
                      role="tooltip"
                      className="absolute z-50 -top-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-panel)] px-2 py-1 text-xs shadow-lg text-[var(--color-text-primary)]"
                    >
                      View&nbsp;details
                    </span>
                  )}
                </div>
              </div>
            )}
          </div> /* <-- closes MLB/NBA right-side wrapper */
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
