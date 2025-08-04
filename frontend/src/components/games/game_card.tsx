// frontend/src/components/games/game_card.tsx
import React, {
  useState,
  useEffect,
  useMemo,
  useCallback,
  useRef,
  useLayoutEffect,
  lazy,
  Suspense,
  memo,
} from "react";
import { UnifiedGame, Sport } from "@/types";
import { useSport } from "@/contexts/sport_context";
import { useTour } from "@/contexts/tour_context";
import OddsDisplay from "./odds_display";

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

const TOOLTIP_SESSION_KEY = "sg_viewDetailsTooltipSeen";

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

// Lazy load injury components
const InjuriesChipButton = lazy(
  () => import("@/components/ui/injuries_chip_button")
);
const InjuryModal = lazy(() => import("./injury_modal"));

/* ------------------------------------------------------------ */
/* Component                                                    */
/* ------------------------------------------------------------ */
interface GameCardProps {
  game: UnifiedGame;
  forceCompact?: boolean;
  isFirst?: boolean;
}

const isSameLocalDay = (a: Date, b: Date) =>
  a.getFullYear() === b.getFullYear() &&
  a.getMonth() === b.getMonth() &&
  a.getDate() === b.getDate();

const GameCardComponent: React.FC<GameCardProps> = ({
  game,
  forceCompact,
  isFirst = false,
}) => {
  const { sport: contextSport } = useSport();
  const isDesktop = useIsDesktop();
  const compactDefault = forceCompact ?? !isDesktop;
  const lastClickRef = useRef<number>(0);
  const tooltipRef = useRef<HTMLSpanElement | null>(null);
  const arrowRef = useRef<HTMLSpanElement | null>(null);
  const [hoverTooltip, setHoverTooltip] = useState(false);

  // determine if this game is for "today" in user's local timezone based on its UTC time
  const isTodayGame = useMemo(() => {
    const dateSrc = game.gameTimeUTC ?? game.game_date; // ISO or YYYY-MM-DD
    if (!dateSrc) return false;
    const gameDate = new Date(dateSrc);
    const now = new Date();
    return isSameLocalDay(now, gameDate);
  }, [game.gameTimeUTC, game.game_date]);

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

  // cutoff after game is over: 3.5 hours after scheduled time
  // cutoff after game is over: 3.5 hours after scheduled time
  const GAME_STALE_MS = 3.5 * 60 * 60 * 1000; // 3.5h

  // parse the game start time
  const gameStartDate = useMemo(() => {
    const dateSrc = game.gameTimeUTC ?? game.game_date;
    return dateSrc ? new Date(dateSrc) : null;
  }, [game.gameTimeUTC, game.game_date]);

  const now = Date.now();

  const isInProgress =
    gameStartDate !== null &&
    now >= gameStartDate.getTime() &&
    now < gameStartDate.getTime() + GAME_STALE_MS;

  const isStale =
    gameStartDate !== null &&
    isTodayGame &&
    now >= gameStartDate.getTime() + GAME_STALE_MS;

  const isFinal = (() => {
    const status = (statusState ?? "").toLowerCase();
    if (
      ["final", "ended", "ft", "post-game", "postgame", "completed"].some((s) =>
        status.includes(s)
      )
    ) {
      return true;
    }
    if (away_final_score != null && home_final_score != null) {
      return true;
    }
    return false;
  })();

  const [expanded, setExpanded] = useState(!compactDefault);

  const { showTooltip, setShowTooltip, dismissTooltip } = useViewDetailsTooltip(
    {
      isFirst,
      isTodayGame,
      compactDefault,
      expanded,
      isInProgress,
    }
  );

  const { currentStepIndex, run: isTourRunning } = useTour();

  const [snapshotOpen, setSnapshotOpen] = useState<boolean>(false);
  const [weatherOpen, setWeatherOpen] = useState<boolean>(false);
  const [injuryModalOpen, setInjuryModalOpen] = useState<boolean>(false);

  useLayoutEffect(() => {
    if (!tooltipRef.current || !arrowRef.current) return;
    if (!(showTooltip || hoverTooltip)) return;
    if (isInProgress) return;

    const tooltipEl = tooltipRef.current;
    const arrowEl = arrowRef.current;

    // Measure
    const arrowRect = arrowEl.getBoundingClientRect();
    const tooltipRect = tooltipEl.getBoundingClientRect();
    const viewportWidth = window.innerWidth;

    // Desired left (centered over arrow) in viewport coords
    const absoluteLeft =
      arrowRect.left + arrowRect.width / 2 - tooltipRect.width / 2;

    // Clamp so it doesn't overflow
    const min = 4; // padding from left edge
    const max = viewportWidth - tooltipRect.width - 4; // padding from right
    const clampedLeft = Math.min(Math.max(absoluteLeft, min), max);

    // Compute left relative to offsetParent (since tooltip is absolutely positioned inside)
    const offsetParent = tooltipEl.offsetParent as HTMLElement | null;
    const parentLeft = offsetParent
      ? offsetParent.getBoundingClientRect().left
      : 0;
    const relativeLeft = clampedLeft - parentLeft;

    // Apply positioning: explicit left, and clear any centering transform
    tooltipEl.style.left = `${relativeLeft}px`;
    tooltipEl.style.transform = ""; // remove translateX(-50%) if previously set
  }, [showTooltip, hoverTooltip, isInProgress]);

  /* ------------------------------------------------------------------ */
  /* 🔔 “View details” tooltip — exactly once per browser-tab session    */
  /* ------------------------------------------------------------------ */
  function useViewDetailsTooltip({
    isFirst,
    isTodayGame,
    compactDefault,
    expanded,
    isInProgress,
  }: {
    isFirst: boolean;
    isTodayGame: boolean;
    compactDefault: boolean;
    expanded: boolean;
    isInProgress: boolean;
  }) {
    const isTooltipEligible = useMemo(() => {
      return (
        !!isFirst &&
        isTodayGame &&
        compactDefault &&
        !expanded &&
        !isInProgress &&
        !sessionStorage.getItem(TOOLTIP_SESSION_KEY)
      );
    }, [isFirst, isTodayGame, compactDefault, expanded]);

    const [showTooltip, setShowTooltip] = useState(() => {
      if (typeof window === "undefined") return false;
      return isTooltipEligible;
    });

    const markSeen = useCallback(() => {
      sessionStorage.setItem(TOOLTIP_SESSION_KEY, "1");
    }, []);

    const dismissTooltip = useCallback(() => {
      if (!showTooltip) return;
      setShowTooltip(false);
      markSeen();
    }, [showTooltip, markSeen]);

    // Sync showTooltip to eligibility; if eligibility goes away hide it.
    useEffect(() => {
      if (isTooltipEligible) {
        setShowTooltip(true);
      } else {
        setShowTooltip(false);
      }
    }, [isTooltipEligible]);

    // Persist the fact we showed it (only when it's visible and eligible)
    useEffect(() => {
      if (showTooltip && isTooltipEligible) {
        markSeen();
      }
    }, [showTooltip, isTooltipEligible, markSeen]);

    return {
      showTooltip,
      setShowTooltip,
      dismissTooltip,
      isTooltipEligible,
    };
  }

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
      setExpanded((prev) => !prev);
      // dismiss after state change intent, not coupled to it
      dismissTooltip();
    },
    [dismissTooltip]
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
      className={`app-card ripple contain-layout relative
    ${compactDefault ? "app-card--compact" : ""}
    ${compactDefault && !expanded ? "edge-gradient" : ""}
    ${expanded && !isDesktop ? "md:col-span-2" : ""}`}
      aria-expanded={expanded}
      onClick={handleCardClick}
    >
      {/* Header */}
      <header
        className={`flex items-start justify-between gap-4 cursor-pointer ${
          sport === "NFL" ? "pb-8" : ""
        }`}
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
        <div className="min-w-0 flex-1 flex flex-col">
          <div>
            <p className="font-semibold text-sm sm:text-base leading-tight break-words">
              {awayTeamName}
            </p>
            <p className="font-semibold text-sm sm:text-base leading-tight break-words">
              {homeTeamName}
            </p>
            <p className="text-xs text-text-secondary mt-1">{timeLine}</p>
          </div>

          {/* Injuries button logic... */}
          {(sport === "NFL" ||
            (sport === "NBA" && compactDefault && !expanded && !isStale)) && (
            <div className="mt-2">
              <InjuriesChipButton
                onClick={(e) => {
                  e.stopPropagation();
                  setInjuryModalOpen(true);
                }}
              />
            </div>
          )}

          {/* Odds Display: Show for ANY non-final game in compact view */}
          {!isFinal && (
            <OddsDisplay
              sport={sport}
              moneylineHome={game.moneylineHome}
              moneylineAway={game.moneylineAway}
              spreadLine={game.spreadLine}
              totalLine={game.totalLine}
            />
          )}
        </div>

        {/* We conditionally render a different layout for the right side based on the sport */}
        {sport === "NFL" ? (
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
          <div
            className={`flex flex-col items-end text-right gap-1 ${
              !isFinal && !hasPrediction ? "pr-8" : ""
            }`}
          >
            {" "}
            {isFinal ? (
              <div className="flex flex-col items-center w-full">
                <p className="font-semibold text-lg leading-tight text-center">
                  {away_final_score} – {home_final_score}
                </p>
                <span className="text-[10px] font-normal text-text-secondary mt-0.5 text-center">
                  final
                </span>
              </div>
            ) : isInProgress ? (
              <span className="text-xs font-medium text-text-secondary tracking-wide">
                in&nbsp;progress
              </span>
            ) : hasPrediction ? (
              <PredBadge away={predAway as number} home={predHome as number} />
            ) : (
              <span className="text-sm text-text-secondary w-full text-center">
                —
              </span>
            )}
            {compactDefault && isTodayGame && (
              <div className="mt-2 flex w-full justify-center">
                <div className="relative">
                  <span
                    ref={arrowRef}
                    className={`card-chevron transition-transform ${
                      expanded ? "rotate-180" : ""
                    }`}
                    style={{
                      transform: expanded ? "rotate(180deg)" : "none",
                      transition: "transform 0.2s ease",
                    }}
                    onMouseEnter={() => {
                      // only show hover tooltip if persistent one isn't showing and basic eligibility (today, compact collapsed)
                      if (
                        !showTooltip &&
                        isTodayGame &&
                        compactDefault &&
                        !expanded &&
                        !isInProgress
                      ) {
                        setHoverTooltip(true);
                      }
                    }}
                    onMouseLeave={() => {
                      setHoverTooltip(false);
                    }}
                    onFocus={(e) => {
                      // keep existing logic for persistent tooltip via focus, not hover
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
                  {(showTooltip || hoverTooltip) &&
                    isTodayGame &&
                    !isInProgress && (
                      <span
                        ref={tooltipRef}
                        id="gamecard-tooltip"
                        role="tooltip"
                        style={{
                          left: "50%",
                          transform: "translateX(-50%)",
                          maxWidth: "160px",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                          pointerEvents: "none",
                        }}
                        className="absolute z-50 -top-[2.375rem] whitespace-nowrap rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-panel)] px-2 py-1 text-xs shadow-lg text-[var(--color-text-primary)] transition-opacity"
                      >
                        View&nbsp;details
                      </span>
                    )}
                </div>
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
      {/* Injuries chip overlay for non-NFL cards (compact, collapsed, non-final, non-MLB) */}
      {sport !== "NFL" &&
        compactDefault &&
        !expanded &&
        isTodayGame &&
        !isFinal &&
        sport !== "MLB" && (
          <div className="absolute bottom-2 left-0 w-full px-4 flex justify-start pointer-events-none">
            <div className="pointer-events-auto">
              <InjuriesChipButton
                onClick={(e) => {
                  e.stopPropagation();
                  setInjuryModalOpen(true);
                }}
              />
            </div>
          </div>
        )}

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
        <InjuryModal
          isOpen={injuryModalOpen}
          onClose={() => setInjuryModalOpen(false)}
          league={sport}
          gameDate={game.game_date}
          teamNames={[game.awayTeamName, game.homeTeamName]}
        />
      </Suspense>
    </article>
  );
};
const GameCard = memo(GameCardComponent);
export default GameCard;
