// frontend/src/components/games/game_card.tsx

import React, { useState } from "react";
import { UnifiedGame, Sport } from "@/types";
import { useSport } from "@/contexts/sport_context";

import { useWeather } from "@/hooks/use_weather";
import WeatherBadge from "./weather_badge";
import WeatherModal from "./weather_modal";
import SnapshotButton from "./snapshot_button";
import SnapshotModal from "./snapshot_modal";

interface GameCardProps {
  game: UnifiedGame;
}

const GameCard: React.FC<GameCardProps> = ({ game }) => {
  const { sport: contextSport } = useSport();
  const {
    id: gameId,
    homeTeamName,
    awayTeamName,
    gameTimeUTC,
    statusState,
    game_date,
    dataType,
    sport,
    away_final_score,
    home_final_score,
    predictionAway,
    predictionHome,
    predicted_home_runs,
    predicted_away_runs,
    awayPitcher,
    awayPitcherHand,
    homePitcher,
    homePitcherHand,
  } = game;

  /* ––––––––––––– local state ––––––––––––– */
  const [isSnapshotModalOpen, setSnapshotModalOpen] = useState(false);
  const [isWeatherModalOpen, setWeatherModalOpen] = useState(false);

  /* ––––––––––––– weather hook ––––––––––––– */
  const supportsWeather = sport === "MLB" || sport === "NFL";
  const {
    data: weatherData,
    isLoading: isWeatherLoading,
    isError: isWeatherError,
  } = useWeather(
    supportsWeather ? sport : undefined,
    supportsWeather ? homeTeamName : undefined
  );
  const isIndoor = weatherData?.isIndoor === true;

  /* ––––––––––––– helpers ––––––––––––– */
  const formattedTime = gameTimeUTC
    ? new Date(gameTimeUTC).toLocaleTimeString([], {
        hour: "numeric",
        minute: "2-digit",
      })
    : game_date;
  const statusSuffix =
    statusState &&
    !["final", "sched", "pre"].some((s) =>
      statusState.toLowerCase().includes(s)
    )
      ? ` (${statusState})`
      : "";

  /* ––––––––––––– render ––––––––––––– */
  return (
    <div className="app-card flex flex-col gap-5" data-tour="game-card">
      <div className="flex items-center justify-between gap-0">
        {/* left column */}
        <div className="min-w-0 flex-1 flex flex-col gap-1.5">
          <p className="font-semibold text-sm sm:text-base leading-tight break-words">
            {awayTeamName}
          </p>
          <p className="font-semibold text-sm sm:text-base leading-tight break-words">
            {homeTeamName}
          </p>
          <p className="text-xs text-text-secondary">
            {formattedTime}
            {statusSuffix}
          </p>
          <SnapshotButton
            className="mt-2"
            onClick={() => setSnapshotModalOpen(true)}
          />
        </div>

        {/* right column */}
        <div className="w-44 flex flex-col items-end gap-4 text-sm">
          {/* score / predictions */}
          {dataType === "historical" ? (
            <p className="font-semibold text-lg whitespace-nowrap">
              {away_final_score ?? "-"} – {home_final_score ?? "-"}
              <span className="block text-xs font-normal text-text-secondary">
                (Final)
              </span>
            </p>
          ) : dataType === "schedule" ? (
            contextSport === "NBA" ? (
              <p className="font-medium text-green-600 dark:text-green-500 whitespace-nowrap text-right">
                {predictionAway?.toFixed(1) ?? "-"} –{" "}
                {predictionHome?.toFixed(1) ?? "-"}
                <span className="block text-xs font-normal text-text-secondary">
                  (Predicted Score)
                </span>
              </p>
            ) : (
              <>
                {sport === "MLB" &&
                  predicted_away_runs != null &&
                  predicted_home_runs != null && (
                    <p className="font-medium text-green-600 dark:text-green-500 whitespace-nowrap text-right">
                      {predicted_away_runs.toFixed(1)} –{" "}
                      {predicted_home_runs.toFixed(1)}
                      <span className="block text-xs font-normal text-text-secondary">
                        (Predicted Score)
                      </span>
                    </p>
                  )}
                {sport === "MLB" && (
                  <div className="space-y-0.5 text-xs text-text-secondary text-right leading-tight">
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
              </>
            )
          ) : (
            <p className="font-medium">—</p>
          )}

          {/* weather badge */}
          {supportsWeather && (
            <WeatherBadge
              isLoading={isWeatherLoading}
              isError={isWeatherError}
              data={weatherData}
              isIndoor={isIndoor}
              onClick={() => setWeatherModalOpen(true)}
            />
          )}
        </div>
      </div>

      {/* modals */}
      <SnapshotModal
        gameId={gameId}
        sport={sport as Sport}
        isOpen={isSnapshotModalOpen}
        onClose={() => setSnapshotModalOpen(false)}
      />
      <WeatherModal
        isOpen={isWeatherModalOpen}
        onClose={() => setWeatherModalOpen(false)}
        weatherData={weatherData}
        isIndoor={isIndoor}
      />
    </div>
  );
};

export default GameCard;
