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

  /* ------------------------------------------------------------------ */
  /*  Modal state                                                       */
  /* ------------------------------------------------------------------ */
  const [isSnapshotModalOpen, setIsSnapshotModalOpen] = useState(false);
  const [isWeatherModalOpen, setWeatherModalOpen] = useState(false);

  /* ------------------------------------------------------------------ */
  /*  Weather hook (MLB + NFL only)                                     */
  /* ------------------------------------------------------------------ */
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

  /* ------------------------------------------------------------------ */
  /*  Render                                                            */
  /* ------------------------------------------------------------------ */
  return (
    <div className="app-card flex flex-col gap-4" data-tour="game-card">
      <div className="flex items-start justify-between gap-4">
        {/* ----- Teams & snapshot button -------------------------------- */}
        <div className="min-w-0 flex-1 max-w-md flex flex-col space-y-2">
          <p className="font-semibold text-sm sm:text-base leading-tight">
            {awayTeamName}
          </p>
          <p className="font-semibold text-sm sm:text-base leading-tight">
            @ {homeTeamName}
          </p>
          <p className="text-xs text-text-secondary pt-1">
            {gameTimeUTC
              ? new Date(gameTimeUTC).toLocaleTimeString([], {
                  hour: "numeric",
                  minute: "2-digit",
                })
              : game_date}
            {statusState &&
              !["final", "sched", "pre"].some((s) =>
                statusState.toLowerCase().includes(s)
              ) &&
              ` (${statusState})`}
          </p>
          <div className="mb-1">
            <SnapshotButton onClick={() => setIsSnapshotModalOpen(true)} />
          </div>
        </div>

        {/* ----- Score / predictions / weather badge -------------------- */}
        <div className="w-36 md:w-auto text-right text-sm">
          {dataType === "historical" ? (
            <p className="font-semibold text-lg w-full">
              {away_final_score ?? "-"} – {home_final_score ?? "-"}
              <span className="block text-xs font-normal text-text-secondary">
                (Final)
              </span>
            </p>
          ) : dataType === "schedule" ? (
            /* ---- NBA predicted points -------------------------------- */
            contextSport === "NBA" ? (
              <p className="font-medium text-green-600 dark:text-green-500">
                {predictionAway?.toFixed(1) ?? "-"} –{" "}
                {predictionHome?.toFixed(1) ?? "-"}
                <span className="block text-xs font-normal text-text-secondary">
                  (Predicted Score)
                </span>
              </p>
            ) : (
              /* ---- MLB or NFL section -------------------------------- */
              <div>
                {/* MLB predicted runs */}
                {sport === "MLB" &&
                predicted_home_runs != null &&
                predicted_away_runs != null ? (
                  <p className="font-medium text-green-600 dark:text-green-500">
                    {predicted_away_runs.toFixed(1)} –{" "}
                    {predicted_home_runs.toFixed(1)}
                    <span className="block text-xs font-normal text-text-secondary">
                      (Predicted Score)
                    </span>
                  </p>
                ) : (
                  <p className="font-medium text-text-secondary">-</p>
                )}

                {/* Pitchers (MLB only) */}
                {sport === "MLB" && (
                  <div className="mt-1">
                    <p className="text-xs font-normal text-text-secondary">
                      {awayPitcher ?? "TBD"}{" "}
                      {awayPitcherHand && `(${awayPitcherHand})`}
                    </p>
                    <p className="text-xs font-normal text-text-secondary">
                      {homePitcher ?? "TBD"}{" "}
                      {homePitcherHand && `(${homePitcherHand})`}
                    </p>
                  </div>
                )}

                {/* Weather badge / indoor button */}
                {supportsWeather && (
                  <div className="mt-2 flex justify-end">
                    <WeatherBadge
                      isLoading={isWeatherLoading}
                      isError={isWeatherError}
                      data={weatherData}
                      isIndoor={isIndoor}
                      onClick={() => setWeatherModalOpen(true)}
                    />
                  </div>
                )}
              </div>
            )
          ) : (
            <p className="font-medium w-full">—</p>
          )}
        </div>
      </div>

      {/* ------------------ Modals ------------------------------------- */}
      <SnapshotModal
        gameId={gameId}
        sport={sport as Sport}
        isOpen={isSnapshotModalOpen}
        onClose={() => setIsSnapshotModalOpen(false)}
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
