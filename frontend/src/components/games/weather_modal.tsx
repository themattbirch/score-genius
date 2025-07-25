// frontend/src/components/games/weather_modal.tsx

import React from "react";
import { createPortal } from "react-dom";
import type { WeatherData } from "@/types";

export interface WeatherModalProps {
  isOpen: boolean;
  onClose: () => void;
  weatherData: WeatherData | undefined;
  isIndoor?: boolean;
}

/* A very thin modal; replace with your own UI lib if you use one */
const WeatherModal: React.FC<WeatherModalProps> = ({
  isOpen,
  onClose,
  weatherData,
  isIndoor,
}) => {
  if (!isOpen) return null;

  const city = weatherData?.city ?? "—";
  const stadiumName = (weatherData as any)?.stadium ?? "Indoor Venue";

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-slate-800 text-white rounded-lg w-80 p-6 space-y-4">
        {/* header */}
        <div className="flex justify-center items-center relative">
          <h2 className="text-xl font-bold text-center w-full">{city}</h2>
          <button
            onClick={onClose}
            className="absolute right-0 top-0 mt-1 mr-1 text-lg font-bold"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* ─── indoor venue copy ───────────────────────────────────── */}
        {isIndoor ? (
          <div className="space-y-2 text-center">
            <p className="text-lg font-medium">{stadiumName}</p>
            <p className="text-sm text-slate-400">
              Game is played indoors&nbsp;(dome).
              <br />
              No weather data available.
            </p>
          </div>
        ) : (
          /* ─── outdoor: existing layout ─────────────────────────── */
          <>
            <div className="flex items-center justify-center space-x-2">
              {weatherData?.icon && (
                <img
                  src={`https://openweathermap.org/img/wn/${weatherData.icon}@2x.png`}
                  alt={weatherData.description}
                  className="w-10 h-10"
                />
              )}
              <p className="text-3xl font-semibold">
                {weatherData?.temperature ?? "–"}°F
              </p>
            </div>

            <div className="border-t border-slate-700 pt-4 grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-xs text-slate-400">Feels Like</p>
                <p>{weatherData?.feels_like ?? "–"}°F</p>
              </div>
              <div>
                <p className="text-xs text-slate-400">Humidity</p>
                <p>{weatherData?.humidity ?? "–"}%</p>
              </div>
              <div className="col-span-2">
                <p className="text-xs text-slate-400">Wind</p>
                <p className="flex items-center justify-center space-x-1">
                  <span
                    className="inline-block rotate-[-45deg]"
                    style={{
                      transform: `rotate(${
                        weatherData?.ballparkWindAngle ?? 0
                      }deg)`,
                    }}
                  >
                    ↑
                  </span>
                  <span>
                    {weatherData?.windSpeed ?? 0} mph&nbsp; (
                    {weatherData?.ballparkWindText})
                  </span>
                </p>
              </div>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body
  );
};

export default WeatherModal;
