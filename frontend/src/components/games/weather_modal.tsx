// frontend/src/components/games/weather_modal.tsx

import React, { useEffect } from "react";
import { WeatherData } from "@/types";

interface WeatherModalProps {
  isOpen: boolean;
  onClose: () => void;
  weatherData: WeatherData | undefined | null;
}

const WeatherModal: React.FC<WeatherModalProps> = ({
  isOpen,
  onClose,
  weatherData,
}) => {
  // Effect to handle closing the modal with the Escape key
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
    }
    return () => document.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    // Backdrop
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose} // Close modal on backdrop click
      role="dialog"
      aria-modal="true"
    >
      {/* Modal Panel */}
      <div
        className="relative bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-sm p-6 text-gray-900 dark:text-gray-100"
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside the panel
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-3 right-3 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          aria-label="Close weather details"
        >
          <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            ></path>
          </svg>
        </button>

        {!weatherData ? (
          <div className="text-center">Loading weather details...</div>
        ) : (
          <div>
            <h2 className="text-2xl font-bold text-center mb-1">
              {weatherData.city}
            </h2>
            <p className="text-center text-gray-500 dark:text-gray-400 capitalize mb-4">
              {weatherData.description}
            </p>

            <div className="flex items-center justify-center my-4">
              <img
                src={`https://openweathermap.org/img/wn/${weatherData.icon}@2x.png`}
                alt={weatherData.description}
                className="w-20 h-20"
              />
              <p className="text-6xl font-bold">{weatherData.temperature}°F</p>
            </div>

            <div className="grid grid-cols-2 gap-4 text-center border-t border-gray-200 dark:border-gray-700 pt-4">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Feels Like
                </p>
                <p className="text-lg font-semibold">
                  {weatherData.feels_like}°F
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Humidity
                </p>
                <p className="text-lg font-semibold">{weatherData.humidity}%</p>
              </div>
              <div className="col-span-2">
                <p className="text-sm text-gray-500 dark:text-gray-400">Wind</p>
                <p className="text-lg font-semibold">
                  {weatherData.windSpeed} mph {weatherData.windDirection}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WeatherModal;
