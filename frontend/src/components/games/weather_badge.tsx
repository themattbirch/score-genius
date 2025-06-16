import React from "react";
import { WeatherData } from "@/types";
import RelativeWindIcon from "./relative_wind_icon";

interface WeatherBadgeProps {
  isLoading: boolean;
  isError: boolean;
  data: WeatherData | undefined;
  onClick: () => void;
}

const WeatherBadge: React.FC<WeatherBadgeProps> = ({
  isLoading,
  isError,
  data,
  onClick,
}) => {
  const renderContent = () => {
    if (isLoading) {
      return <span className="animate-pulse">Loading...</span>;
    }
    if (isError || !data) {
      return <span>--°F / --mph</span>;
    }
    return (
      <span>
        {data.temperature}°F / {data.windSpeed}mph
      </span>
    );
  };

  const tooltipText = data
    ? `${data.city}: ${data.description}, ${data.temperature}°F, Wind: ${data.windSpeed}mph (${data.ballparkWindText})`
    : "Click to view ballpark weather";

  return (
    <button
      onClick={onClick}
      title={tooltipText}
      aria-label={tooltipText}
      className={`
        rounded-full bg-badge-weather text-white text-xs font-semibold
        px-3 py-1 inline-flex items-center
        min-w-[120px] h-8 transition-all duration-200 ease-in-out
        hover:bg-sky-500 focus:outline-none focus:ring-2 focus:ring-offset-2 
        focus:ring-sky-500 focus:ring-offset-gray-800
      `}
    >
      {/* UPDATED: The container div for the icon now has more margin */}
      <div className="flex-shrink-0 flex items-center justify-center w-4 h-4 mr-2 bg-gray-700 rounded-full">
        <RelativeWindIcon rotation={data?.ballparkWindAngle ?? 0} />
      </div>

      <span className="truncate">{renderContent()}</span>
    </button>
  );
};

export default WeatherBadge;
