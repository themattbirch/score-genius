// frontend/src/components/games/weather_badge.tsx
import React, { MouseEventHandler, CSSProperties } from "react";
import type { WeatherData } from "@/types";

export interface WeatherBadgeProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  isLoading: boolean;
  isError: boolean;
  data?: WeatherData;
  isIndoor?: boolean;
}

const WeatherBadge: React.FC<WeatherBadgeProps> = ({
  className,
  isLoading,
  isError,
  data,
  isIndoor,
  onClick,
  ...rest // <-- 1. Collect other props
}) => {
  // 1) capture-phase guard
  const handleMouseDown: React.MouseEventHandler<HTMLButtonElement> = (e) => {
    console.log("▶️ WeatherBadge onMouseDown (capture)");
    e.stopPropagation();
  };

  // 2) bubbling-phase click
  const handleClick: React.MouseEventHandler<HTMLButtonElement> = (e) => {
    console.log("✅ WeatherBadge onClick (bubbling)");
    onClick?.(e);
  };

  const base = `quick-action-chip ${className ?? ""}`;
  const disabledCls = "opacity-60 cursor-not-allowed pointer-events-none";

  if (isIndoor) {
    return (
      <button
        type="button"
        onClick={onClick}
        data-tour="weather-badge"
        className={`${base}`}
        {...rest} // <-- 2. Apply them here
      >
        🏟️ Indoor
      </button>
    );
  }

  if (isLoading) {
    return (
      <button
        type="button"
        disabled
        data-tour="weather-badge"
        className={`${base} ${disabledCls}`}
        {...rest} // <-- 2. And here
      >
        Loading…
      </button>
    );
  }

  if (isError || !data) {
    return (
      <button
        type="button"
        disabled
        data-tour="weather-badge"
        className={`${base} ${disabledCls}`}
        {...rest} // <-- 2. And here
      >
        N/A
      </button>
    );
  }

  const { temperature, windSpeed, ballparkWindAngle } = data;
  const angle = typeof ballparkWindAngle === "number" ? ballparkWindAngle : 0;
  const arrowStyle: CSSProperties = { transform: `rotate(${angle}deg)` };

  return (
    <button
      type="button"
      onMouseDown={handleMouseDown}
      onClick={handleClick}
      data-tour="weather-badge"
      className={base}
      {...rest} // <-- 2. And finally, here
    >
      <span
        className="inline-block leading-none"
        style={arrowStyle}
        aria-hidden="true"
      >
        ↑
      </span>
      <span className="whitespace-nowrap text-xs font-semibold">
        {temperature}°F / {windSpeed}mph
      </span>
    </button>
  );
};

export default WeatherBadge;
