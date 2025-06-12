// frontend/src/components/games/WeatherBadge.jsx

import React from "react";
import PropTypes from "prop-types"; // For type checking props

// Assuming you have an icon for weather, e.g., a simple cloud/sun icon
// For this example, we'll use a placeholder text icon.
// In a real app, you might import an SVG: import { SunIcon } from '@heroicons/react/24/solid'
const WeatherIcon = () => (
  <span role="img" aria-label="weather icon" className="mr-1 text-base">
    ☀️
  </span>
);

/**
 * WeatherBadge Component (Stub)
 * Displays a static weather placeholder for MLB games.
 * Follows FR-GC-2, FR-WB-1, FR-WB-2, FR-WB-3 and UX/UI specifications.
 */
const WeatherBadge = () => {
  const tooltipText = "Ballpark weather coming soon"; // FR-WB-2

  return (
    <div
      className={`
        rounded-full // UX/UI
        bg-badge-weather // UX/UI custom color
        text-white // Assuming white text on badge
        text-xs // Consistent with SnapshotButton
        font-semibold
        px-2 py-1 // Padding
        inline-flex items-center justify-center // For centering content
        min-w-[90px] // Ensure it has a consistent minimum width
        h-8 // UX/UI specified 32px height
      `}
      title={tooltipText} // FR-WB-2
      aria-label={tooltipText}
    >
      <WeatherIcon /> {/* FR-WB-1: Shows "WX" icon */}
      <span className="text-sm">— °F / — mph</span>{" "}
      {/* FR-WB-1: Shows "— °F / — mph" */}
    </div>
  );
};

WeatherBadge.propTypes = {}; // No props for a static stub

export default WeatherBadge;
