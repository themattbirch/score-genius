// frontend/src/components/games/SnapshotButton.jsx

import React from "react";
import PropTypes from "prop-types";

/**
 * SnapshotButton Component
 * Displays a button to open the game snapshot modal.
 * Follows FR-GC-1 and UX/UI specifications.
 */
const SnapshotButton = ({
  onClick,
  isDisabled = false,
  tooltipText = "View Snapshot",
  className = "",
}) => {
  return (
    <button
      onClick={onClick}
      disabled={isDisabled}
      data-tour="snapshot-button"
      title={tooltipText}
      aria-label={tooltipText}
      className={`
        inline-flex w-fit items-center justify-center
        rounded-full bg-btn-snapshot text-white text-xs font-semibold
        px-4 py-1.5
        hover:bg-sky-500 focus:outline-none focus:ring-2 focus:ring-offset-2
        focus:ring-sky-500 focus:ring-offset-gray-800
        transition-opacity duration-200
        ${
          isDisabled
            ? "opacity-50 cursor-not-allowed"
            : "hover:opacity-90 active:opacity-75"
        }
        ${className}
        `}
    >
      Advanced Stats
    </button>
  );
};

SnapshotButton.propTypes = {
  onClick: PropTypes.func.isRequired,
  isDisabled: PropTypes.bool,
  tooltipText: PropTypes.string,
  className: PropTypes.string,
};

export default SnapshotButton;
