// frontend/src/components/games/SnapshotButton.jsx

import React from "react";
import PropTypes from "prop-types"; // For type checking props

/**
 * SnapshotButton Component
 * Displays a button to open the game snapshot modal.
 * Follows FR-GC-1 and UX/UI specifications.
 */
const SnapshotButton = ({
  onClick,
  isDisabled = false,
  tooltipText = "View Snapshot",
}) => {
  return (
    <button
      onClick={onClick}
      disabled={isDisabled}
      data-tour="snapshot-button"
      className={`
        rounded-full // FR-GC-1, UX/UI
        bg-btn-snapshot // UX/UI custom color
        text-white // Assuming white text on button for contrast
        text-xs // UX/UI
        font-semibold // UX/UI
        px-3 py-2 // Padding for a good button size
        inline-flex items-center justify-center // For centering text
        hover:bg-sky-500 focus:outline-none focus:ring-2 focus:ring-offset-2 
        focus:ring-sky-500 focus:ring-offset-gray-800
        transition-opacity duration-200 // For hover effect
        ${
          isDisabled
            ? "opacity-50 cursor-not-allowed"
            : "hover:opacity-90 active:opacity-75"
        } // UX/UI hover:opacity-90, disabled state
      `}
      title={tooltipText} // FR-R-2 (Missing snapshot JSON mitigation)
      aria-label={tooltipText}
    >
      Advanced Stats
    </button>
  );
};

SnapshotButton.propTypes = {
  onClick: PropTypes.func.isRequired,
  isDisabled: PropTypes.bool,
  tooltipText: PropTypes.string,
};

export default SnapshotButton;
