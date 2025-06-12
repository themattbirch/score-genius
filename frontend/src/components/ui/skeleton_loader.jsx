// frontend/src/components/ui/skeleton_loader.jsx

import React from "react";
import PropTypes from "prop-types";

/**
 * Generic Skeleton Loader Component.
 * @param {string} className - Additional Tailwind CSS classes for styling the skeleton container.
 * @param {number} count - Number of skeleton lines/blocks to render.
 * @param {'text' | 'rect' | 'circle'} type - Type of skeleton: 'text' (long line), 'rect' (block), 'circle'.
 */
const SkeletonLoader = ({ className = "", count = 1, type = "text" }) => {
  const baseClasses = "animate-pulse bg-gray-700 rounded"; // Adjust colors to match your theme
  let specificClasses = "";

  switch (type) {
    case "text":
      specificClasses = "h-4 w-full mb-2"; // Standard text line
      break;
    case "rect":
      specificClasses = "h-10 w-full mb-2"; // Larger block
      break;
    case "circle":
      specificClasses = "h-10 w-10 rounded-full mb-2"; // Circle, useful for icons/avatars
      break;
    default:
      specificClasses = "h-4 w-full mb-2";
  }

  return (
    <div className={className}>
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className={`${baseClasses} ${specificClasses}`}></div>
      ))}
    </div>
  );
};

SkeletonLoader.propTypes = {
  className: PropTypes.string,
  count: PropTypes.number,
  type: PropTypes.oneOf(["text", "rect", "circle"]),
};

export default SkeletonLoader;
