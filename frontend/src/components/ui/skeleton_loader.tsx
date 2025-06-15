// frontend/src/components/ui/skeleton_loader.tsx

import React from "react";
// PropTypes is no longer needed once using TypeScript interfaces for props
// import PropTypes from "prop-types";

// Define the interface for SkeletonLoader's props
interface SkeletonLoaderProps {
  className?: string; // Additional Tailwind CSS classes for styling the skeleton container.
  count?: number; // Number of skeleton lines/blocks to render.
  type?: "text" | "rect" | "circle"; // Type of skeleton: 'text' (long line), 'rect' (block), 'circle'.
}

/**
 * Generic Skeleton Loader Component.
 * @param {string} className - Additional Tailwind CSS classes for styling the skeleton container.
 * @param {number} count - Number of skeleton lines/blocks to render.
 * @param {'text' | 'rect' | 'circle'} type - Type of skeleton: 'text' (long line), 'rect' (block), 'circle'.
 */
// Apply the interface to the functional component
const SkeletonLoader: React.FC<SkeletonLoaderProps> = ({
  className = "",
  count = 1,
  type = "text",
}) => {
  // Using 'bg-gray-700' for dark mode and 'bg-gray-300' for light mode for the base skeleton color.
  // The 'dark:' prefix handles the theme switching for this specific background.
  const baseClasses = "animate-pulse bg-gray-300 dark:bg-gray-700 rounded";
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
    default: // Fallback to text type if an invalid type is somehow passed
      specificClasses = "h-4 w-full mb-2";
  }

  return (
    // The provided `className` is for the outer container of the SkeletonLoader,
    // which makes sense for applying overall layout or width.
    <div className={className}>
      {Array.from({ length: count }).map((_, index) => (
        // The actual skeleton element uses baseClasses and specificClasses
        <div key={index} className={`${baseClasses} ${specificClasses}`}></div>
      ))}
    </div>
  );
};

// Remove PropTypes as we are using TypeScript interfaces for type checking
// SkeletonLoader.propTypes = {
//   className: PropTypes.string,
//   count: PropTypes.number,
//   type: PropTypes.oneOf(["text", "rect", "circle"]),
// };

export default SkeletonLoader;