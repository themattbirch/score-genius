import React from "react";

interface RelativeWindIconProps {
  rotation: number;
}

const RelativeWindIcon: React.FC<RelativeWindIconProps> = ({ rotation }) => {
  return (
    <div className="w-6 h-6 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
      {/* Base state (0 degrees rotation) is an arrow pointing UP. */}
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="text-gray-800 dark:text-gray-100 transition-transform duration-500 ease-in-out"
        style={{
          transform: `rotate(${rotation}deg)`,
          transformOrigin: "center",
        }}
      >
        <path
          d="M12 19V5M12 5L6 11M12 5L18 11" // This SVG path draws an arrow pointing UP
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
};

export default RelativeWindIcon;
