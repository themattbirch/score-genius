// frontend/src/components/games/headline_grid.tsx

import React from "react";
import SkeletonLoader from "../ui/skeleton_loader";
import { useTheme } from "@/contexts/theme_context";
import { HeadlineStat } from "@/types"; // Import HeadlineStat type

// Define the interface for HeadlineGrid's props
interface HeadlineGridProps {
  headlines: HeadlineStat[]; // Use the imported HeadlineStat type
  isLoading?: boolean; // isLoading is optional based on its default usage
}

const HeadlineGrid: React.FC<HeadlineGridProps> = ({
  headlines = [],
  isLoading = false,
}) => {
  const { theme } = useTheme();

  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const textColorSecondary = theme === "dark" ? "#9ca3af" : "#475569";
  const panelBgColor = theme === "dark" ? "#161b22" : "#f8fafc";

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-4 p-4 rounded-lg bg-gray-700/50 my-4">
        {Array.from({ length: 6 }).map(
          (
            _,
            i // Assuming ~6 headlines
          ) => (
            <div key={i} className="flex flex-col">
              <SkeletonLoader count={1} type="text" className="w-3/4 mb-1" />
              <SkeletonLoader count={1} type="text" className="w-1/2" />
            </div>
          )
        )}
      </div>
    );
  }

  if (!headlines || headlines.length === 0) {
    return (
      <div className="text-center text-gray-500 p-4 my-4">
        No headline stats available.
      </div>
    );
  }

  return (
    <div
      className="grid grid-cols-2 gap-4 p-4 rounded-lg shadow-md my-4"
      style={{ backgroundColor: panelBgColor }}
    >
      {headlines.map((item, index) => (
        <div key={index} className="flex flex-col">
          <p className="text-sm mb-0.5" style={{ color: textColorSecondary }}>
            {item.label}
          </p>
          <p className="text-lg font-bold" style={{ color: textColorPrimary }}>
            {typeof item.value === "number"
              ? item.value.toLocaleString()
              : item.value}
          </p>
        </div>
      ))}
    </div>
  );
};

export default HeadlineGrid;
