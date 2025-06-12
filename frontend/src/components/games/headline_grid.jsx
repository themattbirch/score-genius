// frontend/src/components/games/headline_grid.jsx

import React from "react";
import PropTypes from "prop-types";
import SkeletonLoader from "../ui/skeleton_loader";
/**
 * HeadlineGrid Component
 * Displays key statistical headlines in a grid format.
 * @param {Array<Object>} headlines - Array of headline objects: [{ label: string, value: any }]
 * @param {boolean} isLoading - Whether the data is currently loading.
 */
const HeadlineGrid = ({ headlines, isLoading }) => {
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
    <div className="grid grid-cols-2 gap-4 p-4 rounded-lg bg-[var(--color-panel)] shadow-md my-4">
      {headlines.map((item, index) => (
        <div key={index} className="flex flex-col">
          <p className="text-text-secondary text-sm mb-0.5">{item.label}</p>
          <p className="text-text-primary text-lg font-bold">
            {typeof item.value === "number"
              ? item.value.toLocaleString()
              : item.value}
          </p>
        </div>
      ))}
    </div>
  );
};

HeadlineGrid.propTypes = {
  headlines: PropTypes.arrayOf(
    PropTypes.shape({
      label: PropTypes.string.isRequired,
      value: PropTypes.oneOfType([PropTypes.string, PropTypes.number])
        .isRequired,
    })
  ),
  isLoading: PropTypes.bool,
};

HeadlineGrid.defaultProps = {
  headlines: [],
  isLoading: false,
};

export default HeadlineGrid;
