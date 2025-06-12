// frontend/src/components/games/snapshot_modal.jsx

import React, { useRef, useEffect, Suspense, lazy } from "react";
import PropTypes from "prop-types";
import { use_snapshot } from "../../hooks/use_snapshot"; // Corrected path
import HeadlineGrid from "./headline_grid"; // Corrected path
import SkeletonLoader from "../ui/skeleton_loader"; // Corrected path and directory

// Lazy load charting components for performance (Technical Design: Charts lazy-loaded)
// These will be implemented in a later step
const BarChartComponent = lazy(() =>
  import(/* webpackChunkName: "charts" */ "./charts/bar_chart_component")
); // Corrected path
const RadarChartComponent = lazy(() =>
  import(/* webpackChunkName: "charts" */ "./charts/radar_chart_component")
); // Corrected path
const PieChartComponent = lazy(() =>
  import(/* webpackChunkName: "charts" */ "./charts/pie_chart_component")
); // Corrected path

/**
 * SnapshotModal Component
 * Full-screen overlay displaying game snapshot data.
 * Handles data fetching, loading states, and integrates chart components.
 * Follows FR-SM-1, FR-SM-2, FR-SM-3, FR-SM-4.
 */
const SnapshotModal = ({ gameId, sport, isOpen, onClose }) => {
  const {
    data: snapshotData,
    isLoading,
    isError,
    error,
  } = use_snapshot(gameId, sport); // Corrected hook name
  const scrollRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === "Escape" && isOpen) {
        onClose();
      }
    };
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  if (isError) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
        <div className="bg-red-800 text-white p-6 rounded-lg shadow-xl max-w-sm mx-auto">
          <p className="font-bold text-lg mb-2">Error Loading Snapshot</p>
          <p>There was an error fetching the game data. Please try again.</p>
          {error && <p className="text-sm mt-2">{error.message}</p>}
          <button
            onClick={onClose}
            className="mt-4 px-4 py-2 bg-red-600 rounded-full text-sm font-semibold hover:bg-red-700"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm overflow-y-auto flex justify-center py-4"
      ref={scrollRef}
      role="dialog"
      aria-modal="true"
      aria-label={`${sport} Game Snapshot`}
    >
      <div className="relative w-full max-w-sm mx-auto bg-[var(--color-bg)] rounded-lg shadow-lg my-4 flex flex-col p-4">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-text-secondary hover:text-text-primary z-10 focus-ring"
          aria-label="Close snapshot"
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M6 18L18 6M6 6l12 12"
            ></path>
          </svg>
        </button>

        <h2 className="text-xl font-bold text-text-primary mb-4 text-center">
          {isLoading ? "Loading Snapshot..." : `${sport} Game Snapshot`}
        </h2>

        {/* Headline Stats (FR-SM-3) */}
        <section className="mb-6">
          <h3 className="text-lg font-semibold text-text-primary mb-2">
            Key Insights
          </h3>
          <HeadlineGrid
            headlines={snapshotData?.headline_stats}
            isLoading={isLoading}
          />
        </section>

        {/* Chart Sections (FR-SM-3) */}
        <Suspense
          fallback={
            <SkeletonLoader count={1} type="rect" className="h-48 my-4" />
          }
        >
          <section className="mb-6">
            <h3 className="text-lg font-semibold text-text-primary mb-2">
              Game Flow
            </h3>
            {/* Pass sport to BarChartComponent */}
            <BarChartComponent
              data={snapshotData?.bar_chart_data}
              sport={sport}
            />
          </section>

          <section className="mb-6">
            <h3 className="text-lg font-semibold text-text-primary mb-2">
              Team Strengths
            </h3>
            <RadarChartComponent data={snapshotData?.radar_chart_data} />
          </section>

          <section className="mb-6">
            <h3 className="text-lg font-semibold text-text-primary mb-2">
              Scoring Distribution
            </h3>
            <PieChartComponent data={snapshotData?.pie_chart_data} />
          </section>
        </Suspense>

        <div className="h-8"></div>
      </div>
    </div>
  );
};

SnapshotModal.propTypes = {
  gameId: PropTypes.string,
  sport: PropTypes.oneOf(["NBA", "MLB"]),
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
};

export default SnapshotModal;
