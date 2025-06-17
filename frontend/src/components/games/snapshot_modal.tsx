// frontend/src/components/games/snapshot_modal.tsx

import React, { useRef, useEffect, Suspense, lazy } from "react";
import { use_snapshot } from "../../hooks/use_snapshot";
import HeadlineGrid from "./headline_grid";
import SkeletonLoader from "../ui/skeleton_loader";
import { useTheme } from "@/contexts/theme_context";
// Import all necessary types from your types file
import {
  Sport,
  SnapshotData,
  HeadlineStat,
  NbaPreGameOffenseDataItem,
  PieChartDataItem,
  BarChartData,
  RadarChartData,
} from "@/types";

interface SnapshotModalProps {
  gameId: string;
  sport: Sport;
  isOpen: boolean;
  onClose: () => void;
}

// SIMPLIFIED LAZY-LOADED COMPONENTS (assuming each chart component uses 'export default MyComponent;')
// With "esModuleInterop": true, TypeScript should correctly infer the types without explicit 'as React.ComponentType'
const BarChartComponent = lazy(() => import("./charts/bar_chart_component"));
const RadarChartComponent = lazy(
  () => import("./charts/radar_chart_component")
);
const PieChartComponent = lazy(() => import("./charts/pie_chart_component"));
const NbaPreGameOffenseChart = lazy(
  () => import("./charts/nba_pre_game_offense_chart")
);

const SnapshotModal: React.FC<SnapshotModalProps> = ({
  gameId,
  sport,
  isOpen,
  onClose,
}) => {
  // Type the return of use_snapshot
  const {
    data: snapshotData, // Type 'data' as SnapshotData
    isLoading,
    isError,
    error,
    refetch,
  } = use_snapshot(gameId, sport) as {
    // Add type assertion here
    data: SnapshotData | undefined; // data can be undefined during loading/error
    isLoading: boolean;
    isError: boolean;
    error: Error | null;
    refetch: () => void;
  };

  const scrollRef = useRef<HTMLDivElement>(null); // Type useRef for better DOM inference

  const { theme } = useTheme();

  const textColorPrimary = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const textColorSecondary = theme === "dark" ? "#9ca3af" : "#475569";
  const panelBgColor = theme === "dark" ? "#161b22" : "#f8fafc";


  useEffect(() => {
    if (isOpen) {
      refetch();
    }
  }, [isOpen, refetch]);

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
    // Explicitly type the event parameter
    const handleEscape = (event: KeyboardEvent) => {
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

  const isNbaPreGameOffenseData =
    sport === "NBA" &&
    snapshotData?.pie_chart_data &&
    // Cast snapshotData.pie_chart_data to the specific type before .some()
    (snapshotData.pie_chart_data as NbaPreGameOffenseDataItem[]).some(
      // Type 'd' within the callback
      (d: NbaPreGameOffenseDataItem) =>
        "metric" in d && "Home" in d && "Away" in d
    );

  const pieChartSectionTitle = isNbaPreGameOffenseData
    ? "Key Offensive Metrics"
    : "Scoring Distribution";

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm overflow-y-auto fle items-start justify-center py-4"
      ref={scrollRef}
      role="dialog"
      aria-modal="true"
      aria-label={`${sport} Game Snapshot`}
    >
      <div
        className="relative w-full max-w-sm mx-auto rounded-lg shadow-lg overflow-hidden flex flex-col p-4 space-y-8"
        style={{ backgroundColor: panelBgColor }}
      >
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

        <h2
          className="text-xl font-bold mb-4 text-center"
          style={{ color: textColorPrimary }}
        >
          {isLoading ? "Loading Snapshot..." : `${sport} Game Snapshot`}
        </h2>

        <section className="mb-6">
          <h3
            className="text-lg font-semibold mb-2"
            style={{ color: textColorPrimary }}
          >
            Key Insights
          </h3>
          <HeadlineGrid
            headlines={snapshotData?.headline_stats || []} // Provide empty array if undefined
            isLoading={isLoading}
          />
        </section>

        <Suspense
          fallback={
            <SkeletonLoader count={1} type="rect" className="h-48 my-4" />
          }
        >
          <section className="mb-6">
            <h3
              className="text-lg font-semibold mb-2 text-center"
              style={{ color: textColorPrimary }}
            >
              Quarter Scoring Averages
            </h3>
            <BarChartComponent
              data={snapshotData?.bar_chart_data}
              sport={sport}
            />
          </section>

          <section className="mb-6">
            <h3
              className="text-lg font-semibold mb-2 text-center"
              style={{ color: textColorPrimary }}
            >
              Team Strengths
            </h3>
            <div className="-mx-4">
              <RadarChartComponent data={snapshotData?.radar_chart_data} />
            </div>
          </section>

          <section className="mb-6">
            <h3
              className="text-lg font-semibold mb-2 text-center"
              style={{ color: textColorPrimary }}
            >
              {pieChartSectionTitle}
            </h3>
            {isNbaPreGameOffenseData ? (
              <NbaPreGameOffenseChart
                data={
                  snapshotData?.pie_chart_data as NbaPreGameOffenseDataItem[]
                }
              />
            ) : (
              <PieChartComponent
                data={snapshotData?.pie_chart_data as PieChartDataItem[]}
              />
            )}
          </section>
        </Suspense>

        <div className="h-8"></div>
      </div>
    </div>
  );
};

export default SnapshotModal;
