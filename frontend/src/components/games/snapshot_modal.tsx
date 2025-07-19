// frontend/src/components/games/snapshot_modal.tsx

/* ------------------------------------------------------------------
 * Snapshot Modal  |  shows per‑game snapshot for MLB / NBA
 * ------------------------------------------------------------------*/
import React, { useRef, useEffect, Suspense, lazy } from "react";
import { use_snapshot } from "@/hooks/use_snapshot";
import HeadlineGrid from "./headline_grid";
import SkeletonLoader from "../ui/skeleton_loader";
import { useTheme } from "@/contexts/theme_context";
import { Sport, SnapshotData, PieChartDataItem } from "@/types";

/* Lazy‑loaded charts */
const BarChartComponent = lazy(() => import("./charts/bar_chart_component"));
const RadarChartComponent = lazy(
  () => import("./charts/radar_chart_component")
);
const PieChartComponent = lazy(() => import("./charts/pie_chart_component"));

/* Props */
interface SnapshotModalProps {
  gameId: string;
  sport: Sport;
  isOpen: boolean;
  onClose: () => void;
}

/* Allow backend to return a simple exhibition message */
interface ExhibitionResponse {
  message: string;
}

type SnapshotResponse = SnapshotData | ExhibitionResponse;

const SnapshotModal: React.FC<SnapshotModalProps> = ({
  gameId,
  sport,
  isOpen,
  onClose,
}) => {
  /* data fetch */
  const {
    data: snapshotData,
    isLoading,
    isError,
    error,
    refetch,
  } = use_snapshot(gameId, sport) as {
    data: SnapshotResponse | undefined;
    isLoading: boolean;
    isError: boolean;
    error: Error | null;
    refetch: () => void;
  };

  /* guards & helpers */
  const isExhibition = !!(snapshotData && "message" in snapshotData);
  const sd = !isExhibition ? (snapshotData as SnapshotData) : undefined;

  const hasPie = (sd?.pie_chart_data?.length ?? 0) > 0;
  const hasKeyMetrics = (sd?.key_metrics_data?.length ?? 0) > 0;

  const barChartTitle =
    sport === "MLB" ? "Scoring Averages" : "Quarter Scoring";
  sport === "NFL" ? "Quarter Averages" : "Quarter Scoring";
  const pieChartTitle =
    sport === "MLB" ? "Avg Runs Vs LHP / RHP" : "Scoring Distribution";
  sport === "NFL" ? "Scoring Averages" : "Scoring Distribution";
  const isMultiPie =
    hasPie &&
    Array.isArray(sd?.pie_chart_data) &&
    (sd!.pie_chart_data as any[])[0]?.data !== undefined;

  /* theme helpers */
  const { theme } = useTheme();
  const textColor = theme === "dark" ? "#f1f5f9" : "#0d1117";
  const panelBg = theme === "dark" ? "#161b22" : "#f8fafc";

  /* effects */
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) refetch();
  }, [isOpen, refetch]);

  useEffect(() => {
    document.body.style.overflow = isOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && isOpen && onClose();
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, [isOpen, onClose]);

  /* early outs */
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

  /* ------------------------------------------------------------------
   * RENDER
   * ------------------------------------------------------------------*/
  return (
    <div
      ref={scrollRef}
      role="dialog"
      aria-modal="true"
      aria-label={`${sport} Game Snapshot`}
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm overflow-y-auto flex items-start justify-center py-4"
    >
      <div
        className="relative w-full max-w-lg mx-auto rounded-lg shadow-lg overflow-hidden flex flex-col p-2 space-y-8"
        style={{ backgroundColor: panelBg }}
      >
        {/* close btn */}
        <button
          onClick={onClose}
          aria-label="Close snapshot"
          className="absolute top-4 right-4 text-text-secondary hover:text-text-primary z-10 focus-ring"
        >
          ×
        </button>

        <h2
          className="text-xl font-bold mb-4 text-center"
          style={{ color: textColor }}
        >
          {isLoading ? "Loading Snapshot…" : `${sport} Game Snapshot`}
        </h2>

        {/* Exhibition message override */}
        {isExhibition && !isLoading ? (
          <div className="px-8 pb-12 text-center space-y-4">
            <p className="text-lg font-semibold" style={{ color: textColor }}>
              Exhibition Game Detected
            </p>
            <p className="text-muted-foreground">
              {(snapshotData as ExhibitionResponse).message}
            </p>
          </div>
        ) : (
          /* Regular snapshot content */
          <>
            {/* Headlines */}
            <section className="mb-6 px-8">
              <h3
                className="text-lg font-semibold mb-2 text-center"
                style={{ color: textColor }}
              >
                Key Insights
              </h3>
              <HeadlineGrid
                headlines={sd?.headline_stats || []}
                isLoading={isLoading}
              />
            </section>

            {/* All charts inside one Suspense */}
            <Suspense
              fallback={
                <SkeletonLoader count={1} type="rect" className="h-48 my-4" />
              }
            >
              {/* Bar Chart */}
              <section className="mb-6">
                <h3
                  className="text-lg font-semibold mb-2 text-center"
                  style={{ color: textColor }}
                >
                  {isLoading ? "Loading…" : barChartTitle}
                </h3>
                <BarChartComponent data={sd?.bar_chart_data} sport={sport} />
              </section>

              {/* Radar */}
              <section className="mb-6">
                <h3
                  className="text-lg font-semibold mb-2 text-center"
                  style={{ color: textColor }}
                >
                  Team Strengths
                </h3>
                <div className="-mx-4">
                  <RadarChartComponent data={sd?.radar_chart_data} />
                </div>
              </section>

              {/* Pie */}
              {hasPie && (
                <section className="mb-6 px-4">
                  <h3
                    className="text-lg font-semibold mb-4 text-center"
                    style={{ color: textColor }}
                  >
                    {pieChartTitle}
                  </h3>

                  {isMultiPie ? (
                    <div className="flex flex-wrap justify-center items-center gap-4 py-4">
                      {(sd!.pie_chart_data as any[]).map((p, i) => (
                        <div key={i} className="flex flex-col items-center">
                          <h4
                            className="mb-2 text-sm font-medium"
                            style={{ color: textColor }}
                          >
                            {p.title}
                          </h4>
                          <PieChartComponent
                            data={p.data as PieChartDataItem[]}
                            sport={sport}
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex justify-center py-4">
                      <PieChartComponent
                        data={sd!.pie_chart_data as PieChartDataItem[]}
                        sport={sport}
                      />
                    </div>
                  )}
                </section>
              )}

              {/* Key Metrics */}
              {hasKeyMetrics && (
                <section className="mb-8 px-4">
                  <h3
                    className="text-lg font-semibold mb-3 text-center"
                    style={{ color: textColor }}
                  >
                    Key Offensive Metrics (Per Game)
                  </h3>
                  <BarChartComponent
                    key={`${gameId}-keymetrics`}
                    data={sd!.key_metrics_data!}
                    sport={sport}
                  />
                </section>
              )}
            </Suspense>
          </>
        )}

        <div className="h-8" />
      </div>
    </div>
  );
};

export default SnapshotModal;
