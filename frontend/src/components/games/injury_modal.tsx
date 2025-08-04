// frontend/src/components/games/injury_modal.tsx
import React, { useMemo, Suspense, lazy, useRef, useEffect } from "react";
import type { Injury } from "@/api/use_injuries";
import { useInjuries } from "@/api/use_injuries";

const InjuryReport = lazy(() => import("@/components/shared/injury_report"));

interface InjuryModalProps {
  isOpen: boolean;
  onClose: () => void;
  league: string;
  gameDate: string;
  teamNames: [string, string];
}

const InjuryModal: React.FC<InjuryModalProps> = ({
  isOpen,
  onClose,
  league,
  gameDate,
  teamNames,
}) => {
  /* ─────────────── hooks (ALWAYS RUN) ─────────────── */
  const backdropRef = useRef<HTMLDivElement | null>(null);

  // reset scroll so the panel starts at the top whenever it opens
  useEffect(() => {
    if (isOpen && backdropRef.current) backdropRef.current.scrollTop = 0;
  }, [isOpen]);

  /* ─────────────── data ─────────────── */
  const {
    data: allInjuries,
    isLoading: isLoadingInjuries,
    error: injuriesError,
  } = useInjuries(league, gameDate, { enabled: isOpen });

  const { teamsWithInjuries, injuriesByTeam } = useMemo(() => {
    if (!allInjuries) return { teamsWithInjuries: [], injuriesByTeam: {} };

    const grouped = allInjuries
      .filter((inj) => teamNames.includes(inj.team_display_name))
      .reduce<Record<string, Injury[]>>((acc, inj) => {
        (acc[inj.team_display_name] ??= []).push(inj);
        return acc;
      }, {});

    return { teamsWithInjuries: Object.keys(grouped), injuriesByTeam: grouped };
  }, [allInjuries, teamNames]);

  const displayDate = useMemo(() => {
    const d = new Date(`${gameDate}T12:00:00Z`);
    return d.toLocaleDateString([], { month: "long", day: "numeric" });
  }, [gameDate]);

  /* ─────────────── early exit (after hooks) ─────────────── */
  if (!isOpen) return null;

  /* ─────────────── UI ─────────────── */
  return (
    <div
      ref={backdropRef}
      className="fixed inset-0 z-40 flex items-start justify-center
                 pt-12 px-4 overflow-auto bg-black/60 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="injury-modal-title"
      onClick={onClose}
    >
      <div
        className="z-50 w-full max-w-lg rounded-2xl border border-border bg-panel
                   shadow-lg ring-1 ring-white/10 flex flex-col
                   min-h-[50vh] lg:min-h-[60vh] max-h-[90vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <h2 id="injury-modal-title" className="text-lg font-semibold">
            {league.toUpperCase()} Injury Report — {displayDate}
          </h2>
          <button
            aria-label="Close modal"
            onClick={onClose}
            className="p-2 rounded hover:bg-white/10"
          >
            ✕
          </button>
        </div>

        {/* body */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <Suspense fallback={<p className="italic opacity-75">Loading…</p>}>
            <InjuryReport
              displayDate={displayDate}
              isPastDate={
                new Date(`${gameDate}T00:00:00`) <
                new Date(new Date().setHours(0, 0, 0, 0))
              }
              allGamesFilteredOut={false}
              isLoadingInjuries={isLoadingInjuries}
              injuriesError={injuriesError as Error | undefined}
              teamsWithInjuries={teamsWithInjuries}
              injuriesByTeam={injuriesByTeam}
            />
          </Suspense>
        </div>

        {/* footer */}
        <div className="px-6 py-3 border-t border-white/10">
          <button
            onClick={onClose}
            className="w-full rounded bg-green-600 py-2 font-medium hover:bg-green-500"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};

export default InjuryModal;
