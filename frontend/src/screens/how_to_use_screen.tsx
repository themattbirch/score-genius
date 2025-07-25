// frontend/src/screens/how_to_use_screen.tsx
import React, { useState } from "react";
import { HelpCircle, Play, ArrowRight } from "lucide-react";
import { useTour } from "@/components/ui/joyride_tour";
import SnapshotModal from "@/components/games/snapshot_modal";

/**
 * Simple “ghost” pill button for inline examples
 */
const GhostPill: React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement> & { children: React.ReactNode }
> = ({ children, className = "", ...props }) => (
  <button
    {...props}
    className={`
      inline-flex items-center justify-center rounded-2xl border px-4 py-2 text-sm font-semibold
      border-slate-300 text-slate-600 bg-white/60 hover:bg-white
      dark:border-slate-600/60 dark:text-slate-300 dark:bg-slate-800/60 dark:hover:bg-slate-800
      transition-colors focus-ring
      ${className}
    `}
  >
    {children}
  </button>
);

const HowToUseScreen: React.FC = () => {
  const { start } = useTour();
  const [isSnapshotOpen, setIsSnapshotOpen] = useState(false);

  return (
    <>
      <section
        className={`
          mx-auto max-w-lg p-6 md:px-8 lg:px-10
          text-slate-800 dark:text-text-primary
          pb-[env(safe-area-inset-bottom)] space-y-10
        `}
      >
        {/* Title */}
        <header className="flex items-center gap-2">
          <HelpCircle
            size={26}
            strokeWidth={1.8}
            className="stroke-slate-800 dark:stroke-text-primary"
          />
          <h1 className="text-2xl md:text-3xl font-semibold">
            How to Use ScoreGenius
          </h1>
        </header>

        <ol
          className={`
            relative space-y-12 pl-8 sm:pl-9
            before:absolute before:top-0 before:bottom-0 before:left-4
            before:w-px before:bg-slate-200 dark:before:bg-slate-700/60
          `}
        >
          {/* 1 */}
          <li className="relative flex items-start gap-4">
            <span className="howto-step-num mt-1 inline-flex h-9 w-9 items-center justify-center rounded-full bg-brand-green text-white font-bold motion-safe:animate-pulse">
              1
            </span>
            <div className="flex-1 space-y-3 leading-relaxed">
              <h2 className="text-brand-green font-semibold text-xl">
                Pick your sport
              </h2>
              <p className="text-base text-slate-600 dark:text-text-secondary">
                Tap <span className="pill pill-green mx-1">NBA</span> or{" "}
                <span className="pill pill-green mx-1">MLB</span> in the header
                to switch views.
              </p>
            </div>
          </li>

          {/* 2 */}
          <li className="relative flex items-start gap-4">
            <span className="howto-step-num mt-1 inline-flex h-9 w-9 items-center justify-center rounded-full bg-brand-green text-white font-bold">
              2
            </span>
            <div className="flex-1 space-y-3 leading-relaxed">
              <h2 className="text-brand-green font-semibold text-xl">
                Browse games &amp; expand
              </h2>
              <p className="text-base text-slate-600 dark:text-text-secondary">
                In the <strong>Games</strong> tab you’ll see today’s matchups.
                Use the calendar to change dates.
              </p>
              <p className="text-base text-slate-600 dark:text-text-secondary">
                <span className="inline-flex items-center gap-2 whitespace-nowrap">
                  Click the
                  <span className="card-chevron ring-1 ring-slate-300/60 dark:ring-slate-600/60 motion-safe:animate-pulse">
                    ▾
                  </span>
                  on any game card to expand it.
                </span>
              </p>
            </div>
          </li>

          {/* 3 */}
          <li className="relative flex items-start gap-4">
            <span className="howto-step-num mt-1 inline-flex h-9 w-9 items-center justify-center rounded-full bg-brand-green text-white font-bold">
              3
            </span>
            <div className="flex-1 space-y-4 leading-relaxed">
              <h2 className="text-brand-green font-semibold text-xl">
                H2H Stats &amp; Weather
              </h2>
              <p className="text-base text-slate-600 dark:text-text-secondary">
                After expanding, click{" "}
                <GhostPill
                  type="button"
                  aria-label="H2H Stats example"
                  onClick={(e) => e.preventDefault()}
                  className="mx-1"
                >
                  H2H Stats
                </GhostPill>{" "}
                for head‑to‑head analysis or{" "}
                <GhostPill
                  type="button"
                  aria-label="Weather example"
                  onClick={(e) => e.preventDefault()}
                  className="mx-1"
                >
                  Weather
                </GhostPill>{" "}
                for ballpark forecast.
              </p>
              <div>
                <GhostPill
                  type="button"
                  aria-label="Snapshot demo"
                  onClick={() => setIsSnapshotOpen(true)}
                  className="mt-2 border-l-2 border-brand-green/70 pl-5 pr-4 py-2 inline-flex gap-2 text-base font-medium"
                >
                  View snapshot demo <ArrowRight size={18} strokeWidth={2} />
                </GhostPill>
              </div>
            </div>
          </li>

          {/* 4 */}
          <li className="relative flex items-start gap-4">
            <span className="howto-step-num mt-1 inline-flex h-9 w-9 items-center justify-center rounded-full bg-brand-green text-white font-bold">
              4
            </span>
            <div className="flex-1 space-y-3 leading-relaxed">
              <h2 className="text-brand-green font-semibold text-xl">
                Explore advanced stats
              </h2>
              <p className="text-base text-slate-600 dark:text-text-secondary">
                Switch to the <strong>Stats</strong> tab for team and player
                rankings, plus advanced metrics. Click any column header to
                sort.
              </p>
            </div>
          </li>
        </ol>

        {/* CTA */}
        <div className="pt-2">
          <button
            onClick={start}
            className="flex w-full items-center justify-center gap-2 rounded-2xl bg-brand-green px-6 py-4 text-base font-semibold text-white shadow-sm hover:bg-brand-green/90 focus-ring transition-colors"
          >
            <Play size={18} strokeWidth={2.2} />
            Start Interactive Tour
          </button>
        </div>
      </section>

      <SnapshotModal
        isOpen={isSnapshotOpen}
        onClose={() => setIsSnapshotOpen(false)}
        gameId="164864"
        sport="MLB"
      />
    </>
  );
};

export default HowToUseScreen;
