import React, { useState } from "react";
import { HelpCircle, Play, BarChart4 } from "lucide-react";
import { useTour } from "@/components/ui/joyride_tour";
import SnapshotModal from "@/components/games/snapshot_modal"; // Import the modal

const HowToUseScreen: React.FC = () => {
  const { start } = useTour();
  const [isSnapshotOpen, setIsSnapshotOpen] = useState(false); // State for the modal

  return (
    // Use a React Fragment to wrap the section and the modal
    <>
      <section className="mx-auto max-w-lg space-y-8 p-6 text-slate-800 dark:text-text-primary">
        {/* Title */}
        <header className="flex items-center gap-2">
          <HelpCircle
            size={24}
            strokeWidth={1.8}
            className="stroke-slate-800 dark:stroke-text-primary"
          />
          <h1 className="text-xl font-semibold text-slate-800 dark:text-text-primary">
            How to Use ScoreGenius
          </h1>
        </header>

        {/* Step list (no changes here) */}
        <ol className="space-y-8 border-l-2 border-brand-green dark:border-brand-green-light pl-4">
          <li>
            <h2 className="font-semibold text-slate-800 dark:text-brand-green mb-1">
              1. Pick your sport
            </h2>
            <p className="text-sm text-slate-500 dark:text-text-secondary leading-relaxed">
              Tap NBA or MLB in the header to switch between basketball and
              baseball views. See buttons as an example:{" "}
              <span className="pill pill-green mx-1">NBA</span>
              <span className="pill pill-green mx-1">MLB</span>
            </p>
          </li>
          <li>
            <h2 className="font-semibold text-slate-800 dark:text-brand-green mb-1">
              2. Browse today’s games
            </h2>
            <p className="text-sm text-slate-500 dark:text-text-secondary leading-relaxed">
              The <strong>Games</strong> tab lists scheduled matchups. Tap a
              game for detailed predictions, betting edges, and injury news.
            </p>
          </li>
          <li>
            <h2 className="font-semibold text-slate-800 dark:text-brand-green mb-1">
              3. Compare model vs. market
            </h2>
            <p className="text-sm text-slate-500 dark:text-text-secondary leading-relaxed">
              Green numbers signal positive value compared to Vegas odds; red
              means caution. Lines update in real time as odds shift.
            </p>
          </li>
          <li>
            <h2 className="font-semibold text-slate-800 dark:text-brand-green mb-1">
              4. Deep-dive stats
            </h2>
            <p className="text-sm text-slate-500 dark:text-text-secondary leading-relaxed">
              In the <strong>Stats</strong> tab, explore team rankings, advanced
              metrics, and sort any column header to re-order the table.
            </p>
          </li>
        </ol>

        {/* --- Action Buttons Container --- */}
        <div className="flex flex-col items-center gap-4 pt-4">
          {/* Launch Joyride Button */}
          <button
            onClick={start}
            className="flex w-full max-w-xs items-center justify-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-green-700"
          >
            <Play size={16} strokeWidth={2} />
            Start Interactive Tour
          </button>

          {/* --- Temporary Test Button --- */}
          <button
            onClick={() => setIsSnapshotOpen(true)}
            className="flex w-full max-w-xs items-center justify-center gap-2 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-sky-700"
          >
            <BarChart4 size={16} strokeWidth={2} />
            Test NFL Snapshot (Game 13406)
          </button>
        </div>
      </section>

      {/* --- Snapshot Modal Component --- */}
      <SnapshotModal
        isOpen={isSnapshotOpen}
        onClose={() => setIsSnapshotOpen(false)}
        gameId="13406"
        sport="NFL"
      />
    </>
  );
};

export default HowToUseScreen;
