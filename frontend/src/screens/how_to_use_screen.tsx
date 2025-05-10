// frontend/src/screens/how_to_use_screen.tsx
import React from "react";
import { HelpCircle, Play } from "lucide-react"; // Ensure Play is imported if you kept it
import { useTour } from "@/components/ui/joyride_tour";

const HowToUseScreen: React.FC = () => {
  const { start } = useTour();

  return (
    // --- Point 2: Increased vertical spacing between header, list and button ---
    <section className="mx-auto max-w-lg space-y-8 p-6 text-text-primary">
      {" "}
      {/* Title */}
      <header className="flex items-center gap-2">
        {/* --- Point 3: Explicit light/dark colors for icon --- */}
        <HelpCircle
          size={24}
          strokeWidth={1.8}
          className="stroke-slate-800 dark:stroke-slate-200" // Example explicit colors
        />
        {/* --- Point 3: Explicit light/dark colors for heading --- */}
        <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-100">
          {" "}
          {/* Explicit colors */}
          How to Use Score Genius
        </h1>
      </header>
      {/* Step list */}
      {/* --- Point 2: (Covered by parent space-y-8) --- */}
      <ol className="space-y-8 border-l-2 border-brand-green pl-4">
        <li>
          <h2 className="font-semibold text-slate-900 dark:text-brand-green mb-1">
            1. Pick your sport
          </h2>
          {/* Pills inline after "example:" */}
          <p className="text-sm text-gray-700 dark:text-slate-400 leading-relaxed">
            Tap NBA or MLB in the header to switch between basketball and
            baseball views. See buttons as an example:{" "}
            {/* Added space before pills */}
            <span className="pill pill-green mx-1">NBA</span>
            <span className="pill pill-green mx-1">MLB</span>
          </p>
        </li>

        <li>
          {/* --- Point 3 & 1 repeated --- */}
          <h2 className="font-semibold text-slate-900 dark:text-brand-green mb-1">
            2. Browse today’s games
          </h2>
          <p className="text-sm text-gray-700 dark:text-slate-400 leading-relaxed">
            The <strong>Games</strong> tab lists scheduled matchups. Tap a game
            for detailed predictions, betting edges, and injury news.
          </p>
        </li>

        <li>
          {/* --- Point 3 & 1 repeated --- */}
          <h2 className="font-semibold text-slate-900 dark:text-brand-green mb-1">
            3. Compare model vs. market
          </h2>
          <p className="text-sm text-gray-700 dark:text-slate-400 leading-relaxed">
            Green numbers signal positive value compared to Vegas odds; red
            means caution. Lines update in real time as odds shift.
          </p>
        </li>

        <li>
          {/* --- Point 3 & 1 repeated --- */}
          <h2 className="font-semibold text-slate-900 dark:text-brand-green mb-1">
            4. Deep-dive stats
          </h2>
          <p className="text-sm text-gray-700 dark:text-slate-400 leading-relaxed">
            In the <strong>Stats</strong> tab, explore team rankings, advanced
            metrics, and sort any column header to re-order the table.
          </p>
        </li>
      </ol>
      {/* Launch Joyride Button */}
      {/* --- Point 2: (Covered by parent space-y-8, plus its own mt-8) --- */}
      <button
        onClick={start}
        className="mx-auto mt-8 flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-green-700"
      >
        <Play size={16} strokeWidth={2} />
        Start Interactive Tour
      </button>
    </section>
  );
};

export default HowToUseScreen;
