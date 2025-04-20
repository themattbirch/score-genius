import React from 'react';
import { HelpCircle } from 'lucide-react';

const HowToUseScreen: React.FC = () => (
  <section className="mx-auto max-w-lg space-y-6 p-6 text-text-primary">
    {/* Title */}
    <header className="flex items-center gap-2">
      <HelpCircle size={24} strokeWidth={1.8} />
      <h1 className="text-xl font-semibold">How to Use Score Genius</h1>
    </header>

    {/* Step list */}
    <ol className="space-y-4 border-l-2 border-brand-green pl-4">
      <li>
        <h2 className="font-medium text-brand-green">
          1. Pick your sport
        </h2>
        <p className="text-sm text-text-secondary">
          Tap <span className="pill pill-green">NBA</span> or 
          <span className="pill pill-green">MLB</span> in the header to switch
          between basketball and baseball views.
        </p>
      </li>

      <li>
        <h2 className="font-medium text-brand-green">
          2. Browse today’s games
        </h2>
        <p className="text-sm text-text-secondary">
          The <strong>Games</strong> tab lists scheduled matchups. Tap a game
          for detailed predictions, betting edges, and injury news.
        </p>
      </li>

      <li>
        <h2 className="font-medium text-brand-green">
          3. Compare model vs. market
        </h2>
        <p className="text-sm text-text-secondary">
          Green numbers signal positive value compared to Vegas odds; red means
          caution. Lines update in real time as odds shift.
        </p>
      </li>

      <li>
        <h2 className="font-medium text-brand-green">
          4. Deep‑dive stats
        </h2>
        <p className="text-sm text-text-secondary">
          In the <strong>Stats</strong> tab, explore team rankings, advanced
          metrics, and injury trends to understand each prediction.
        </p>
      </li>
    </ol>

    {/* Coming‑soon note */}
    <p className="text-center text-xs text-text-secondary">
      Interactive tour &amp; tips coming soon — powered by React Joyride.
    </p>
  </section>
);

export default HowToUseScreen;
