// frontend/src/components/layout/BottomTabBar.tsx

import React from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  Calendar as GamesIcon,
  BarChart2 as StatsIcon,
  HelpCircle as HelpIcon,
  MoreHorizontal as MoreIcon,
} from "lucide-react";
import clsx from "clsx";
import type { LucideProps } from "lucide-react";

/** ------------------------------------------------------------------
 *  Tab metadata
 *  ------------------------------------------------------------------ */
type Tab = {
  path: string;
  label: string;
  Icon: React.ComponentType<LucideProps>;
};

const TABS: Tab[] = [
  { path: "/games", label: "Games", Icon: GamesIcon },
  { path: "/stats", label: "Stats", Icon: StatsIcon },
  { path: "/how-to-use", label: "How To Use", Icon: HelpIcon },
  { path: "/more", label: "More", Icon: MoreIcon },
];

/** ------------------------------------------------------------------
 *  Component
 *  ------------------------------------------------------------------ */
const BottomTabBar: React.FC = () => {
  const { pathname } = useLocation();

  return (
    <nav
      className={clsx(
        "fixed inset-x-0 bottom-0 z-40 flex",
        "border-t border-slate-700/40 bg-github-dark",
        "pb-[env(safe-area-inset-bottom)]"
      )}
    >
      {TABS.map(({ path, label, Icon }) => {
        const isActive = pathname.startsWith(path);

        return (
          <NavLink
            key={path}
            to={path}
            data-tour={path === "/stats" ? "tab-stats" : undefined}
            className={clsx(
              "flex flex-1 flex-col items-center gap-0.5 py-2 text-xs font-medium",
              "transition duration-150",
              isActive
                ? "text-brand-green"
                : "text-text-secondary hover:text-text-primary"
            )}
          >
            <Icon size={20} strokeWidth={1.8} />
            <span>{label}</span>
          </NavLink>
        );
      })}
    </nav>
  );
};

export default BottomTabBar;
