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

const BottomTabBar: React.FC = () => {
  const { pathname } = useLocation();

  return (
    <nav
      className={clsx(
        "fixed inset-x-0 bottom-0 z-40 flex",
        // ----- light vs dark wrapper ------------------------------------------------
        "border-t bg-white/90 backdrop-blur-md shadow-[0_-1px_2px_rgba(0,0,0,0.05)]",
        "dark:bg-github-dark/95 dark:border-slate-700/40 dark:shadow-none",
        "border-slate-300",
        // safe‑area padding for iOS
        "pb-[env(safe-area-inset-bottom)]"
      )}
    >
      {TABS.map(({ path, label, Icon }) => {
        const isActive = pathname.startsWith(path);

        const tourAttr: Record<string, string | undefined> = {
          "/games": "tab-games",
          "/stats": "tab-stats",
          "/more": "tab-more",
        };

        return (
          <NavLink
            key={path}
            to={path}
            data-tour={tourAttr[path]}
            className={clsx(
              "flex flex-1 flex-col items-center gap-0.5 py-2 text-xs font-medium transition-colors",
              isActive
                ? "text-brand-green"
                : "text-slate-500 hover:text-slate-900 dark:text-text-secondary dark:hover:text-text-primary"
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
