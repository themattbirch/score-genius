// frontend/src/components/layout/Header.tsx
import React from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import { Calendar as CalendarIcon } from "lucide-react";

import LogoWordmark from "./logo_wordmark";

import { Popover, PopoverTrigger, PopoverContent } from "../ui/popover";
import { Calendar } from "../ui/calendar";
import { useSport, Sport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";

interface HeaderProps {
  showDatePicker?: boolean;
}

const SPORT_OPTIONS: Sport[] = ["MLB", "NBA", "NFL"];

const Header: React.FC<HeaderProps> = ({ showDatePicker = false }) => {
  const navigate = useNavigate();
  const { sport, setSport } = useSport();
  const { date, setDate } = useDate();

  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });

  return (
    <header
      className={clsx(
        "sticky top-0 z-40 flex items-center justify-between px-6 py-2",
        "border-b bg-white/90 backdrop-blur-md shadow-[0_1px_2px_rgba(0,0,0,0.05)] border-slate-300",
        "dark:bg-black dark:border-slate-700/40"
      )}
    >
      {/* Logo */}
      <button
        onClick={() => navigate("/games")}
        className="flex items-center gap-2"
      >
        <LogoWordmark className="logo-svg h-5 sm:h-7 md:h-8 w-auto" />
      </button>

      {/* Sport switcher + optional date picker */}
      <div className="flex items-center gap-3">
        <SportSwitcher active={sport} onChange={setSport} />
        {showDatePicker && (
          <Popover>
            <PopoverTrigger asChild>
              <button
                data-tour="date-picker"
                className={clsx(
                  "inline-flex items-center gap-1 rounded-lg border px-3 py-1 text-sm",
                  "border-slate-300 bg-white text-slate-700 hover:bg-gray-50",
                  "dark:border-slate-600/60 dark:bg-slate-800 dark:text-slate-300"
                )}
              >
                <CalendarIcon size={16} strokeWidth={1.8} />
                {formattedDate}
              </button>
            </PopoverTrigger>
            <PopoverContent
              side="bottom"
              align="end"
              sideOffset={4}
              className="bg-[var(--color-panel)] rounded-lg shadow-lg p-4 w-72"
            >
              <Calendar
                selected={date}
                onSelect={(d) => d && setDate(d)}
                className="calendar-reset [--rdp-cell-size:2.5rem]"
              />
            </PopoverContent>
          </Popover>
        )}
      </div>
    </header>
  );
};

/* ───────── SportSwitcher ───────── */

interface SportSwitcherProps {
  active: Sport;
  onChange: (s: Sport) => void;
}

const SportSwitcher: React.FC<SportSwitcherProps> = ({ active, onChange }) => (
  <div
    data-tour="sport-switch"
    className={clsx(
      "flex gap-1.5 overflow-x-auto sm:overflow-visible",
      "px-1 py-1 rounded-full border",
      "border-slate-300 bg-white dark:border-slate-600/60 dark:bg-slate-800"
    )}
  >
    {SPORT_OPTIONS.map((s) => {
      const isActive = s === active;
      return (
        <button
          key={s}
          type="button"
          aria-pressed={isActive}
          onClick={() => onChange(s)}
          className={clsx(
            "flex-shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold transition-colors focus:outline-none",
            isActive
              ? "bg-green-600 text-white shadow-sm"
              : "bg-[var(--color-panel)] text-text-secondary hover:bg-[var(--color-surface-hover)] dark:bg-[var(--color-panel)] dark:text-text-secondary dark:hover:bg-slate-700"
          )}
        >
          {s}
        </button>
      );
    })}
  </div>
);

export default Header;
