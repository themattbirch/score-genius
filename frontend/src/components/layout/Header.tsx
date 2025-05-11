// frontend/src/components/layout/Header.tsx

import React from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import { Calendar as CalendarIcon } from "lucide-react";

import { Popover, PopoverTrigger, PopoverContent } from "../ui/popover";
import { Calendar } from "../ui/calendar";
import { useSport, Sport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";

interface HeaderProps {
  showDatePicker?: boolean;
}

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
        "sticky top-0 z-40 flex items-center justify-between px-2 sm:px-4 py-2",
        // --- light vs dark wrapper -----------------------------------
        "border-b bg-white/90 backdrop-blur-md shadow-[0_1px_2px_rgba(0,0,0,0.05)] border-slate-300",
        "dark:bg-github-dark dark:border-slate-700/40 dark:shadow-none"
      )}
    >
      {/* ───── Logo & Title ───── */}
      <button
        type="button"
        className="flex flex-none items-center gap-1 sm:gap-2"
        onClick={() => navigate("/games")}
      >
        <img
          src="/basketball_header_logo.png"
          alt="Logo"
          className="h-6 w-6 sm:h-8 sm:w-8 flex-none select-none"
          draggable={false}
        />
        <span className="text-sm sm:text-base font-semibold tracking-tight text-slate-900 dark:text-white">
          Score&nbsp;Genius
        </span>
      </button>

      {/* ───── Right‑side controls ───── */}
      <div className="flex items-center gap-2 sm:gap-3">
        <SportToggle active={sport} onChange={setSport} />

        {showDatePicker && (
          <Popover>
            <PopoverTrigger asChild>
              <button
                data-tour="date-picker"
                className={clsx(
                  "inline-flex items-center gap-1 rounded-lg border px-2 sm:px-3 py-1 text-xs sm:text-sm transition-colors",
                  "border-slate-300 bg-white text-slate-700 hover:bg-gray-50",
                  "dark:border-slate-600/60 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
                )}
              >
                <CalendarIcon
                  size={14}
                  strokeWidth={1.8}
                  className="sm:size-4"
                />
                {formattedDate}
              </button>
            </PopoverTrigger>

            <PopoverContent
              side="bottom"
              align="center"
              sideOffset={8}
              className="bg-[var(--color-panel)] rounded-lg shadow-lg p-4 w-[20rem]"
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

/* ───────── Sport Toggle ───────── */

interface SportToggleProps {
  active: Sport;
  onChange: (s: Sport) => void;
}

const SportToggle: React.FC<SportToggleProps> = ({ active, onChange }) => {
  const base =
    "relative w-16 py-1 text-center text-xs font-semibold transition-colors focus:outline-none";

  return (
    <div
      className={clsx(
        "flex overflow-hidden rounded-full border",
        "border-slate-300 bg-white",
        "dark:border-slate-600/60 dark:bg-slate-800"
      )}
      data-tour="sport-switch"
    >
      {(["NBA", "MLB"] as Sport[]).map((s) => {
        const isActive = active === s;
        return (
          <button
            key={s}
            type="button"
            aria-pressed={isActive}
            onClick={() => onChange(s)}
            className={clsx(
              base,
              isActive
                ? "bg-brand-green text-github-dark"
                : "text-slate-500 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white"
            )}
          >
            {s}
          </button>
        );
      })}
    </div>
  );
};

export default Header;
