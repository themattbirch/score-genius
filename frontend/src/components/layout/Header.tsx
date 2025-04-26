// frontend/src/components/layout/Header.tsx

import React from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import { Calendar as CalendarIcon } from "lucide-react";

import { Popover, PopoverTrigger, PopoverContent } from "../ui/popover"; // Adjust path if needed
import { Calendar } from "../ui/calendar"; // Adjust path if needed

import { useSport, Sport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";

interface HeaderProps {
  /** Only true on the Games screen so we don’t render the picker elsewhere */
  showDatePicker?: boolean;
}

const Header: React.FC<HeaderProps> = ({ showDatePicker = false }) => {
  const navigate = useNavigate();

  /* ---- global state ------------------------------------------------ */
  const { sport, setSport } = useSport();
  const { date, setDate } = useDate();

  // Original Format:
  // const formattedDate = date.toLocaleDateString("en-US", {
  //   month: "short",
  //   day: "2-digit", // e.g., Apr 09
  // });

  // Shorter Format (Consider for mobile):
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short", // e.g., "Apr"
    day: "numeric", // e.g., "9" -> results in "Apr 9"
  });
  // Even Shorter Alternative:
  // const formattedDate = `${date.getMonth() + 1}/${date.getDate()}`; // e.g., "4/9"

  return (
    // Apply responsive padding: px-2 default, px-4 on sm screens and up
    <header className="sticky top-0 z-40 flex items-center justify-between border-b border-slate-700/40 bg-github-dark px-2 sm:px-4 py-2 shadow-sm">
      {/* Logo + title -------------------------------------------------- */}
      <button
        type="button"
        // Apply responsive gap, add flex-none
        className="flex flex-none items-center gap-1 sm:gap-2"
        onClick={() => navigate("games")}
      >
        <img
          src="/orange_football_header_logo.png"
          alt="Logo"
          // Apply responsive size
          className="h-6 w-6 sm:h-8 sm:w-8 flex-none select-none"
          draggable={false}
        />
        {/* Apply responsive text size */}
        <span className="text-sm sm:text-base font-semibold tracking-tight text-white">
          Score&nbsp;Genius
        </span>
      </button>

      {/* Right‑side controls ------------------------------------------ */}
      {/* Apply responsive gap */}
      <div className="flex items-center gap-2 sm:gap-3">
        <SportToggle active={sport} onChange={setSport} />

        {showDatePicker && (
          <Popover>
            <PopoverTrigger asChild>
              <button
                // Apply responsive padding and text size
                className="inline-flex items-center gap-1 rounded-lg border border-slate-600/60
                          bg-slate-800 px-2 sm:px-3 py-1 text-xs sm:text-sm text-slate-300"
                data-tour="date-picker"
              >
                {/* Apply responsive icon size */}
                {/* Note: size prop sets base, className sets sm override */}
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
              align="start"
              sideOffset={8}
              className="
                bg-[var(--color-panel)] rounded-lg shadow-lg
                overflow-visible min-h-[20rem] w-[18rem] p-4
              "
            >
              <Calendar
                selected={date}
                onSelect={(d) => d && setDate(d)}
                // Keep internal calendar day size consistent for now
                className="[&_.rdp-day]:w-8 [&_.rdp-day]:h-8"
              />
            </PopoverContent>
          </Popover>
        )}
      </div>
    </header>
  );
};

/* ------------------------------------------------------------------ */
/* Sport toggle (keeping original, text is already small)            */
/* ------------------------------------------------------------------ */
interface SportToggleProps {
  active: Sport;
  onChange: (sport: Sport) => void;
}

const SportToggle: React.FC<SportToggleProps> = ({ active, onChange }) => {
  // Base classes include text-xs, which is already small
  const base =
    "relative w-16 py-1 text-center text-xs font-semibold transition focus:outline-none";
  const pill = "border border-slate-600/60 bg-slate-800 text-slate-300"; // Inactive state // Removed rounded-full from here, apply on parent

  return (
    // Apply rounding to the container div
    <div
      className="flex overflow-hidden rounded-full border border-slate-600/60 bg-slate-800 "
      data-tour="sport-switch"
    >
      {(["NBA", "MLB"] as Sport[]).map((s) => (
        <button
          key={s}
          type="button"
          // Apply base, remove pill conditionally, add active styles
          className={clsx(base, {
            [pill]: false, // Never apply the inactive styles directly here
            "text-slate-300": active !== s, // Inactive text color
            "bg-brand-green text-github-dark": active === s, // Active background/text
          })}
          onClick={() => onChange(s)}
          aria-pressed={active === s}
        >
          {s}
        </button>
      ))}
    </div>
  );
};

export default Header;
