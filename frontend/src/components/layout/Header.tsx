// frontend/src/components/layout/Header.tsx

import React from 'react';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import { Calendar as CalendarIcon } from 'lucide-react';

/** ------------------------------------------------------------------
 *  Types
 *  ------------------------------------------------------------------ */
export type Sport = 'NBA' | 'MLB';

interface HeaderProps {
  /** Currently‑selected sport */
  sport: Sport;
  /** Callback when user toggles sport */
  onSportChange: (sport: Sport) => void;
  /** Show the date‑picker button (Games screen only) */
  showDatePicker?: boolean;
  /** Currently‑selected date (UTC midnight) */
  selectedDate?: Date;
  /** Callback when user clicks date button */
  onDateChange?: () => void;
}

/** ------------------------------------------------------------------
 *  Component
 *  ------------------------------------------------------------------ */
const Header: React.FC<HeaderProps> = ({
  sport,
  onSportChange,
  showDatePicker = false,
  selectedDate,
  onDateChange,
}) => {
  const navigate = useNavigate();

  const formattedDate =
    selectedDate?.toLocaleDateString('en-US', {
      month: 'short',
      day: '2-digit',
    });

  return (
    <header className="sticky top-0 z-40 flex items-center justify-between border-b border-slate-700/40 bg-github-dark px-4 py-2 shadow-sm">
      {/* Logo + title ------------------------------------------------- */}
      <button
        type="button"
        className="flex items-center gap-2"
        onClick={() => navigate('/games')}
      >
        <img
          src="/orange_football_header_logo.png"
          alt=""
          className="h-5 w-5 flex-none select-none"
          draggable={false}
        />
        <span className="font-semibold tracking-tight text-white">
          Score&nbsp;Genius
        </span>
      </button>

      {/* Right‑side controls ---------------------------------------- */}
      <div className="flex items-center gap-3">
        {/* Sport toggle */}
        <SportToggle active={sport} onChange={onSportChange} />

        {/* Date button (optional) */}
        {showDatePicker && onDateChange && (
          <button
            type="button"
            onClick={onDateChange}
            className="inline-flex items-center gap-1 rounded-lg border border-slate-600/60 bg-slate-800 px-3 py-1 text-sm font-medium text-slate-200 transition hover:bg-slate-700 active:bg-slate-600"
          >
            <CalendarIcon size={16} strokeWidth={1.75} />
            {formattedDate && <span>{formattedDate}</span>}
          </button>
        )}
      </div>
    </header>
  );
};

/** ------------------------------------------------------------------
 *  Segmented sport selector
 *  ------------------------------------------------------------------ */
interface SportToggleProps {
  active: Sport;
  onChange: (sport: Sport) => void;
}

const SportToggle: React.FC<SportToggleProps> = ({ active, onChange }) => {
  const base =
    'relative w-16 py-1 text-center text-xs font-semibold transition focus:outline-none';
  const pill =
    'rounded-full border border-slate-600/60 bg-slate-800 text-slate-300';

  return (
    <div className="flex overflow-hidden rounded-full border border-slate-600/60 bg-slate-800">
      {(['NBA', 'MLB'] as Sport[]).map((sport) => (
        <button
          key={sport}
          type="button"
          className={clsx(base, {
            [pill]: active !== sport,
            'bg-brand-green text-github-dark': active === sport,
          })}
          onClick={() => onChange(sport)}
          aria-pressed={active === sport}
        >
          {sport}
        </button>
      ))}
    </div>
  );
};

/** ------------------------------------------------------------------
 *  Brand colors (Tailwind config or globals)
 *  ------------------------------------------------------------------ */
/*  .bg-github-dark  –  #0d1117
    .bg-brand-green  –  #34d058
    These are assumed to be included in tailwind.config.js under
    theme.extend.colors. */

export default Header;
