import React from 'react';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import { Calendar as CalendarIcon } from 'lucide-react';

import { Popover, PopoverTrigger, PopoverContent } from '../ui/popover';
import { Calendar } from '../ui/calendar';

import { useSport, Sport } from '@/contexts/sport_context';
import { useDate } from '@/contexts/date_context';

interface HeaderProps {
  /** Only true on the Games screen so we don’t render the picker elsewhere */
  showDatePicker?: boolean;
}

const Header: React.FC<HeaderProps> = ({ showDatePicker = false }) => {
  const navigate = useNavigate();

  /* ---- global state ------------------------------------------------ */
  const { sport, setSport } = useSport();
  const { date, setDate } = useDate();

  const formattedDate = date.toLocaleDateString('en-US', {
    month: 'short',
    day: '2-digit',
  });

  return (
    <header className="sticky top-0 z-40 flex items-center justify-between border-b border-slate-700/40 bg-github-dark px-4 py-2 shadow-sm">
      {/* Logo + title -------------------------------------------------- */}
      <button
        type="button"
        className="flex items-center gap-2"
        onClick={() => navigate('/games')}
      >
        <img
          src="/orange_football_header_logo.png"
          alt="Logo"
          className="h-5 w-5 flex-none select-none"
          draggable={false}
        />
        <span className="font-semibold tracking-tight text-white">
          Score&nbsp;Genius
        </span>
      </button>

      {/* Right‑side controls ------------------------------------------ */}
      <div className="flex items-center gap-3">
        <SportToggle active={sport} onChange={setSport} />

        {showDatePicker && (
          <Popover>
            <PopoverTrigger asChild>
              <button className="inline-flex items-center gap-1 rounded-lg border border-slate-600/60 bg-slate-800 px-3 py-1 text-sm">
                <CalendarIcon size={16} strokeWidth={1.8} />
                {formattedDate}
              </button>
            </PopoverTrigger>

            <PopoverContent className="w-auto p-0">
              <Calendar
                selected={date}
                onSelect={(d) => d && setDate(d)}
                initialFocus
              />
            </PopoverContent>
          </Popover>
        )}
      </div>
    </header>
  );
};

/* ------------------------------------------------------------------ */
/* Sport toggle                                                       */
/* ------------------------------------------------------------------ */
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
      {(['NBA', 'MLB'] as Sport[]).map((s) => (
        <button
          key={s}
          type="button"
          className={clsx(base, {
            [pill]: active !== s,
            'bg-brand-green text-github-dark': active === s,
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
