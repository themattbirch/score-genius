// frontend/src/contexts/sport_context.tsx

import React, { createContext, useContext, useState, ReactNode } from 'react';

export type Sport = 'NBA' | 'MLB';

interface SportContextValue {
  sport: Sport;
  setSport: (s: Sport) => void;
}

const SportContext = createContext<SportContextValue | undefined>(undefined);

export const SportProvider = ({ children }: { children: ReactNode }) => {
  const [sport, setSport] = useState<Sport>('NBA');
  return (
    <SportContext.Provider value={{ sport, setSport }}>
      {children}
    </SportContext.Provider>
  );
};

export const useSport = (): SportContextValue => {
  const ctx = useContext(SportContext);
  if (!ctx) throw new Error('useSport must be used inside <SportProvider>');
  return ctx;
};
