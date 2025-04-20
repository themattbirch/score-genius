// frontend/src/contexts/date_context.tsx

import React, { createContext, useContext, useState, ReactNode } from 'react';

interface DateCtx {
  date: Date;
  setDate: (d: Date) => void;
}

const DateContext = createContext<DateCtx | undefined>(undefined);

export const DateProvider = ({ children }: { children: ReactNode }) => {
  const [date, setDate] = useState<Date>(new Date());
  return (
    <DateContext.Provider value={{ date, setDate }}>
      {children}
    </DateContext.Provider>
  );
};

export const useDate = (): DateCtx => {
  const ctx = useContext(DateContext);
  if (!ctx) throw new Error('useDate must be used inside <DateProvider>');
  return ctx;
};
