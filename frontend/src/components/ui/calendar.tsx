// frontend/src/ui/calendar.tsx
"use client"

import * as React from "react"
import { DayPicker, type DayPickerSingleProps } from "react-day-picker"
import "react-day-picker/dist/style.css"

/**
 * Single‑date calendar wrapper.
 */
export interface CalendarProps
  extends Omit<DayPickerSingleProps, "className" | "mode"> {
  /** extra wrapper classes */
  className?: string
  /** whether to auto‑focus the calendar when it mounts (consumed here) */
  initialFocus?: boolean
}

export const Calendar: React.FC<CalendarProps> = ({
  className,
  initialFocus,  // pulled out so TS knows it exists
  ...restProps    // includes selected, onSelect, required, etc.
}) => {
  // NOTE: DayPicker v9 does not support `initialFocus` natively.
  // If you really need that behavior, you'll have to add a `ref` + `useEffect` to focus the correct element here.
  return (
    <DayPicker
      mode="single"
      className={className}
      {...restProps}
    />
  )
}
