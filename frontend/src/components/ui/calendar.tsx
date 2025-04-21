// frontend/src/ui/calendar.tsx
"use client";

import * as React from "react";
import {
  DayPicker,
  type DayPickerSingleProps,
  type DayPickerProps,
} from "react-day-picker";
import { ChevronLeft, ChevronRight } from "lucide-react";
import "react-day-picker/dist/style.css";

export interface CalendarProps
  extends Omit<DayPickerSingleProps, "classNames" | "components" | "mode"> {
  className?: string;
}

export const Calendar: React.FC<CalendarProps> = ({ className, ...rest }) => {
  // Custom Components: Use Lucide icons with "currentColor" stroke
  const components: Partial<DayPickerProps["components"]> = {
    Chevron: ({ orientation, className: cn, ...props }: any) => {
      const Icon = orientation === "left" ? ChevronLeft : ChevronRight;
      // Keep stroke="currentColor"
      return <Icon {...props} className={cn} stroke="currentColor" />;
    },
  };

  // Class Names: Apply Tailwind classes, INCLUDING text color for nav buttons
  const classNames: DayPickerProps["classNames"] = {
    months: "grid grid-cols-1",
    month: "bg-[var(--color-panel)] rounded-lg w-[18rem] shadow-lg",
    caption:
      "flex items-center justify-between px-4 py-2 bg-[var(--color-panel)] text-[var(--color-text-primary)] rounded-t-lg",
    nav: "flex items-center gap-2",
    // --- MODIFICATION START ---
    // Add text color class directly to the buttons
    button_previous:
      "h-8 w-8 flex items-center justify-center text-[var(--color-text-primary)]",
    button_next:
      "h-8 w-8 flex items-center justify-center text-[var(--color-text-primary)]",
    // --- MODIFICATION END ---
    caption_label: "font-semibold text-lg",
    head: "grid grid-cols-7 text-sm text-[var(--color-text-secondary)] px-2",
    row: "grid grid-cols-7 gap-1 px-2 mb-1",
    cell: "w-8 h-8 flex items-center justify-center rounded",
    day: "text-[var(--color-text-secondary)] hover:bg-[var(--color-brand-orange)/20]",
    day_selected: "bg-[var(--color-brand-green)] text-[var(--color-bg)]",
    day_today: "underline",
  };

  // Style Variables: Override default react-day-picker CSS variables
  const styleVars = {
    "--rdp-accent-color": "var(--color-text-primary)",
    "--rdp-accent-background-color": "var(--color-panel)",
  } as React.CSSProperties;

  // Styles Overrides: We remove the color override here as it wasn't working
  // and is now handled by classNames. Keep backgroundImage: none.
  const stylesOverrides: DayPickerProps["styles"] = {
    nav_button_previous: {
      backgroundImage: "none",
    },
    nav_button_next: {
      backgroundImage: "none",
    },
  };

  return (
    <DayPicker
      mode="single"
      {...rest}
      className={className}
      components={components}
      classNames={classNames} // Apply classes (now includes text color)
      style={styleVars}
      styles={stylesOverrides} // Only applies backgroundImage:none now
    />
  );
};
