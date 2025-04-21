// frontend/src/ui/calendar.tsx
"use client";

// Make sure React is imported if not already implicitly
import React from "react";
import {
  DayPicker,
  type DayPickerSingleProps,
  type DayPickerProps,
} from "react-day-picker";
import { ChevronLeft, ChevronRight } from "lucide-react";
import "react-day-picker/dist/style.css";
// 1. Import useTheme
import { useTheme } from "@/contexts/theme_context"; // Adjust path if needed

export interface CalendarProps
  extends Omit<DayPickerSingleProps, "classNames" | "components" | "mode"> {
  className?: string;
}

export const Calendar: React.FC<CalendarProps> = ({ className, ...rest }) => {
  const { theme } = useTheme();

  const components: Partial<DayPickerProps["components"]> = {
    Chevron: ({ orientation, className: cn, ...props }: any) => {
      const Icon = orientation === "left" ? ChevronLeft : ChevronRight;

      // --- TEST: Use a different dark color literal for light mode ---
      const lightModeColor = "#6b7280"; // A standard dark grey hex code
      const darkModeColor = "#f1f5f9"; // Your light color for dark mode
      const strokeColor = theme === "light" ? lightModeColor : darkModeColor;
      // --- END TEST ---

      return (
        <Icon
          {...props}
          className={cn}
          stroke={strokeColor} // Apply the literal hex code string
        />
      );
    },
  };

  // Class Names - Keep as is (no color class on buttons)
  const classNames: DayPickerProps["classNames"] = {
    months: "grid grid-cols-1",
    month: "bg-[var(--color-panel)] rounded-lg w-[18rem] shadow-lg",
    caption:
      "flex items-center justify-between px-4 py-2 bg-[var(--color-panel)] text-[var(--color-text-primary)] rounded-t-lg",
    nav: "flex items-center gap-2",
    button_previous: "h-8 w-8 flex items-center justify-center",
    button_next: "h-8 w-8 flex items-center justify-center",
    caption_label: "font-semibold text-lg",
    head: "grid grid-cols-7 text-sm text-[var(--color-text-secondary)] px-2",
    row: "grid grid-cols-7 gap-1 px-2 mb-1",
    cell: "w-8 h-8 flex items-center justify-center rounded",
    day: "text-[var(--color-text-secondary)] hover:bg-[var(--color-brand-orange)/20]",
    day_selected: "bg-[var(--color-brand-green)] text-[var(--color-bg)]",
    day_today: "underline",
  };

  // Style Variables - Keep as is
  const styleVars = {
    "--rdp-accent-color": "var(--color-text-primary)",
    "--rdp-accent-background-color": "var(--color-panel)",
  } as React.CSSProperties;

  // Styles Overrides - Keep button text color for robustness/fallback
  const stylesOverrides: DayPickerProps["styles"] = {
    nav_button_previous: {
      color: "var(--color-text-primary)",
      backgroundImage: "none",
    },
    nav_button_next: {
      color: "var(--color-text-primary)",
      backgroundImage: "none",
    },
  };

  // Render DayPicker
  return (
    <DayPicker
      mode="single"
      {...rest}
      className={className}
      components={components}
      classNames={classNames}
      style={styleVars}
      styles={stylesOverrides}
    />
  );
};
