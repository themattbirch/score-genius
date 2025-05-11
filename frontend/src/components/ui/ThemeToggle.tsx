import React from "react";
import { Sun, Moon } from "lucide-react";
import { twMerge } from "tailwind-merge";
import { useTheme } from "@/contexts/theme_context";

type ThemeToggleProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  /**
   * When ThemeToggle sits inside a card you’re styling externally,
   * use variant="card" so the button *inherits* that card’s bg/border.
   * Default keeps the old standalone look.
   */
  variant?: "default" | "card";
};

const ThemeToggle: React.FC<ThemeToggleProps> = ({
  variant = "default",
  className = "",
  ...rest
}) => {
  const { theme, toggleTheme } = useTheme();

  // Only inject internal colours in the default variant
  const internalColours =
    variant === "default"
      ? "bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600/60 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700"
      : ""; // let the parent control colours

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className={twMerge(
        "flex items-center gap-2 rounded-lg border px-4 py-2 text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500",
        internalColours,
        className
      )}
      {...rest}
    >
      {theme === "light" ? (
        <>
          <Moon size={16} strokeWidth={1.8} />
          <span>Dark Mode</span>
        </>
      ) : (
        <>
          <Sun size={16} strokeWidth={1.8} />
          <span>Light Mode</span>
        </>
      )}
    </button>
  );
};

export default ThemeToggle;
