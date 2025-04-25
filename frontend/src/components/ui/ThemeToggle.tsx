// frontend/src/components/ui/ThemeToggle.tsx
import React from "react";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "@/contexts/theme_context";

// --- Allow additional props via intersection type ---
type ThemeToggleProps = {
  className?: string;
} & React.ButtonHTMLAttributes<HTMLButtonElement>; // Allow standard button attributes like data-*

const ThemeToggle: React.FC<ThemeToggleProps> = ({ // Use the new type
  className = "",
  // --- Explicitly destructure onClick to avoid potential override by ...rest ---
  // Although onClick is also a ButtonHTMLAttribute, handling it explicitly is safer
  // if you have component-specific logic tied to it (we do: toggleTheme).
  // We'll apply toggleTheme specifically later.
  // Remove onClick from here if it exists in ...rest to avoid conflict:
  // onClick,
  ...rest // Capture any other props passed in (like data-tour)
}) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      // --- Spread the captured ...rest props here ---
      {...rest}
      // --- Apply component-specific onClick ---
      onClick={toggleTheme}
      // --- Combine passed className with default styles ---
      // Note: Ensure your default styles here match what you had in MoreScreen before
      className={`flex items-center gap-2 rounded-lg border px-4 py-2 text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500
                  bg-white dark:bg-slate-800            
                  border-slate-300 dark:border-slate-600/60
                  text-slate-700 dark:text-slate-200
                  hover:bg-slate-100 dark:hover:bg-slate-700
                  ${className}`} // Append className passed via props
    >
      {theme === "light" ? (
        <Moon size={16} strokeWidth={1.8} /> // Slightly smaller icon maybe?
      ) : (
        <Sun size={16} strokeWidth={1.8} />
      )}
      <span>{theme === "light" ? "Dark Mode" : "Light Mode"}</span> {/* Wrap text in span for clarity */}
    </button>
  );
};

export default ThemeToggle;
