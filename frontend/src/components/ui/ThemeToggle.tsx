import React from "react";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "@/contexts/theme_context";

const ThemeToggle: React.FC<{ className?: string }> = ({ className = "" }) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      data-tour="theme-toggle"
      className={`flex items-center gap-2 rounded-lg border border-slate-600/60 bg-slate-800 px-4 py-2 text-sm text-slate-200 hover:bg-slate-700 ${className}`}
    >
      {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
      {theme === "light" ? "Dark Mode" : "Light Mode"}
    </button>
  );
};

export default ThemeToggle;
