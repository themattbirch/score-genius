// src/contexts/theme_context.tsx
import React, {
  createContext,
  useState,
  useEffect,
  useContext,
  useMemo,
  type ReactNode,
} from "react";

// Define the possible theme types
export type Theme = "light" | "dark";

// Define shape of the context value
interface ThemeContextProps {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

// Create the context
const ThemeContext = createContext<ThemeContextProps | undefined>(undefined);

// Define the Provider component
export const ThemeProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  // State to hold current theme
  const [theme, setThemeState] = useState<Theme>(() => {
    try {
      const storedTheme = localStorage.getItem("theme") as Theme | null;
      // If nothing stored, default to 'dark' <--- CHANGED HERE
      const initial = storedTheme || "dark";
      console.log(
        `%c[ThemeProvider useState Init] Setting initial theme state...: ${initial}`,
        "color: blue"
      );
      return initial;
    } catch (error) {
      console.error("Error reading localStorage for theme", error);
      // Also change the error fallback default if you want consistency
      console.log(
        `%c[ThemeProvider useState Init] Defaulting theme state to 'dark' due to error`,
        "color: blue"
      );
      return "dark"; // <-- CHANGED ERROR FALLBACK TOO
    }
  });

  // *** ADD LOG *** - Inside the component body, before useEffect
  console.log(
    `%c[ThemeProvider Render] Rendering provider. Current theme state: ${theme}`,
    "color: blue"
  );

  // Effect to apply the class to <html> and update localStorage
  useEffect(() => {
    // *** ADD LOGS ***
    console.log(
      `%c[ThemeProvider useEffect] ---- START ---- (Theme: ${theme})`,
      "color: green; font-weight: bold;"
    );
    const root = document.documentElement; // Get <html> element
    console.log(
      `%c[ThemeProvider useEffect] <html> classes BEFORE change: "${root.className}"`,
      "color: green;"
    );

    root.classList.remove("light", "dark"); // Remove any existing theme class
    if (theme === "light" || theme === "dark") {
      root.classList.add(theme); // Add the current theme class
      console.log(
        `%c[ThemeProvider useEffect] Applied class: "${theme}"`,
        "color: green;"
      );
    } else {
      console.warn(
        `[ThemeProvider useEffect] Invalid theme value: "${theme}", no class applied.`
      );
    }

    console.log(
      `%c[ThemeProvider useEffect] <html> classes AFTER change: "${root.className}"`,
      "color: green;"
    );

    // Persist the theme preference to localStorage
    try {
      localStorage.setItem("theme", theme);
    } catch (error) {
      console.error("Error writing theme to localStorage", error);
    }

    console.log(
      `%c[ThemeProvider useEffect] ---- END ----`,
      "color: green; font-weight: bold;"
    );
    // *** END ADD LOGS ***
  }, [theme]); // Dependency array ensures this runs when 'theme' state changes

  // Function to toggle the theme
  const toggleTheme = () => {
    setThemeState((prevTheme) => (prevTheme === "light" ? "dark" : "light"));
  };

  // Memoize the context value to optimize performance
  const value = useMemo(
    () => ({
      theme,
      setTheme: setThemeState,
      toggleTheme,
    }),
    [theme] // Re-create value only if theme changes
  );

  // Provide the context value to children
  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
};

// Custom hook to easily consume the ThemeContext
export const useTheme = (): ThemeContextProps => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    // Ensure the hook is used within a ThemeProvider
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
};
