// src/screens/more_screen.tsx
import React from "react";
// 1. Import useTheme hook (adjust path if your context file is elsewhere)
import { useTheme } from "@/contexts/theme_context";

const MoreScreen: React.FC = () => {
  // 2. Get theme state and toggle function from the context
  const { theme, toggleTheme } = useTheme();

  console.log(
    `%c[MoreScreen] Rendering... Current theme: ${theme}`,
    "color: teal"
  );

  return (
    // 3. Add a simple div with padding and the button
    <div className="p-4">
      <h2 className="text-lg font-semibold mb-4">More Screen</h2>

      <p className="mb-2">
        Current theme state:{" "}
        <span className="font-mono font-semibold">{theme}</span>
      </p>

      <button
        onClick={toggleTheme} // Call the toggle function on click
        className="rounded bg-indigo-600 px-4 py-2 text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2" // Basic styling
      >
        Toggle to {theme === "light" ? "Dark" : "Light"} Mode
      </button>

      <p className="mt-4 text-sm text-gray-500">
        Clicking the button should change the theme state, update localStorage,
        and add/remove the 'dark' class on the &lt;html&gt; tag (check DevTools
        Elements tab and console logs).
      </p>
    </div>
  );
};

export default MoreScreen;
