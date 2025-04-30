import defaultTheme from "tailwindcss/defaultTheme";
import forms from "@tailwindcss/forms";
import typography from "@tailwindcss/typography";
import lineClamp from "@tailwindcss/line-clamp";

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./app.html",
    "./public/**/*.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "github-dark": "#0d1117",
        "brand-green": { DEFAULT: "#00B140", light: "#34d058" },
        "brand-orange": "#FF7F00",
        "text-primary": "#F1F5F9",
        "text-secondary": "#9CA3AF",
      },
      fontFamily: {
        sans: ['"Source Sans 3"', ...defaultTheme.fontFamily.sans],
        serif: ['"PT Serif"', ...defaultTheme.fontFamily.serif],
      },
      container: { center: true, padding: "1rem" },
      borderRadius: { xl: "1rem" },
    },
  },
  plugins: [forms, typography, lineClamp],
};
