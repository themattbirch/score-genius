// tailwind.config.cjs
const defaultTheme = require("tailwindcss/defaultTheme");
const forms = require("@tailwindcss/forms");
const typography = require("@tailwindcss/typography");
const plugin = require("tailwindcss/plugin");

module.exports = {
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
        "btn-snapshot": "var(--color-btn-snapshot, #00B140)", // Default to brand-green DEFAULT
        "badge-weather": "var(--color-badge-weather, #FF7F00)", // Default to brand-orange
      },
      fontFamily: {
        sans: ['"Source Sans 3"', ...defaultTheme.fontFamily.sans],
        serif: ['"PT Serif"', ...defaultTheme.fontFamily.serif],
      },
      container: { center: true, padding: "1rem" },
      borderRadius: { xl: "1rem" },
    },
  },
  plugins: [
    plugin(function ({ addVariant, e }) {
      addVariant("dark", ({ modifySelectors, separator }) => {
        modifySelectors(
          ({ className }) => `.dark .${e(`dark${separator}${className}`)}`
        );
      });
    }),
    forms,
    typography,
    /** ---- Day-Picker reset --------------------------------------- **/
    function ({ addUtilities }) {
      addUtilities({
        /* remove outer margin + let it stretch */
        ".calendar-reset .rdp": {
          margin: "0 !important",
          width: "100% !important",
          maxWidth: "none !important",
          display: "block !important",
        },
        /* stop centering & kill 16px gap */
        ".calendar-reset .rdp-months": {
          display: "block !important",
          justifyContent: "flex-start !important",
          gap: "0 !important",
        },
        /* kill 16px margin on the month card */
        ".calendar-reset .rdp-month": {
          margin: "0 !important",
          padding: "0 !important",
          width: "100% !important",
        },
        ".calendar-reset .rdp-month_grid": {
          width: "100% !important",
          tableLayout: "fixed !important",
        },
        /* turn the caption row into a 3-col grid */
        ".calendar-reset .rdp-month_caption": {
          paddingLeft: "0.75rem !important",
          paddingTop: "0.75rem !important",
        },

        /* keep the label flush left inside its column */
        ".calendar-reset .rdp-month_caption > .rdp-caption_label, \
 .calendar-reset .rdp-month_caption span": {
          justifySelf: "start !important",
        },
      });
    },
  ],
};
