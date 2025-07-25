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
        "border-subtle": "var(--color-border-subtle)",
        "border-strong": "var(--color-border-strong)",
        muted: "var(--color-text-secondary)",
        surface: "var(--color-panel)",
        "surface-hover": "var(--color-surface-hover)",
        "text-base": "var(--color-text-primary)",
        "text-muted": "var(--color-text-secondary)",
        "accent-neutral": "var(--color-accent-neutral)",
        "surface-hover": "var(--color-surface-hover)",
        "border-subtle": "var(--color-border-subtle)",
        "border-strong": "var(--color-border-strong)",
        "pred-badge": "var(--color-pred-badge)",
        boxShadow: {
          "howto-glow": "0 0 0 3px rgba(148,163,184,.28)",
        },
      },
      boxShadow: {
        card: "var(--shadow-card)",
        "card-hover": "var(--shadow-card-hover)",
        focus: "0 0 0 2px rgba(0,177,64,.4)", // accessible green ring
      },
      transitionDuration: {
        120: "120ms",
        160: "160ms",
      },
      screens: {
        "3xl": "1920px", // optional ultra-wide grid
      },
      fontFamily: {
        sans: ['"Source Sans 3"', ...defaultTheme.fontFamily.sans],
        serif: ['"PT Serif"', ...defaultTheme.fontFamily.serif],
      },
      container: { center: true, padding: "1rem" },
      borderRadius: {
        xl: "1rem",
        // Compact radius for mobile cards
        lg: "0.75rem",
      },
    },
  },
  plugins: [
    plugin(function ({ addVariant, e }) {
      addVariant("dark", ({ modifySelectors, separator }) => {
        modifySelectors(
          ({ className }) => `.dark .${e(`dark${separator}${className}`)}`
        );
      });
      // ARIA state variants
      addVariant("aria-selected", '&[aria-selected="true"]');
      addVariant("aria-expanded", '&[aria-expanded="true"]');
      addVariant("aria-current", '&[aria-current="true"]');
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
