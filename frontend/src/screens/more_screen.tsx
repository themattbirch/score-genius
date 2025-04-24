// frontend/src/screens/more_screen.tsx

import React from "react";
import { ExternalLink } from "lucide-react";
import ThemeToggle from "@/components/ui/ThemeToggle";

const LinkItem = ({ href, label }: { href: string; label: string }) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className="
      flex items-center justify-between
      rounded-lg border
      bg-slate-100 dark:bg-slate-800
      border-slate-300 dark:border-slate-600/60
      px-4 py-3 text-sm
      text-slate-700 dark:text-slate-200
      hover:bg-slate-200 dark:hover:bg-slate-700
      transition-colors
    "
  >
    {label}
    <ExternalLink
      size={16}
      className="opacity-60 text-slate-500 dark:text-slate-400"
    />
  </a>
);

const MoreScreen: React.FC = () => (
  <section
    className="
    mx-auto
    w-full
    max-w-2xl
    px-4 sm:px-6 lg:px-8
    py-4 sm:py-6 lg:py-8
    space-y-6
  "
  >
    <div className="space-y-3">
      <h2
        className="
        text-base sm:text-lg
        font-semibold
        text-slate-800 dark:text-slate-100
        text-center sm:text-left
        mb-2 sm:mb-3
      "
      >
        Information
      </h2>

      <LinkItem href="https://scoregenius.io" label="About Score Genius" />
      <LinkItem href="https://scoregenius.io/disclaimer" label="Disclaimer" />
      <LinkItem href="https://scoregenius.io/terms" label="Terms of Service" />
      <LinkItem href="https://scoregenius.io/privacy" label="Privacy Policy" />
    </div>

    <div className="space-y-3">
      <h2
        className="
        text-base sm:text-lg
        font-semibold
        text-slate-800 dark:text-slate-100
        text-center sm:text-left
        mb-2 sm:mb-3
      "
      >
        Display Options
      </h2>
      <ThemeToggle
        className="w-full justify-center
                bg-white dark:bg-slate-800
                border border-slate-300 dark:border-slate-600/60
                text-slate-700 dark:text-slate-200
                hover:bg-slate-100 dark:hover:bg-slate-700 // Hover is now slate-100 in light
                px-4 py-3 text-sm
                rounded-lg
        "
      />
    </div>
  </section>
);

export default MoreScreen;
