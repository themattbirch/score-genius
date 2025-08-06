// frontend/src/screens/more_screen.tsx
import React from "react";
import {
  ExternalLink,
  Twitter,
  Facebook,
  Instagram,
  PlayCircle,
  FileText,
  Info,
  ShieldCheck,
  BookOpen,
  Bug,
  MessageSquareText,
  RotateCcw,
  Sun,
  Moon,
  Monitor,
} from "lucide-react";

import ThemeToggle from "@/components/ui/ThemeToggle";
import { useTour } from "@/contexts/tour_context";
import { useOnline } from "@/contexts/online_context";
import OfflineBanner from "@/components/offline_banner";

type LucideIcon = React.ComponentType<React.SVGProps<SVGSVGElement>>;

/* ------------------------------------------------------------------ */
/* Shared UI bits                                                      */
/* ------------------------------------------------------------------ */
const rowBase =
  "w-full flex items-center justify-between gap-3 rounded-lg px-4 py-3 text-sm transition-colors focus-ring";
const rowLight =
  "bg-white border border-slate-300 text-slate-700 hover:bg-slate-100";
const rowDark =
  "dark:bg-[var(--color-panel)] dark:border-slate-600/60 dark:text-text-primary dark:hover:bg-slate-700/60";
const sectionTitle =
  "mb-2 sm:mb-3 text-center sm:text-left text-base sm:text-lg font-semibold text-slate-800 dark:text-text-primary";

/** External link row */
interface LinkRowProps {
  href: string;
  label: string;
  icon?: LucideIcon;
  ariaLabel?: string;
}
const LinkRow: React.FC<LinkRowProps> = ({
  href,
  label,
  icon: Icon,
  ariaLabel,
}) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    aria-label={ariaLabel ?? `${label} (opens in new tab)`}
    className={`${rowBase} ${rowLight} ${rowDark}`}
  >
    <span className="flex items-center gap-2">
      {Icon && (
        <Icon className="w-5 h-5 opacity-80 group-hover:opacity-100 transition-opacity" />
      )}
      {label}
    </span>
    <ExternalLink
      size={16}
      className="opacity-60 group-hover:opacity-90 transition-opacity"
    />
  </a>
);

/** Simple button row (internal actions) */
interface ActionRowProps {
  label: string;
  onClick: () => void;
  icon?: LucideIcon;
  rightEl?: React.ReactNode;
}
const ActionRow: React.FC<ActionRowProps> = ({
  label,
  onClick,
  icon: Icon,
  rightEl,
}) => (
  <button
    type="button"
    onClick={onClick}
    className={`${rowBase} ${rowLight} ${rowDark} text-left`}
  >
    <span className="flex items-center gap-2">
      {Icon && (
        <Icon className="opacity-80 group-hover:opacity-100 transition-opacity" />
      )}
      {label}
    </span>
    {rightEl}
  </button>
);

/* ------------------------------------------------------------------ */
/* Screen                                                              */
/* ------------------------------------------------------------------ */
const MoreScreen: React.FC = () => {
  // 1️⃣ offline fallback
  const online = useOnline();
  if (!online) {
    window.location.href = "/app/offline.html";
    return null;
  }

  const { start } = useTour();

  // Optional: detect/build version from env if you inject it
  const appVersion = import.meta.env.VITE_APP_VERSION ?? "";

  return (
    <section className="mx-auto w-full max-w-2xl px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-8 pb-[env(safe-area-inset-bottom)]">
      {/* ───────────── Information ───────────── */}
      <div className="space-y-3">
        <h2 className={sectionTitle}>Information</h2>
        <LinkRow
          href="https://scoregenius.io"
          label="About ScoreGenius"
          icon={Info}
        />
        <LinkRow
          href="https://scoregenius.io/documentation"
          label="Documentation"
          icon={BookOpen}
        />
        <LinkRow
          href="https://scoregenius.io/disclaimer"
          label="Disclaimer"
          icon={ShieldCheck}
        />
        <LinkRow
          href="https://scoregenius.io/terms"
          label="Terms of Service"
          icon={ShieldCheck}
        />
        <LinkRow
          href="https://scoregenius.io/privacy"
          label="Privacy Policy"
          icon={FileText}
        />
      </div>

      {/* ───────────── Connect / Social ───────────── */}
      <div className="space-y-3">
        <h2 className={sectionTitle}>Connect with Us</h2>
        <LinkRow
          href="https://facebook.com/scoregeniusapp"
          label="Facebook – @scoregeniusapp"
          icon={Facebook}
        />
        <LinkRow
          href="https://instagram.com/scoregeniusapp"
          label="Instagram – @scoregeniusapp"
          icon={Instagram}
        />
        <LinkRow
          href="https://x.com/scoregeniusapp"
          label="X / Twitter – @scoregeniusapp"
          icon={Twitter}
        />
        <LinkRow
          href="https://youtube.com/@scoregenius"
          label="YouTube – scoregenius"
          icon={PlayCircle}
        />
      </div>

      {/* ───────────── Feedback & Support (optional but recommended) ───────────── */}
      <div className="space-y-3">
        <h2 className={sectionTitle}>Feedback & Support</h2>
        <LinkRow
          href="mailto:hello@scoregenius.io?subject=Bug%20Report"
          label="Report a bug"
          icon={Bug}
        />
        <LinkRow
          href="mailto:hello@scoregenius.io?subject=Feature%20Request"
          label="Request a feature"
          icon={MessageSquareText}
        />
        <LinkRow
          href="https://scoregenius.io/support"
          label="Support"
          icon={Monitor}
        />
      </div>

      {/* ───────────── Appearance / Options ───────────── */}
      <div className="space-y-3">
        <h2 className={sectionTitle}>Appearance & Options</h2>

        {/* If ThemeToggle already handles internal state, just embed it. */}
        <ThemeToggle
          variant="card"
          data-tour="theme-toggle"
          className={`${rowBase} ${rowLight} ${rowDark} w-full !justify-start`}
        />

        <ActionRow
          label="Restart Interactive Tour"
          icon={RotateCcw}
          onClick={() => start()}
        />
      </div>

      {/* ───────────── Version / Meta info ───────────── */}
      {appVersion && (
        <p className="text-center text-xs text-text-secondary mt-6">
          ScoreGenius v{appVersion}
        </p>
      )}
    </section>
  );
};

export default MoreScreen;
