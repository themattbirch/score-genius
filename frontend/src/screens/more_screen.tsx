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
} from "lucide-react";
import ThemeToggle from "@/components/ui/ThemeToggle";
import type { IconType } from "@/types";

const cardBase =
  "flex items-center justify-between rounded-lg border px-4 py-3 text-sm transition-colors group";
const cardLight =
  "bg-slate-100 border-slate-300 text-slate-700 hover:bg-slate-200";
const cardDark =
  "dark:bg-[var(--color-panel)] dark:border-slate-600/60 dark:text-text-primary dark:hover:bg-slate-700";

interface LinkProps {
  href: string;
  label: string;
  Icon?: IconType;
}
const LinkItem: React.FC<LinkProps> = ({ href, label, Icon }) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className={`${cardBase} ${cardLight} ${cardDark}`}
  >
    <div className="flex items-center space-x-2">
      {Icon && (
        <Icon
          size={20}
          className="text-slate-700 dark:text-text-primary group-hover:opacity-90"
        />
      )}
      <span>{label}</span>
    </div>
    <ExternalLink
      size={16}
      className="text-slate-500 dark:text-text-secondary opacity-60"
    />
  </a>
);

const MoreScreen: React.FC = () => (
  <section className="mx-auto w-full max-w-2xl px-4 sm:px-6 lg:px-8 py-4 sm:py-6 lg:py-8 space-y-6">
    {/* ─────────────────── Information ─────────────────── */}
    <div className="space-y-3">
      <h2 className="mb-2 sm:mb-3 text-center sm:text-left text-base sm:text-lg font-semibold text-slate-800 dark:text-text-primary">
        Information
      </h2>
      <LinkItem
        href="https://scoregenius.io"
        label="About ScoreGenius"
        Icon={Info}
      />
      <LinkItem
        href="https://scoregenius.io/disclaimer"
        label="Disclaimer"
        Icon={ShieldCheck}
      />
      <LinkItem
        href="https://scoregenius.io/documentation"
        label="Documentation"
        Icon={BookOpen}
      />
      <LinkItem
        href="https://scoregenius.io/terms"
        label="Terms of Service"
        Icon={ShieldCheck}
      />
      <LinkItem
        href="https://scoregenius.io/privacy"
        label="Privacy Policy"
        Icon={FileText}
      />
    </div>

    {/* ─────────────────── Connect with Us ─────────────────── */}
    <div className="space-y-3">
      <h2 className="mb-2 sm:mb-3 text-center sm:text-left text-base sm:text-lg font-semibold text-slate-800 dark:text-text-primary">
        Connect with Us
      </h2>
      <LinkItem
        href="https://facebook.com/scoregeniusapp"
        label="Facebook – scoregeniusapp"
        Icon={Facebook}
      />
      <LinkItem
        href="https://instagram.com/scoregeniusapp"
        label="Instagram – scoregeniusapp"
        Icon={Instagram}
      />
      <LinkItem
        href="https://x.com/scoregeniusapp"
        label="X / Twitter – scoregeniusapp"
        Icon={Twitter}
      />
      <LinkItem
        href="https://youtube.com/@scoregenius"
        label="YouTube – scoregenius"
        Icon={PlayCircle}
      />
    </div>

    {/* ─────────────────── Display Options ─────────────────── */}
    <div className="space-y-3">
      <h2 className="mb-2 sm:mb-3 text-center sm:text-left text-base sm:text-lg font-semibold text-slate-800 dark:text-text-primary">
        Display Options
      </h2>
      <ThemeToggle
        variant="card"
        data-tour="theme-toggle"
        className={`${cardBase} ${cardLight} ${cardDark} w-full !justify-center`}
      />
    </div>
  </section>
);

export default MoreScreen;
