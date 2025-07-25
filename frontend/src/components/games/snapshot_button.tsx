// frontend/src/components/games/snapshot_button.tsx
import React from "react";
import { BarChart3 } from "lucide-react";
import clsx from "clsx";

export interface SnapshotButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  isDisabled?: boolean;
  tooltipText?: string;
  label?: string;
}

const SnapshotButton: React.FC<SnapshotButtonProps> = ({
  onClick,
  isDisabled = false,
  tooltipText = "View Snapshot",
  className = "",
  label = "H2H Stats",
  ...rest
}) => {
  // 1) capture-phase guard
  const handleMouseDown: React.MouseEventHandler<HTMLButtonElement> = (e) => {
    console.log("▶️ SnapshotButton onMouseDown (capture)");
    e.stopPropagation();
  };

  // 2) bubbling-phase click
  const handleClick: React.MouseEventHandler<HTMLButtonElement> = (e) => {
    console.log("✅ SnapshotButton onClick (bubbling)");
    onClick?.(e);
  };

  return (
    <button
      type="button"
      {...rest}
      onMouseDown={handleMouseDown}
      onClick={handleClick}
      disabled={isDisabled}
      data-tour="snapshot-button"
      title={tooltipText}
      aria-label={tooltipText}
      className={clsx(
        "quick-action-chip",
        isDisabled && "opacity-50 cursor-not-allowed pointer-events-none",
        className
      )}
    >
      <BarChart3 size={16} strokeWidth={2} aria-hidden="true" />
      <span className="text-xs font-semibold">{label}</span>
    </button>
  );
};

export default SnapshotButton;
