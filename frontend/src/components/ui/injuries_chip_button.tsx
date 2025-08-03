// frontend/src/components/ui/injuries_chip_button.tsx

import React from "react";

interface InjuriesChipButtonProps {
  onClick: (event: React.MouseEvent<HTMLButtonElement>) => void;
}

const InjuriesChipButton: React.FC<InjuriesChipButtonProps> = ({ onClick }) => {
  return (
    <button
      onClick={onClick}
      className="rounded-full border border-slate-300/80 bg-slate-50/80 px-4 py-1 text-xs font-medium text-slate-600 shadow-sm backdrop-blur-sm transition-colors hover:border-slate-400 hover:bg-slate-200/80 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-green-500 dark:border-slate-600/90 dark:bg-slate-800/80 dark:text-slate-300 dark:hover:border-slate-500 dark:hover:bg-slate-700/80"
    >
      Injuries
    </button>
  );
};

export default InjuriesChipButton;
