// frontend/src/components/shared/injury_report.tsx

import React from "react";
import SkeletonBox from "@/components/ui/skeleton_box";
import { Injury } from "@/api/use_injuries";
import { ChevronDown } from "lucide-react";

interface InjuryReportProps {
  displayDate: string;
  isPastDate: boolean;
  allGamesFilteredOut: boolean;
  isLoadingInjuries: boolean;
  injuriesError?: Error;
  teamsWithInjuries: string[];
  injuriesByTeam: Record<string, Injury[]>;
}

const InjuryReport: React.FC<InjuryReportProps> = ({
  displayDate,
  isPastDate,
  allGamesFilteredOut,
  isLoadingInjuries,
  injuriesError,
  teamsWithInjuries,
  injuriesByTeam,
}) => {
  // 1) Past-date or “all filtered out” messages
  if (isPastDate) {
    return (
      <p className="text-left text-sm text-text-secondary">
        Games have been completed. No injury statuses to report.
      </p>
    );
  }
  if (allGamesFilteredOut) {
    return (
      <p className="text-left text-sm text-text-secondary">
        No remaining games today. No injury statuses to report.
      </p>
    );
  }

  // 2) Loading / error / empty
  if (isLoadingInjuries) {
    return (
      <p className="text-left text-sm italic text-text-secondary">
        Loading injuries…
      </p>
    );
  }
  if (injuriesError) {
    return (
      <p className="text-left text-sm text-red-500">
        Could not load injury report.
      </p>
    );
  }
  if (teamsWithInjuries.length === 0) {
    return (
      <p className="text-left text-sm text-text-secondary">
        No significant injuries reported for playing teams on {displayDate}.
      </p>
    );
  }

  // 3) The actual list
  return (
    <div className="space-y-4">
      {teamsWithInjuries.map((team) => (
        <details key={team} className="app-card group">
          <summary className="flex cursor-pointer items-center justify-between gap-3 px-4 py-3 text-slate-800 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-green-400 dark:text-text-primary dark:hover:bg-gray-700/50">
            <span className="min-w-0 flex-1 font-medium">{team}</span>
            <span className="flex-shrink-0 rounded-full border border-green-500 px-2.5 py-1 text-xs font-medium text-green-800 shadow-md dark:text-green-100">
              {injuriesByTeam[team].length} available
            </span>
            <ChevronDown className="h-4 w-4 flex-shrink-0 transition-transform group-open:rotate-180" />
          </summary>
          <div className="mt-2 px-4 py-2">
            <ul className="space-y-1">
              {injuriesByTeam[team].map((inj) => (
                <li
                  key={inj.id}
                  className="flex items-start justify-between rounded-md pt-3 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                >
                  <div className="flex-1 pr-4">
                    <p className="font-medium text-slate-800 dark:text-text-primary">
                      {inj.player}
                    </p>
                    {inj.injury_type && (
                      <p className="mt-1 text-xs text-gray-500 dark:text-text-secondary">
                        {inj.injury_type}
                      </p>
                    )}
                  </div>
                  <span className="ml-auto mr-10 flex-shrink-0 rounded-full border border-gray-300 bg-gray-100 px-2.5 py-1 text-xs font-medium text-slate-800 dark:border-border dark:bg-transparent dark:text-text-primary">
                    {inj.status}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </details>
      ))}
    </div>
  );
};

export default InjuryReport;
