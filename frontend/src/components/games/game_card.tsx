// frontend/src/components/games/game_card.tsx
import React from "react";
// Import BOTH game types from their respective hooks
import { Game as NBAGame } from "@/api/use_nba_schedule";
import { MLBGame } from "@/api/use_mlb_schedule";
// Assuming useInjuries hook and Injury type work for both sports or can be adapted
import { useInjuries, Injury } from "@/api/use_injuries";
import { useSport } from "@/contexts/sport_context";
import { useDate } from "@/contexts/date_context";

// Helper type guard to check if the game object has NBA properties
function isNBAGame(game: NBAGame | MLBGame): game is NBAGame {
  // Use a property unique to NBAGame, like 'tipoff' or 'predictionHome'
  // Make sure it's a reliable differentiator
  return (game as NBAGame).tipoff !== undefined;
}

// Update props to accept either game type
interface GameCardProps {
  game: NBAGame | MLBGame;
}

const GameCard: React.FC<GameCardProps> = ({ game }) => {
  const { sport } = useSport(); // Get current sport (might be useful for logic/fetching)
  const { date } = useDate(); // Get date for injury fetching
  const isoDate = date ? date.toISOString().slice(0, 10) : '';

  // Fetch injuries - TODO: Verify this hook works correctly for MLB team names/players
  const { data: injuries = [] } = useInjuries(sport, isoDate);

  // Determine common properties, handling different field names
  const gameId = isNBAGame(game) ? game.id : game.game_id;
  const homeTeamName = isNBAGame(game) ? game.homeTeam : game.home_team_name;
  const awayTeamName = isNBAGame(game) ? game.awayTeam : game.away_team_name;
  const gameTime = isNBAGame(game) ? game.tipoff : game.scheduled_time_utc;
  const status = isNBAGame(game) ? "Scheduled" : game.status_state; // Simplistic status mapping - refine if needed

  console.log(`📋 GameCard rendering for ${sport} game:`, gameId, isoDate);

  // Filter injuries - uses derived team names which *should* work if injury data uses same names
  // TODO: Double check injury data source uses full team names matching schedule data for both sports
  const teamInjuries = injuries.filter((inj: Injury) => {
      const injuryTeam = inj.team; // Assuming inj.team exists and holds the comparable team name
      return injuryTeam === homeTeamName || injuryTeam === awayTeamName;
  });

  return (
    <div className="app-card flex flex-col gap-2">
      {/* Top Row: Teams & Time / Primary Details */}
      <div className="flex items-center justify-between gap-4">
        {/* Left Side: Teams & Time */}
        <div className="min-w-0 flex-1">
          <p className="truncate font-semibold text-sm sm:text-base">
            {awayTeamName} @ {homeTeamName}
          </p>
          <p className="text-xs text-text-secondary">
            {/* Display formatted time */}
            {new Date(gameTime).toLocaleTimeString([], {
              hour: "numeric",
              minute: "2-digit",
            })}
            {/* Display status if not 'Scheduled' or similar default */}
            {status && !status.toLowerCase().includes('sched') && ` (${status})`}
          </p>
        </div>

        {/* Right Side: Sport-Specific Info (Predictions / Pitchers / Odds) */}
        <div className="flex-none text-right text-sm">
          {isNBAGame(game) ? (
            // --- NBA Specific Display ---
            <>
              <p className="font-medium">
                {/* Display predictions, using '-' as fallback */}
                {game.predictionAway ?? '-'} – {game.predictionHome ?? '-'}
              </p>
              <p className="text-xs text-[var(--color-text-secondary)]"> {/* Use secondary color for odds */}
                {/* Display odds, using 'N/A' as fallback */}
                Spread {game.spread ?? 'N/A'}, Total {game.total ?? 'N/A'}
              </p>
            </>
          ) : (
            // --- MLB Specific Display --- (If not NBA game)
            <>
              {/* Away Pitcher Line - Styled w/ Handedness */}
              <p className="text-xs font-normal text-[var(--color-text-secondary)] truncate max-w-[100px] sm:max-w-[150px]">
                {game.away_probable_pitcher_name ?? 'TBD Pitcher'}
                {game.away_probable_pitcher_handedness && ` (${game.away_probable_pitcher_handedness}HP)`}
              </p>
              {/* Home Pitcher Line - Styled w/ Handedness */}
              <p className="text-xs font-normal text-[var(--color-text-secondary)] truncate max-w-[100px] sm:max-w-[150px]">
                vs {game.home_probable_pitcher_name ?? 'TBD Pitcher'}
                {game.home_probable_pitcher_handedness && ` (${game.home_probable_pitcher_handedness}HP)`}
              </p>
              {/* TODO: Display MLB Odds (Moneyline, Spread, Total) if available */}
              {/* Example: Keep example styling consistent */}
              {/*
              {(game.moneyline_home_clean && game.moneyline_away_clean) && (
                <p className="text-xs text-[var(--color-text-secondary)]">
                  ML {game.moneyline_away_clean} / {game.moneyline_home_clean}
                </p>
              )}
              */}
            </>
          )}
        </div>
      </div>

      {/* Bottom Row: Injury Chips (Common Logic - check team name matching) */}
      {teamInjuries.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {teamInjuries.slice(0, 2).map((inj) => (
            <span
              key={`${gameId}-${inj.player}-${inj.status}`} // Make key more robust if inj.id isn't unique across fetches
              className="pill bg-brand-orange text-xs"
              title={`${inj.player}: ${inj.detail}`} // Show full name on hover
            >
              {/* Attempt to show last name */}
              {inj.player?.split(" ").pop()} {inj.status}
            </span>
          ))}
          {teamInjuries.length > 2 && (
            <span className="pill bg-brand-orange/60 text-xs">
              +{teamInjuries.length - 2} more
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default GameCard;