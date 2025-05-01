// frontend/src/api/use_game_predictions.ts
import { useQuery } from "@tanstack/react-query"
import { supabase } from "../lib/supabaseClient"  // adjust if your path differs

export interface GamePrediction {
  game_id: number
  predicted_home_score: number
  predicted_away_score: number
  spread: number
  total: number
}

export function useGamePredictions({
  sport,
  date,
  enabled,
}: {
  sport: "NBA" | "MLB"
  date: Date
  enabled?: boolean
}) {
  const table = `${sport.toLowerCase()}_game_schedule`
  const gameDate = date.toISOString().slice(0, 10)

  return useQuery<GamePrediction[], Error>({
    queryKey: ["game_predictions", sport, gameDate],
    queryFn: async () => {
      const { data, error } = await supabase
        .from(table)
        .select(
          "game_id, predicted_home_score, predicted_away_score, spread, total"
        )
        .eq("game_date", gameDate)

      if (error) throw error
      return (data ?? []) as GamePrediction[]
    },
    enabled,
  })
}
