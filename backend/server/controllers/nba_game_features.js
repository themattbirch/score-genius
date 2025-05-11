// backend/server/controllers/nba_game_features.js

import { supabase } from "../caching/supabase_client";

const NodeCache = require("node-cache");
const cache = new NodeCache({ stdTTL: 300 }); // 300 seconds = 5 minutes

const { supabase } = require("../../caching/supabase_client");

exports.getNbaGameFeatures = async (req, res) => {
  const { gameId } = req.params;
  const cacheKey = `nba_game_features_${gameId}`;

  // 1. Try to pull from cache
  const cached = cache.get(cacheKey);
  if (cached) {
    return res.json({ features: cached, source: "cache" });
  }

  // 2. If cache miss, fetch from Supabase
  const { data, error } = await supabase
    .from("game_features")
    .select("*")
    .eq("game_id", gameId)
    .single();

  if (error) {
    return res.status(500).json({ error: error.message });
  }

  // 3. Store in cache, then return
  cache.set(cacheKey, data);
  res.json({ features: data, source: "supabase" });
};
