// backend/server/routes/nba_game_features.js

import { getGameFeatures } from "../controllers/gameFeatures";
router.get("/api/games/:gameId/features", getGameFeatures);
