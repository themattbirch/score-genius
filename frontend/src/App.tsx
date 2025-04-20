// frontend/src/App.tsx

import React, { useState } from "react";
import { Routes, Route, Navigate, Outlet, useLocation } from "react-router-dom";

import GamesScreen from "./screens/game_screen";
import GameDetailScreen from "./screens/game_detail_screen";
import StatsScreen from "./screens/stats_screen";
import MoreScreen from "./screens/more_screen";
import HowToUseScreen from "./screens/how_to_use_screen";

import Header from './components/layout/Header';
import { SportProvider } from './contexts/sport_context';
import { DateProvider } from '@/contexts/date_context';
import BottomTabBar from "./components/layout/BottomTabBar";
import type { Sport } from '@/contexts/sport_context';

const Layout: React.FC = () => {
  const isGamesRoute = useLocation().pathname.startsWith("/games");
  return (
    <div className="flex h-screen flex-col">
      <Header showDatePicker={isGamesRoute} />

      <main className="flex-1 overflow-auto pb-14 lg:pb-0">
        <Outlet />
      </main>

      <BottomTabBar />
    </div>
  );
};

const App: React.FC = () => (
  <SportProvider>
    <DateProvider>
      <Routes>
        {/* App Shell with shared layout */}
        <Route element={<Layout />}>
          {/* Redirect root → /games */}
          <Route index element={<Navigate to="/games" replace />} />

          {/* Games list & detail */}
          <Route path="games" element={<GamesScreen />} />
          <Route path="games/:gameId" element={<GameDetailScreen />} />

          {/* Other tabs */}
          <Route path="stats" element={<StatsScreen />} />
          <Route path="more" element={<MoreScreen />} />
          <Route path="how-to-use" element={<HowToUseScreen />} />
        </Route>

        {/* Catch‑all fallback */}
        <Route path="*" element={<Navigate to="/games" replace />} />
      </Routes>
    </DateProvider>
  </SportProvider>
);

export default App;