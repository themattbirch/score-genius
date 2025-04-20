import React, { useState } from "react";
import { Routes, Route, Navigate, Outlet, useLocation } from "react-router-dom";

import GamesScreen from "./screens/game_screen";
import GameDetailScreen from "./screens/game_detail_screen";
import StatsScreen from "./screens/stats_screen";
import MoreScreen from "./screens/more_screen";
import HowToUseScreen from "./screens/how_to_use_screen";

import Header, { Sport } from "./components/layout/Header";
import BottomTabBar from "./components/layout/BottomTabBar";

const Layout: React.FC = () => {
  const [sport, setSport] = useState<Sport>("NBA");
  const [selectedDate] = useState<Date>(new Date());
  const isGamesRoute = useLocation().pathname.startsWith("/games");

  return (
    <div className="flex h-screen flex-col">
      <Header
        sport={sport}
        onSportChange={setSport}
        showDatePicker={isGamesRoute}
        selectedDate={selectedDate}
        onDateChange={() => {
          /* TODO */
        }}
      />

      <main className="flex-1 overflow-auto pb-14 lg:pb-0">
        <Outlet />
      </main>

      <BottomTabBar />
    </div>
  );
};

const App: React.FC = () => (
  <Routes>
    {/* Shell */}
    <Route element={<Layout />}>
      {/* Default → /games */}
      <Route index element={<Navigate to="/games" replace />} />

      {/* Games */}
      <Route path="games" element={<GamesScreen />} />
      <Route path="games/:gameId" element={<GameDetailScreen />} />

      {/* Stats & More */}
      <Route path="stats" element={<StatsScreen />} />
      <Route path="more" element={<MoreScreen />} />

      {/* How To Use */}
      <Route path="how-to-use" element={<HowToUseScreen />} />
    </Route>

    {/* Global fallback */}
    <Route path="*" element={<Navigate to="/games" replace />} />
  </Routes>
);

export default App;
