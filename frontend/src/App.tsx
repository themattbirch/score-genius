// frontend/src/App.tsx

// frontend/src/App.tsx

import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Routes, Route, Navigate, Outlet, useLocation } from "react-router-dom";

import GamesScreen from "./screens/game_screen";
import GameDetailScreen from "./screens/game_detail_screen";
import StatsScreen from "./screens/stats_screen";
import MoreScreen from "./screens/more_screen";
import HowToUseScreen from "./screens/how_to_use_screen";
import { TourProvider } from "@/components/ui/joyride_tour";
import Header from "./components/layout/Header";
import { SportProvider } from "./contexts/sport_context";
import { DateProvider } from "@/contexts/date_context";
import BottomTabBar from "./components/layout/BottomTabBar";
import { ThemeProvider } from "./contexts/theme_context";

// 1) Create a single QueryClient instance
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 120 * 1000,
      retry: 1,
    },
  },
});

// 2) Layout stays the same
const Layout: React.FC = () => {
  const isGamesRoute = useLocation().pathname.startsWith("/games");
  return (
    <div className="flex h-screen flex-col">
      <Header />
      <main className="flex-1 overflow-auto pb-14 lg:pb-0">
        <Outlet />
      </main>
      <BottomTabBar />
    </div>
  );
};

// 3) Wrap everything in QueryClientProvider
const App: React.FC = () => (
  <QueryClientProvider client={queryClient}>
    {/* --- ADD THIS TEMPORARY LINE FOR BUILD VERIFICATION --- */}
    <div
      style={{
        color: "red",
        fontSize: "20px",
        position: "fixed",
        top: "10px",
        left: "10px",
        zIndex: 9999,
      }}
    >
      BUILD TEST: {Math.random().toFixed(4)}
    </div>
    {/* --- END TEMPORARY LINE --- */}
    <TourProvider>
      <ThemeProvider>
        <SportProvider>
          <DateProvider>
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<Navigate to="/games" replace />} />
                <Route path="games" element={<GamesScreen />} />
                <Route path="games/:gameId" element={<GameDetailScreen />} />
                <Route path="stats" element={<StatsScreen />} />
                <Route path="more" element={<MoreScreen />} />
                <Route path="how-to-use" element={<HowToUseScreen />} />
              </Route>
              <Route path="*" element={<Navigate to="/games" replace />} />
            </Routes>
          </DateProvider>
        </SportProvider>
      </ThemeProvider>
    </TourProvider>
  </QueryClientProvider>
);

export default App;
