// frontend/src/App.tsx
import React, { Suspense, memo } from "react";
import { Routes, Route, Navigate, Outlet } from "react-router-dom";

// ——————————————————————————————————————————————————————————————
// 1️⃣ EAGER‑LOAD the main, above‑the‑fold screen:
// ——————————————————————————————————————————————————————————————
import GamesScreen from "./screens/game_screen";

// ——————————————————————————————————————————————————————————————
// 2️⃣ LAZY‑LOAD everything else (below‑the‑fold):
// ——————————————————————————————————————————————————————————————
const GameDetailScreen = React.lazy(
  () => import("./screens/game_detail_screen")
);
const StatsScreen = React.lazy(() => import("./screens/stats_screen"));
const MoreScreen = React.lazy(() => import("./screens/more_screen"));
const HowToUseScreen = React.lazy(() => import("./screens/how_to_use_screen"));

// ——————————————————————————————————————————————————————————————
// Layout, Contexts & UI chrome (always eager)
// ——————————————————————————————————————————————————————————————
import Header from "./components/layout/Header";
import BottomTabBar from "./components/layout/BottomTabBar";
import { SportProvider } from "./contexts/sport_context";
import { DateProvider } from "@/contexts/date_context";
import { ThemeProvider } from "./contexts/theme_context";
import { TourProvider } from "@/components/ui/joyride_tour";

// Memoized wrapper to avoid needless re‑renders
const Layout: React.FC = memo(() => (
  <div className="layout-container flex h-screen flex-col">
    <Header />
    <main className="flex flex-col flex-1 overflow-auto pb-14 lg:pb-0">
      <Outlet />
    </main>
    <BottomTabBar />
  </div>
));
Layout.displayName = "Layout";

// A tiny fallback for lazy screens
const Loader: React.FC<{ message?: string }> = ({ message = "Loading…" }) => (
  <div className="flex h-screen items-center justify-center text-xs text-gray-400">
    {message}
  </div>
);

const App: React.FC = () => (
  <ThemeProvider>
    <SportProvider>
      <TourProvider>
        <DateProvider>
          <Suspense fallback={<Loader />}>
            <Routes>
              <Route element={<Layout />}>
                {/* redirect root → /games */}
                <Route index element={<Navigate to="/games" replace />} />

                {/* GAMES: eager, no suspense delay */}
                <Route path="games" element={<GamesScreen />} />

                {/* everything else wrapped in its own Suspense */}
                <Route
                  path="games/:gameId"
                  element={
                    <Suspense fallback={<Loader message="Loading game…" />}>
                      <GameDetailScreen />
                    </Suspense>
                  }
                />
                <Route
                  path="stats"
                  element={
                    <Suspense fallback={<Loader message="Loading stats…" />}>
                      <StatsScreen />
                    </Suspense>
                  }
                />
                <Route
                  path="more"
                  element={
                    <Suspense fallback={<Loader message="Loading more…" />}>
                      <MoreScreen />
                    </Suspense>
                  }
                />
                <Route
                  path="how-to-use"
                  element={
                    <Suspense fallback={<Loader message="Loading help…" />}>
                      <HowToUseScreen />
                    </Suspense>
                  }
                />
              </Route>

              {/* catch‑all → back to /games */}
              <Route path="*" element={<Navigate to="/games" replace />} />
            </Routes>
          </Suspense>
        </DateProvider>
      </TourProvider>
    </SportProvider>
  </ThemeProvider>
);

export default App;
