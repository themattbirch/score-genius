// frontend/src/App.tsx

import React, { Suspense, memo } from "react";
import { Routes, Route, Navigate, Outlet } from "react-router-dom";

// Lazy‑load each screen for route‑based code splitting
const GamesScreen = React.lazy(() => import("./screens/game_screen"));
const GameDetailScreen = React.lazy(
  () => import("./screens/game_detail_screen")
);
const StatsScreen = React.lazy(() => import("./screens/stats_screen"));
const MoreScreen = React.lazy(() => import("./screens/more_screen"));
const HowToUseScreen = React.lazy(() => import("./screens/how_to_use_screen"));

// Lazy‑load TourProvider for code splitting
const LazyTourProvider = React.lazy(() =>
  import("@/components/ui/joyride_tour").then((mod) => ({
    default: mod.TourProvider,
  }))
);

// Context & layout imports
import Header from "./components/layout/Header";
import { SportProvider } from "./contexts/sport_context";
import { DateProvider } from "@/contexts/date_context";
import BottomTabBar from "./components/layout/BottomTabBar";
import { ThemeProvider } from "./contexts/theme_context";

// Memoized Layout: re‑renders only when outlet changes
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

// Tiny loader for Suspense fallback
const Loader: React.FC<{ message?: string }> = ({ message = "Loading…" }) => (
  <div className="flex h-screen items-center justify-center text-xs text-gray-400">
    {message}
  </div>
);

const App: React.FC = () => (
  <ThemeProvider>
    <SportProvider>
      <Suspense fallback={<></>}>
        <LazyTourProvider>
          <DateProvider>
            <Suspense fallback={<Loader />}>
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
            </Suspense>
          </DateProvider>
        </LazyTourProvider>
      </Suspense>
    </SportProvider>
  </ThemeProvider>
);

export default App;
