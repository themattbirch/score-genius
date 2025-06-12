# ScoreGenius “Snapshots + Weather” Feature
**Product Requirements Document – v2.0**  
*Date: 11 Jun 2025*

---

## 1. Purpose
Enhance the existing **Games** screen with two contextual add-ons:

1. **Snapshots** – interactive, stats-rich modal for every NBA & MLB game.  
2. **Weather Badge** – ballpark weather stub on MLB cards, paving the way for live data later.

---

## 2. Goals & Objectives
| Goal                                      | Success Metric                                                  |
| ----------------------------------------- | --------------------------------------------------------------- |
| Expose pre-computed Snapshot insights     | ≥ 60% of daily active users open ≥ 1 snapshot                  |
| Maintain fast initial render              | Games tab FCP ≤ 1.2 s (p75 mobile)                              |
| Prepare for future ballpark weather       | WeatherBadge visible on 100% of MLB cards                      |
| Bundle weight restraint                   | +≤ 40 KB gzip for new JS (lazy-loaded charts excluded)         |

---

## 3. Scope

### In-Scope
- UI changes inside **Games** tab  
- Snapshot modal, chart components, and skeleton loaders  
- MLB WeatherBadge stub (static)  
- API calls to existing `/api/v1/{sport}/snapshots/:gameId`

### Out-of-Scope
- Backend scheduling of snapshot generation  
- Live weather ingestion; stub only  
- Desktop breakpoints (mobile-first for v2.0)

---

## 4. Functional Requirements

### 4.1 Game Card
| ID       | Requirement                                                            |
| -------- | ----------------------------------------------------------------------- |
| FR-GC-1  | Every card shows a **SnapshotButton** below matchup text               |
| FR-GC-2  | MLB cards also show **WeatherBadge** under predicted score             |
| FR-GC-3  | Card height can expand up to +20 px to accommodate new row             |

### 4.2 Snapshot Modal
| ID       | Requirement                                                            |
| -------- | ----------------------------------------------------------------------- |
| FR-SM-1  | Opens full-screen overlay (`fixed inset-0`, z-index 50)               |
| FR-SM-2  | Fetches snapshot JSON (SWR, 120 s revalidate)                          |
| FR-SM-3  | Displays: HeadlineGrid → BarChart → RadarChart → PieChart             |
| FR-SM-4  | Close via X icon or swipe-down gesture                                  |
| FR-SM-5  | Modal remembers last scroll position when reopened                     |

### 4.3 WeatherBadge (Stub)
| ID       | Requirement                                                            |
| -------- | ----------------------------------------------------------------------- |
| FR-WB-1  | Shows “WX” icon + “— °F / — mph”                                      |
| FR-WB-2  | Tooltip: “Ballpark weather coming soon”                                |
| FR-WB-3  | Badge remains in stub state until live data arrives                    |

---

## 5. Technical Design

### 5.1 Component Tree
```plaintext
<GameCard>
  ├─ MatchupInfo
  ├─ ScorePrediction
  ├─ SnapshotButton
  └─ WeatherBadge (MLB only)

# 5. Technical Design

### 5.2 Libraries
- **React + Tailwind** (existing stack)
- **React Query** – caching + SWR
- **Recharts** – bar, radar, pie (tree-shaken)

### 5.3 Data Flow
1. Button click → `useSnapshot(gameId, sport)` query
2. If cache miss → loader skeleton
3. Charts get memoized JSON slices

### 5.4 Performance
- Charts lazy-loaded:
  ```js
  import(/* webpackChunkName: "charts" */ './Charts')
- Modal unmounts on close to free memory
- Support prefers-reduced-motion

# 6. UX / UI Specifications

| Element          | Styles                                                       |
| :--------------- | :----------------------------------------------------------- |
| **SnapshotButton** | rounded-full, bg-btn-snapshot, text-xs font-semibold, hover:opacity-90 |
| **WeatherBadge** | rounded-full, bg-badge-weather, icon + text, 32 px height  |
| **Modal** | bg-black/80 backdrop-blur; content max-w-sm mx-auto          |
| **Charts** | 1:1 aspect ratio, max-height: 240 px, responsive           |

# 7. Analytics & Telemetry

* **snapshot_open event:** (sport, gameId, time_to_render)
* **weather\_badge\_seen event for MLB cards**
* **Funnel:** impression → snapshot open → modal close

# 8. Risks & Mitigations

| Risk                 | Mitigation                                       |
| :------------------- | :----------------------------------------------- |
| Large JS chunk       | Lazy-import charts                               |
| Missing snapshot JSON| Disable button with tooltip “Snapshot not ready” |
| Modal memory leak    | Unmount chart subtree on close                   |

# 9. Milestones

| Date   | Deliverable                                            |
| :----- | :----------------------------------------------------- |
| Jun 14 | Tailwind tokens, SnapshotButton & WeatherBadge stub merged |
| Jun 18 | SnapshotModal w/ skeleton + headline grid              |
| Jun 22 | Bar/Radar/Pie charts integrated                        |
| Jun 24 | QA on Android & iOS                                    |
| Jun 26 | Production deploy                                      |

# 10. Open Questions

* Should modal support swipe-left/right to cycle games’ snapshots?
* WeatherBadge color – brand-green or amber?
* Log snapshot view duration for deeper engagement insights?
* Copy
