// frontend/src/components/offline_boundary.tsx
import React, { Component, ReactNode } from "react";

interface State {
  hasError: boolean;
}

export default class OfflineBoundary extends Component<
  { children: ReactNode },
  State
> {
  state: State = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: any) {
    console.error("Caught in ErrorBoundary:", error);
  }

  private retry = () => this.setState({ hasError: false });

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            display: "grid",
            placeItems: "center",
            height: "100%",
            padding: 16,
          }}
        >
          <div style={{ maxWidth: 560, textAlign: "center" }}>
            <h2>Something went wrong on this screen.</h2>
            <p>Try again. If it persists, check DevTools → Console/Network.</p>
            <button
              onClick={this.retry}
              style={{ padding: "8px 14px", marginTop: 12 }}
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
