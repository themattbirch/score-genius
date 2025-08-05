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
    // Optional: log to your analytics service
    console.error("Caught in OfflineBoundary:", error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <iframe
          src="/app/offline.html"
          style={{ border: 0, width: "100%", height: "100%" }}
        />
      );
    }

    return this.props.children;
  }
}
