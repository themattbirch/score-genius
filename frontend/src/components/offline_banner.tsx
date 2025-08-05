// frontend/src/components/offline_banner.tsx
import React from "react";

interface OfflineBannerProps {
  message?: string;
}

const OfflineBanner: React.FC<OfflineBannerProps> = ({
  message = "⚠️ You’re offline",
}) => (
  <div className="fixed top-0 inset-x-0 bg-yellow-100 text-yellow-800 text-center py-2 z-50">
    <span>{message}</span>
    <button
      onClick={() => window.location.reload()}
      className="underline ml-2"
    >
      Retry
    </button>
  </div>
);

export default OfflineBanner;
