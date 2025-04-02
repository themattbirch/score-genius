// frontend/src/components/LoadingSpinner.tsx
import React from "react";

const LoadingSpinner: React.FC = () => {
  return (
    <div className="loading-spinner">
      <div className="spinner"></div>
      <style>{`
        .loading-spinner {
          display: flex;
          justify-content: center;
          align-items: center;
          height: 100%;
        }
        .spinner {
          border: 4px solid rgba(0,0,0,0.1);
          width: 36px;
          height: 36px;
          border-radius: 50%;
          border-left-color: #1e90ff;
          animation: spin 1s ease infinite;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default LoadingSpinner;
