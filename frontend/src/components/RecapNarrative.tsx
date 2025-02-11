// frontend/src/components/RecapNarrative.tsx
import React from "react";

interface RecapNarrativeProps {
  narrative: string;
}

const RecapNarrative: React.FC<RecapNarrativeProps> = ({ narrative }) => {
  return (
    <div className="recap-narrative">
      <p>{narrative}</p>
    </div>
  );
};

export default RecapNarrative;
