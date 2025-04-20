import React from "react";
import { useParams } from "react-router-dom"; // Import useParams

const GameDetailScreen: React.FC = () => {
  // Example of getting the gameId param
  const { gameId } = useParams<{ gameId: string }>();

  return <div>Game Detail Screen Placeholder for Game ID: {gameId}</div>;
};

export default GameDetailScreen;
