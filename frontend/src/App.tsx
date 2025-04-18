import React from "react";
// No CSS import needed for this barebones test
// import './App.css';

function App() {
  // Get current timestamp for easy deployment verification
  const deployTime = new Date().toLocaleString();

  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h1>Score Genius - Deployment Test</h1>
      <p>
        If you see this, the frontend pipeline successfully built and deployed!
      </p>
      <p>Test deployment timestamp: {deployTime}</p>
    </div>
  );
}

export default App;
