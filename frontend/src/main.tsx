import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // Import the placeholder App component
// No CSS needed for this test
// import './index.css';

// Standard React 18 rendering logic
const rootElement = document.getElementById('root');
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} else {
  console.error("Failed to find the root element. Check your index.html file.");
}