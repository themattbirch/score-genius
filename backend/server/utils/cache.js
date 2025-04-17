// backend/server/utils/cache.js
import NodeCache from "node-cache"; // Use import

// Create a new cache instance
const cache = new NodeCache({ stdTTL: 600, checkperiod: 120 });

console.log("Cache instance created.");

export default cache; // Use export default
