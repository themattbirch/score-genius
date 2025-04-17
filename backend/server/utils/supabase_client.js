// backend/server/utils/supabase_client.js
import { createClient } from "@supabase/supabase-js";

// Load URL and SERVICE KEY from environment variables
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY; // Make sure this matches .env

// --- Debug Logging (Optional but helpful) ---
console.log(`DEBUG supabase_client.js: Initializing with URL = ${supabaseUrl}`);
console.log(
  `DEBUG supabase_client.js: Service Key Found = ${!!supabaseServiceKey}`
);
// --- End Debug ---

// --- CORRECTED Validation Check ---
// Check if BOTH URL and the SERVICE key were loaded successfully
if (!supabaseUrl || !supabaseServiceKey) {
  console.error(
    "FATAL ERROR: Supabase URL or SUPABASE_SERVICE_KEY missing from environment variables."
  );
  console.error(
    "Ensure your .env file is loaded correctly in server.js and contains both variables."
  );
  process.exit(1); // Exit if essential config is missing
}
// --- End Correction ---

// Initialize client using the Service Key
const supabase = createClient(supabaseUrl, supabaseServiceKey);

console.log("Supabase client initialized with Service Role Key."); // Updated log message

// Export the client instance
export default supabase;