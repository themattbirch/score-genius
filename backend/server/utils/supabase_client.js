//backend/server/utils/supabase_client.js

import { createClient } from '@supabase/supabase-js';

// dotenv config should be run in server.js before this module is imported
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error("FATAL ERROR: Supabase URL or Anon Key missing in environment variables. Make sure .env is loaded correctly in server.js.");
  process.exit(1); // Exit if essential config is missing
}

// Initialize client
const supabase = createClient(supabaseUrl, supabaseAnonKey);

console.log("Supabase client initialized (ESM).");

// Export the client instance using ES module syntax
export default supabase;