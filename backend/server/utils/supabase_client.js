// backend/server/utils/supabase_client.js

import "dotenv/config"; // loads .env into process.env
import { createClient } from "@supabase/supabase-js";

// Load URL and SERVICE KEY from environment variables
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY;

// Debug logging
console.log(`DEBUG supabase_client.js: SUPABASE_URL=${supabaseUrl}`);
console.log(
  `DEBUG supabase_client.js: SUPABASE_SERVICE_KEY loaded? ${!!supabaseServiceKey}`
);

let supabase;

if (!supabaseUrl || !supabaseServiceKey) {
  if (process.env.NODE_ENV === "test") {
    console.warn("⚠️  Skipping Supabase client init in test environment.");
    // Minimal stub so imports don’t break
    supabase = {
      from: () => ({
        select: () => ({ data: null, error: null }),
      }),
      // add any other no-op methods you need for your tests…
    };
  } else {
    console.error(
      "FATAL ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY are required."
    );
    console.error(
      "Ensure your .env file is loaded and contains both variables."
    );
    process.exit(1);
  }
} else {
  // both vars present → real client
  supabase = createClient(supabaseUrl, supabaseServiceKey);
  console.log("Supabase client initialized with Service Role Key.");
}

export default supabase;
