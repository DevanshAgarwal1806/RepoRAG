
import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL  = "https://zbjejvcjpligzgcdhvyu.supabase.co";   
const SUPABASE_ANON = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpiamVqdmNqcGxpZ3pnY2Rodnl1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM1MDcxMzYsImV4cCI6MjA4OTA4MzEzNn0.SLnC3BIrxkLirUNdBbmg5zpDGJY_VlDS4qXI_7IGFVE";                  

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON);