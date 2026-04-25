import { useEffect, useState } from "react";
import { supabase } from "./supabaseClient";
import LandingPage from "./LandingPage";   // your existing file
import Dashboard   from "./Dashboard";

export default function App() {
  const [user,        setUser]        = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      setAuthLoading(false);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setUser(session?.user ?? null);
        setAuthLoading(false);
      }
    );
    return () => subscription.unsubscribe();
  }, []);

  if (authLoading) return null; // or a splash screen

  // Signed-in → Dashboard, otherwise → Landing
  return user ? <Dashboard /> : <LandingPage />;
}