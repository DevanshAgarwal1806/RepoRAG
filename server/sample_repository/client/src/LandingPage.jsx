import { useEffect, useRef, useState } from "react";
import "./styles/styles.css";
import { supabase } from "./supabaseClient";

/* ─────────────────────────────────────────────
   DAG VISUALIZATION COMPONENT
───────────────────────────────────────────── */
const DAGVisualization = () => {
  const [activeNode, setActiveNode] = useState(0);

  const nodes = [
    { id: 0, label: "INPUT",   text: "user_prompt",    top: "8%",  left: "35%", delay: "0s" },
    { id: 1, label: "PHASE 1", text: "dag_generator",  top: "28%", left: "10%", delay: "0.1s" },
    { id: 2, label: "PHASE 1", text: "llm_judge",      top: "28%", left: "58%", delay: "0.2s", orange: true },
    { id: 3, label: "EVAL",    text: "deepeval_score", top: "50%", left: "35%", delay: "0.3s" },
    { id: 4, label: "PHASE 2", text: "tool_selector",  top: "68%", left: "5%",  delay: "0.4s" },
    { id: 5, label: "PHASE 3", text: "executor_node",  top: "68%", left: "55%", delay: "0.5s", orange: true },
    { id: 6, label: "PHASE 3", text: "reflexion_node", top: "86%", left: "35%", delay: "0.6s" },
  ];

  const edges = [
    [0, 1], [0, 2],
    [1, 3], [2, 3],
    [3, 4], [3, 5],
    [4, 6], [5, 6],
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveNode(prev => (prev + 1) % nodes.length);
    }, 1200);
    return () => clearInterval(interval);
  }, []);

  const getNodeCenter = (node) => {
    const leftVal = parseFloat(node.left) / 100;
    const topVal  = parseFloat(node.top)  / 100;
    return { x: leftVal * 460 + 70, y: topVal * 420 + 16 };
  };

  return (
    <div className="hero-dag">
      <svg className="dag-svg" viewBox="0 0 460 420">
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="3" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="rgba(0,245,212,0.4)" />
          </marker>
        </defs>
        {edges.map(([from, to], i) => {
          const a        = getNodeCenter(nodes[from]);
          const b        = getNodeCenter(nodes[to]);
          const isActive = activeNode === from || activeNode === to;
          return (
            <line
              key={i}
              x1={a.x} y1={a.y}
              x2={b.x} y2={b.y}
              stroke={isActive ? "rgba(0,245,212,0.6)" : "rgba(0,245,212,0.12)"}
              strokeWidth={isActive ? "1.5" : "1"}
              strokeDasharray={isActive ? "none" : "4,4"}
              markerEnd="url(#arrowhead)"
              style={{ transition: "all 0.4s" }}
            />
          );
        })}
      </svg>
      {nodes.map(node => (
        <div
          key={node.id}
          className={[
            "dag-node",
            node.orange            ? "orange" : "",
            activeNode === node.id ? "active" : "",
          ].filter(Boolean).join(" ")}
          style={{ top: node.top, left: node.left, animationDelay: node.delay }}
        >
          <span className="node-label">{node.label}</span>
          {node.text}
        </div>
      ))}
    </div>
  );
};

/* ─────────────────────────────────────────────
   GOOGLE SIGN-IN BUTTON
───────────────────────────────────────────── */
const GoogleSignInButton = () => {
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const handleSignIn = async () => {
    setLoading(true);
    setError(null);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        // After Google redirects back, Supabase lands the user here.
        // Change to your production domain when deploying.
        redirectTo: window.location.origin,
      },
    });
    if (error) {
      setError(error.message);
      setLoading(false);
    }
    // On success the browser redirects — no further action needed here.
  };

  return (
    <div className="google-auth-wrapper">
      <button
        className="btn-google"
        onClick={handleSignIn}
        disabled={loading}
        aria-label="Sign in with Google"
      >
        {/* Google "G" logo */}
        <svg
          className="google-icon"
          viewBox="0 0 48 48"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
          <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
          <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
          <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
        </svg>
        {loading ? "Redirecting…" : "Sign in with Google"}
      </button>
      {error && <p className="auth-error">{error}</p>}
    </div>
  );
};

/* ─────────────────────────────────────────────
   USER GREETING (shown when signed in)
───────────────────────────────────────────── */
const UserGreeting = ({ user, onSignOut }) => {
  const name   = user.user_metadata?.full_name
               || user.user_metadata?.name
               || user.email?.split("@")[0]
               || "there";
  const avatar = user.user_metadata?.avatar_url;

  return (
    <div className="user-greeting">
      {avatar && (
        <img
          src={avatar}
          alt={name}
          className="user-avatar"
          referrerPolicy="no-referrer"
        />
      )}
      <span className="user-name-text">
        Hello,&nbsp;<span className="user-name-accent">{name}</span>
      </span>
      <button className="btn-signout" onClick={onSignOut}>
        Sign out
      </button>
    </div>
  );
};

/* ─────────────────────────────────────────────
   STATIC DATA
───────────────────────────────────────────── */
const techStack = [
  { layer: "Inference (Planning)",  name: "Groq",        detail: "Llama 3 / 70B" },
  { layer: "Reasoning (Eval)",      name: "Gemini 2.0",  detail: "Flash — LLM Judge" },
  { layer: "Testing Framework",     name: "DeepEval",    detail: "Answer Relevancy" },
  { layer: "Orchestration",         name: "LangGraph",   detail: "State + DAG Engine" },
  { layer: "State Management",      name: "Custom JSON", detail: "Typed State Dict" },
  { layer: "Tooling Protocol",      name: "OpenAPI",     detail: "Swagger Ingestion" },
  { layer: "Search Intelligence",   name: "Tavily API",  detail: "Structured Web Intel" },
  { layer: "Ecosystem",             name: "LangChain",   detail: "Arxiv · GitHub · Wiki" },
];

const phases = [
  {
    num: "01",
    title: "Task Decomposition & Judge Graph",
    desc: "Groq generates a DAG in JSON. Gemini Flash 2.0 acts as the LLM Judge, scoring logic and feasibility via DeepEval. The generator and judge iterate until the Answer Relevancy score converges to high confidence.",
    tags: ["Groq LLaMA 70B", "Gemini Flash 2.0", "DeepEval", "DAG JSON"],
  },
  {
    num: "02",
    title: "Dynamic Tool Selection",
    desc: "Feed the agent any OpenAPI/Swagger spec and it instantly understands every endpoint — generating callable tools on the fly. Integrated with Arxiv, Wikipedia, GitHub, and Tavily for structured web intelligence.",
    tags: ["OpenAPI / Swagger", "Tavily API", "LangChain Community", "Auto-codegen"],
  },
  {
    num: "03",
    title: "Execution & Reflexion Graph",
    desc: "LangGraph executes the DAG step-by-step. The Reflexion Observer reads raw API responses, extracts task-specific values, and autonomously re-routes on failure — cycling until success without human input.",
    tags: ["LangGraph", "Executor Node", "Reflexion Node", "Auto-Recovery"],
  },
];

const flowSteps = [
  {
    title: "Prompt Ingestion",
    desc: "A complex user goal enters the system. SynapseAI decomposes it into a structured JSON DAG using Groq's Llama 3 70B, mapping every dependency and execution order.",
  },
  {
    title: "Judge Evaluation Loop",
    desc: "Gemini Flash 2.0 scrutinises the proposed DAG for logic and feasibility. DeepEval computes an Answer Relevancy score. The Generator and Judge iterate until convergence.",
  },
  {
    title: "Dynamic Tool Binding",
    desc: "Drop in any OpenAPI/Swagger spec — the agent reads it, understands every endpoint, and generates callable tools on the fly. No manual wiring required.",
  },
  {
    title: "Autonomous Execution",
    desc: "The Executor Node picks tasks in dependency order, calling the best available tool. Raw API responses flow directly into the shared LangGraph State.",
  },
  {
    title: "Reflexion & Recovery",
    desc: "The Observer Node reads the raw response, extracts task-specific values, and routes: success loops to the next task; failure triggers autonomous re-routing with an alternate tool.",
  },
];

const stats = [
  { num: "3", label: "Self-Correcting Phases" },
  { num: "∞", label: "Dynamic Tools via OpenAPI" },
  { num: "2", label: "LLM Inference Engines" },
  { num: "0", label: "Manual Nudges Required" },
];

const impacts = [
  {
    icon: "🧠",
    title: "Self-Evaluates",
    desc: "An LLM-as-a-Judge loop refines the plan before a single tool is called, catching logic errors early.",
  },
  {
    icon: "⚙️",
    title: "Self-Configures",
    desc: "Ingest any OpenAPI spec and the agent builds its own tool belt — no developer wiring needed.",
  },
  {
    icon: "🔄",
    title: "Self-Corrects",
    desc: "The Reflexion phase observes failures in real-time and autonomously re-routes logic without human nudging.",
  },
];

/* ─────────────────────────────────────────────
   MAIN LANDING PAGE
───────────────────────────────────────────── */
export default function LandingPage() {
  const flowRef = useRef(null);

  // null  = loading, false = signed out, object = signed-in user
  const [user,        setUser]        = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  /* ── Supabase auth listener ── */
  useEffect(() => {
    // Restore session from URL hash after Google redirect
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? false);
      setAuthLoading(false);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => setUser(session?.user ?? false)
    );
    return () => subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    setUser(false);
  };

  /* ── Canvas + scroll-reveal ── */
  useEffect(() => {
    const canvas = document.getElementById("synapse-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let W = (canvas.width  = window.innerWidth);
    let H = (canvas.height = window.innerHeight);
    let animId;

    const particles = Array.from({ length: 60 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5,
    }));

    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
        if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0,245,212,0.7)";
        ctx.fill();
      });
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx   = particles[i].x - particles[j].x;
          const dy   = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 130) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(0,245,212,${0.12 * (1 - dist / 130)})`;
            ctx.lineWidth   = 0.5;
            ctx.stroke();
          }
        }
      }
      animId = requestAnimationFrame(draw);
    };

    draw();

    const resize = () => {
      W = canvas.width  = window.innerWidth;
      H = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", resize);

    const observer = new IntersectionObserver(
      entries => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add("visible"); }),
      { threshold: 0.15 }
    );
    document.querySelectorAll(".flow-step").forEach(el => observer.observe(el));

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
      observer.disconnect();
    };
  }, []);

  return (
    <>
      <canvas id="synapse-canvas" />

      {/* ── NAV ── */}
      <nav>
        <a href="#" className="nav-logo">Synapse<span>AI</span></a>
        <ul className="nav-links">
          <li><a href="#phases">Architecture</a></li>
          <li><a href="#tech">Tech Stack</a></li>
          <li><a href="#flow">How It Works</a></li>
          <li><a href="#impact">Impact</a></li>
        </ul>

        {/* ── Auth slot (right side of nav) ── */}
        <div className="nav-auth">
          {authLoading ? (
            <span className="auth-loading">·  ·  ·</span>
          ) : user ? (
            <UserGreeting user={user} onSignOut={handleSignOut} />
          ) : (
            <GoogleSignInButton />
          )}
        </div>
      </nav>

      {/* ── HERO ── */}
      <section className="hero">
        <div className="hero-inner">
          <div>
            <div className="hero-badge">Autonomous Agentic Orchestrator</div>
            <h1 className="hero-title">
              <span className="accent">Synapse</span>AI<br />
              <span className="accent2">Self-Evolving</span><br />
              Agent Pipeline
            </h1>
            <p className="hero-sub">
              Decomposes complex human goals into executable DAGs. Self-evaluates,
              self-configures, and self-corrects — with minimal human oversight.
            </p>
            <div className="hero-cta">
              <a href="#flow"              className="btn-primary">Explore Architecture</a>
              <a href="https://github.com/DevanshAgarwal1806/AI_27" className="btn-ghost">View on GitHub</a>
            </div>
          </div>
          <DAGVisualization />
        </div>
      </section>

      {/* ── STATS BAR ── */}
      <div className="stats-bar">
        {stats.map((s, i) => (
          <div className="stat-item" key={i}>
            <span className="stat-num">{s.num}</span>
            <span className="stat-label">{s.label}</span>
          </div>
        ))}
      </div>

      {/* ── PHASES ── */}
      <section className="phases" id="phases">
        <div className="section-header">
          <span className="section-tag">// System Architecture</span>
          <h2 className="section-title">Three Phases. Zero Babysitting.</h2>
        </div>
        <div className="phases-grid">
          {phases.map((p, i) => (
            <div className="phase-card" key={i}>
              <div className="phase-number">{p.num}</div>
              <div className="phase-title">{p.title}</div>
              <p className="phase-desc">{p.desc}</p>
              <div className="phase-tags">
                {p.tags.map((t, j) => <span className="phase-tag" key={j}>{t}</span>)}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── TECH STACK ── */}
      <section className="tech" id="tech">
        <div className="section-header">
          <span className="section-tag">// Tech Stack</span>
          <h2 className="section-title">The Engine Under the Hood</h2>
        </div>
        <div className="tech-grid">
          {techStack.map((t, i) => (
            <div className="tech-card" key={i}>
              <div className="tech-dot" />
              <div className="tech-layer">{t.layer}</div>
              <div className="tech-name">{t.name}</div>
              <div className="tech-detail">{t.detail}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── EXECUTION FLOW ── */}
      <section className="flow" id="flow" ref={flowRef}>
        <div className="section-header">
          <span className="section-tag">// Execution Flow</span>
          <h2 className="section-title">How SynapseAI Thinks</h2>
        </div>
        <div className="flow-inner">
          {flowSteps.map((step, i) => (
            <div className="flow-step" key={i}>
              <div className="flow-num">{String(i + 1).padStart(2, "0")}</div>
              <div>
                <div className="flow-content-title">{step.title}</div>
                <p className="flow-content-desc">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── IMPACT ── */}
      <section className="impact" id="impact">
        <div className="section-header">
          <span className="section-tag">// Why It Matters</span>
        </div>
        <p className="impact-quote">
          SynapseAI transforms the user from a{" "}
          <span className="hl2">"babysitter"</span> into a{" "}
          <span className="hl">"supervisor."</span>
        </p>
        <div className="impact-cards">
          {impacts.map((c, i) => (
            <div className="impact-card" key={i}>
              <div className="impact-icon">{c.icon}</div>
              <div className="impact-card-title">{c.title}</div>
              <p className="impact-card-desc">{c.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="cta-section">
        <h2 className="cta-title">Ready to Deploy?</h2>
        <p className="cta-sub">Clone the repo. Drop in your API keys. Let SynapseAI take over.</p>
        <div className="cta-btns">
          <a href="https://github.com/DevanshAgarwal1806/AI_27" className="btn-primary">Clone on GitHub</a>
          <a href="#phases"            className="btn-ghost">Read the Docs</a>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer>
        <span>SynapseAI — Autonomous Agentic Orchestrator</span>
        <span>Built with LangGraph · Groq · Gemini · DeepEval</span>
      </footer>
    </>
  );
}