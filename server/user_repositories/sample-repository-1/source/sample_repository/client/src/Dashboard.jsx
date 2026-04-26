import { useEffect, useRef, useState } from "react";
import { supabase } from "./supabaseClient";
import "./styles/styles.css";

// ─── Lightweight Markdown renderer ───────────────────────────────────────────
function renderMarkdown(text) {
  const lines = text.split("\n");
  const elements = [];
  let i = 0;
  let keyCounter = 0;
  const key = () => keyCounter++;

  const parseInline = (str) => {
    // Bold+italic, bold, italic, inline code, strikethrough
    const parts = [];
    const regex = /(\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|~~(.+?)~~)/g;
    let last = 0;
    let m;
    while ((m = regex.exec(str)) !== null) {
      if (m.index > last) parts.push(str.slice(last, m.index));
      if (m[2]) parts.push(<strong key={key()}><em>{m[2]}</em></strong>);
      else if (m[3]) parts.push(<strong key={key()}>{m[3]}</strong>);
      else if (m[4]) parts.push(<em key={key()}>{m[4]}</em>);
      else if (m[5]) parts.push(<code key={key()} className="md-inline-code">{m[5]}</code>);
      else if (m[6]) parts.push(<del key={key()}>{m[6]}</del>);
      last = m.index + m[0].length;
    }
    if (last < str.length) parts.push(str.slice(last));
    return parts.length === 1 && typeof parts[0] === "string" ? parts[0] : parts;
  };

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    if (line.startsWith("```")) {
      const lang = line.slice(3).trim();
      const codeLines = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      elements.push(
        <div key={key()} className="md-code-block">
          {lang && <div className="md-code-lang">{lang}</div>}
          <pre><code>{codeLines.join("\n")}</code></pre>
        </div>
      );
      i++;
      continue;
    }

    // Headings
    const h3 = line.match(/^### (.+)/);
    const h2 = line.match(/^## (.+)/);
    const h1 = line.match(/^# (.+)/);
    if (h1) { elements.push(<h1 key={key()} className="md-h1">{parseInline(h1[1])}</h1>); i++; continue; }
    if (h2) { elements.push(<h2 key={key()} className="md-h2">{parseInline(h2[1])}</h2>); i++; continue; }
    if (h3) { elements.push(<h3 key={key()} className="md-h3">{parseInline(h3[1])}</h3>); i++; continue; }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
      elements.push(<hr key={key()} className="md-hr" />);
      i++;
      continue;
    }

    // Blockquote
    if (line.startsWith("> ")) {
      const quoteLines = [];
      while (i < lines.length && lines[i].startsWith("> ")) {
        quoteLines.push(lines[i].slice(2));
        i++;
      }
      elements.push(
        <blockquote key={key()} className="md-blockquote">
          {quoteLines.map((ql, qi) => <p key={qi}>{parseInline(ql)}</p>)}
        </blockquote>
      );
      continue;
    }

    // Unordered list
    if (/^(\s*[-*+] )/.test(line)) {
      const items = [];
      const baseIndent = line.match(/^(\s*)/)[1].length;
      while (i < lines.length && /^(\s*[-*+] )/.test(lines[i])) {
        const indent = lines[i].match(/^(\s*)/)[1].length;
        const content = lines[i].replace(/^\s*[-*+] /, "");
        items.push({ content, indent });
        i++;
      }
      const buildList = (itemArr) => (
        <ul className="md-ul">
          {itemArr.map((item, idx) => (
            <li key={idx} className="md-li">{parseInline(item.content)}</li>
          ))}
        </ul>
      );
      elements.push(<div key={key()}>{buildList(items)}</div>);
      continue;
    }

    // Ordered list
    if (/^\d+\. /.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\. /.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\. /, ""));
        i++;
      }
      elements.push(
        <ol key={key()} className="md-ol">
          {items.map((item, idx) => <li key={idx} className="md-li">{parseInline(item)}</li>)}
        </ol>
      );
      continue;
    }

    // Blank line → spacer
    if (line.trim() === "") {
      elements.push(<div key={key()} className="md-spacer" />);
      i++;
      continue;
    }

    // Plain paragraph
    elements.push(<p key={key()} className="md-p">{parseInline(line)}</p>);
    i++;
  }

  return elements;
}

// ─── Typing indicator dots ────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="chat-bubble assistant typing-bubble">
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </div>
  );
}

// ─── Typewriter hook ──────────────────────────────────────────────────────────
function useTypewriter(fullText, active, speed = 10) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);
  const rafRef = useRef(null);
  const idxRef = useRef(0);

  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    if (!active) { setDisplayed(fullText); setDone(true); return; }
    setDisplayed(""); setDone(false); idxRef.current = 0;
    let last = performance.now();
    const tick = (now) => {
      const el = now - last;
      if (el >= speed) {
        const steps = Math.max(1, Math.floor(el / speed));
        idxRef.current = Math.min(idxRef.current + steps, fullText.length);
        setDisplayed(fullText.slice(0, idxRef.current));
        last = now;
        if (idxRef.current >= fullText.length) { setDone(true); return; }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [fullText, active]);

  return { displayed, done };
}

// ─── Assistant bubble with typewriter + markdown ──────────────────────────────
function AssistantBubble({ text, animate }) {
  const { displayed, done } = useTypewriter(text, animate);
  return (
    <div className="chat-bubble assistant bubble-animate">
      <div className="bubble-role-label">SYNAPSE</div>
      <div className="bubble-text md-content">
        {done || !animate
          ? renderMarkdown(displayed)
          : (
            <>
              {displayed}
              <span className="cursor-blink">▌</span>
            </>
          )}
      </div>
    </div>
  );
}

// ── Reduced DAG node dimensions to fit more nodes ──────────────────────────
const DAG_NODE_WIDTH = 120;
const DAG_NODE_HEIGHT = 58;
const DAG_COL_GAP = 16;
const DAG_ROW_GAP = 40;
const DAG_PADDING_X = 14;
const DAG_PADDING_Y = 16;

function normalizeDagTasks(dag) {
  if (Array.isArray(dag)) return dag.filter(Boolean);
  if (!dag || typeof dag !== "object") return [];

  if (Array.isArray(dag.tasks)) {
    return dag.tasks.filter(Boolean);
  }

  if (Array.isArray(dag.nodes)) {
    return dag.nodes.filter(Boolean).map((node, index) => ({
      ...node,
      id: node.id ?? `step_${index + 1}`,
      dependencies: Array.isArray(node.dependencies) ? node.dependencies : [],
    }));
  }

  const entries = Object.entries(dag).filter(([, value]) => value && typeof value === "object");
  const looksLikeTaskMap = entries.some(([, value]) => "description" in value || "dependencies" in value);

  if (!looksLikeTaskMap) return [];

  return entries.map(([id, value]) => ({
    id,
    description: value.description ?? value.label ?? "",
    dependencies: Array.isArray(value.dependencies) ? value.dependencies : [],
  }));
}

function buildDagLayout(tasks) {
  const taskMap = new Map(
    tasks.map((task, index) => {
      const id = task.id ?? `step_${index + 1}`;
      return [
        id,
        {
          ...task,
          id,
          dependencies: Array.isArray(task.dependencies) ? task.dependencies : [],
        },
      ];
    }),
  );

  const children = new Map();
  const indegree = new Map();

  taskMap.forEach((_task, id) => {
    indegree.set(id, 0);
    children.set(id, []);
  });

  taskMap.forEach((task, id) => {
    task.dependencies.forEach((depId) => {
      if (!taskMap.has(depId)) return;
      indegree.set(id, (indegree.get(id) ?? 0) + 1);
      children.get(depId)?.push(id);
    });
  });

  const queue = [];
  indegree.forEach((count, id) => {
    if (count === 0) queue.push(id);
  });

  const depthById = {};
  const orderedIds = [];

  while (queue.length > 0) {
    const id = queue.shift();
    orderedIds.push(id);
    const depth = depthById[id] ?? 0;

    children.get(id)?.forEach((childId) => {
      depthById[childId] = Math.max(depthById[childId] ?? 0, depth + 1);
      indegree.set(childId, (indegree.get(childId) ?? 1) - 1);
      if (indegree.get(childId) === 0) queue.push(childId);
    });
  }

  taskMap.forEach((_task, id) => {
    if (!orderedIds.includes(id)) {
      orderedIds.push(id);
      depthById[id] = depthById[id] ?? 0;
    }
  });

  const layers = [];
  orderedIds.forEach((id) => {
    const layerIndex = depthById[id] ?? 0;
    if (!layers[layerIndex]) layers[layerIndex] = [];
    layers[layerIndex].push(id);
  });

  const positions = {};
  const maxColumns = Math.max(...layers.map((layer) => layer?.length ?? 0), 1);

  layers.forEach((layer, layerIndex) => {
    if (!layer?.length) return;
    const rowWidth = layer.length * DAG_NODE_WIDTH + Math.max(0, layer.length - 1) * DAG_COL_GAP;
    const totalWidth = maxColumns * DAG_NODE_WIDTH + Math.max(0, maxColumns - 1) * DAG_COL_GAP;
    const startX = DAG_PADDING_X + Math.max(0, (totalWidth - rowWidth) / 2);

    layer.forEach((id, index) => {
      positions[id] = {
        left: startX + index * (DAG_NODE_WIDTH + DAG_COL_GAP),
        top: DAG_PADDING_Y + layerIndex * (DAG_NODE_HEIGHT + DAG_ROW_GAP),
      };
    });
  });

  return {
    positions,
    orderedTasks: orderedIds.map((id) => taskMap.get(id)).filter(Boolean),
    width:
      DAG_PADDING_X * 2 +
      maxColumns * DAG_NODE_WIDTH +
      Math.max(0, maxColumns - 1) * DAG_COL_GAP,
    height:
      DAG_PADDING_Y * 2 +
      layers.length * DAG_NODE_HEIGHT +
      Math.max(0, layers.length - 1) * DAG_ROW_GAP,
  };
}

// ─── Visual DAG Graph ─────────────────────────────────────────────────────────
function DagGraph({ dag }) {
  const containerRef = useRef(null);
  const tasks = normalizeDagTasks(dag);
  const hasData = tasks.length > 0;
  const { orderedTasks, positions, width, height } = buildDagLayout(tasks);

  if (!hasData) {
    return (
      <div className="dag-empty">
        <div className="dag-empty-icon">◈</div>
        <p>No DAG yet</p>
        <p className="dag-empty-sub">Send a prompt to see the execution graph</p>
      </div>
    );
  }

  return (
    <div className="dag-visual-wrap" ref={containerRef} style={{ "--dag-width": `${width}px`, "--dag-height": `${height}px` }}>
      <svg className="dag-svg-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMinYMin meet">
        {orderedTasks.map((step, i) => {
          const stepId = step.id ?? `step_${i + 1}`;
          const targetPos = positions[stepId];
          if (!targetPos || !step.dependencies?.length) return null;
          return step.dependencies.map((depId) => {
            const sourcePos = positions[depId];
            if (!sourcePos) return null;
            const sourceX = sourcePos.left + DAG_NODE_WIDTH / 2;
            const sourceY = sourcePos.top + DAG_NODE_HEIGHT;
            const targetX = targetPos.left + DAG_NODE_WIDTH / 2;
            const targetY = targetPos.top;
            const midY = sourceY + (targetY - sourceY) / 2;
            const path = `M ${sourceX} ${sourceY} C ${sourceX} ${midY}, ${targetX} ${midY}, ${targetX} ${targetY}`;
            return <path key={`${depId}->${stepId}`} d={path} className="dag-edge" />;
          });
        })}
      </svg>
      <div className="dag-nodes-container" style={{ width: `${width}px`, height: `${height}px` }}>
        {orderedTasks.map((step, i) => {
          const stepId = step.id ?? `step_${i + 1}`;
          const position = positions[stepId];
          return (
            <div key={stepId} data-id={stepId} className="dag-node"
              style={{ left: `${position?.left ?? 0}px`, top: `${position?.top ?? 0}px` }}>
              <div className="dag-node-header">
                <span className="dag-node-id">{stepId}</span>
              </div>
              {step.description && <div className="dag-node-desc">{step.description}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Chat messages area ───────────────────────────────────────────────────────
function ChatArea({ messages, loading, animateLastAssistant }) {
  const bottomRef = useRef(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="chat-welcome">
        <div className="chat-welcome-icon">⬡</div>
        <h3 className="chat-welcome-title">Give me the Query and let's get started!</h3>
        <p className="chat-welcome-sub">Type your task. SynapseAI orchestrates agents and shows a live execution DAG.</p>
      </div>
    );
  }

  return (
    <div className="chat-messages">
      {messages.map((msg, i) => {
        if (msg.role === "assistant") {
          return <AssistantBubble key={i} text={msg.text} animate={i === messages.length - 1 && animateLastAssistant} />;
        }
        return (
          <div key={i} className="chat-bubble user bubble-animate" style={{ animationDelay: `${i * 25}ms` }}>
            <div className="bubble-role-label">YOU</div>
            <div className="bubble-text">{msg.text}</div>
          </div>
        );
      })}
      {loading && <TypingIndicator />}
      <div ref={bottomRef} />
    </div>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const [user, setUser]               = useState(null);
  const [chats, setChats]             = useState([]);
  const [activeChat, setActiveChat]   = useState(null);
  const [prompt, setPrompt]           = useState("");
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [animateNew, setAnimateNew]   = useState(false);
  const [deletingId, setDeletingId]   = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const textareaRef                   = useRef(null);

  const chatIsComplete = !!(activeChat?.messages?.some((m) => m.role === "assistant"));

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setUser(session?.user ?? null));
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_e, s) => setUser(s?.user ?? null));
    return () => subscription.unsubscribe();
  }, []);

  useEffect(() => {
    const load = async () => {
      if (!user?.id) { setChats([]); setActiveChat(null); return; }
      const { data, error: err } = await supabase
        .from("chat_history")
        .select("id,user_prompt,final_output,current_dag,created_at")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false });
      if (err) { setError("Failed to load history"); return; }
      if (Array.isArray(data)) {
        const mapped = data.map((r) => ({
          id: r.id, prompt: r.user_prompt,
          final_output: r.final_output, current_dag: r.current_dag, createdAt: r.created_at,
        }));
        setChats(mapped);
        if (mapped.length > 0) openHistoryChat(mapped[0]);
      }
    };
    load();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const openHistoryChat = (row) => {
    setActiveChat({
      id: row.id,
      messages: [
        { role: "user",      text: row.prompt },
        { role: "assistant", text: row.final_output || "No output returned." },
      ],
      dag: row.current_dag || {},
    });
    setAnimateNew(false); setError(null);
  };

  const handleNewChat = () => {
    setActiveChat(null); setPrompt(""); setError(null); setAnimateNew(false);
    setTimeout(() => textareaRef.current?.focus(), 50);
  };

  const handleDelete = async (e, chatId) => {
    e.stopPropagation();
    setDeletingId(chatId);
    const { error: delErr } = await supabase.from("chat_history").delete().eq("id", chatId);
    if (delErr) { setError("Failed to delete chat"); setDeletingId(null); return; }
    setChats((prev) => prev.filter((c) => c.id !== chatId));
    if (activeChat?.id === chatId) { setActiveChat(null); setPrompt(""); }
    setDeletingId(null);
  };

  const handleSubmit = async () => {
    const trimmed = prompt.trim();
    if (!trimmed || loading || chatIsComplete) return;

    const userMsg = { role: "user", text: trimmed };
    setActiveChat((prev) => ({ id: prev?.id ?? null, messages: [...(prev?.messages ?? []), userMsg], dag: prev?.dag ?? {} }));
    setPrompt(""); setLoading(true); setError(null); setAnimateNew(false);

    try {
      const res = await fetch("http://localhost:8000/run", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_prompt: trimmed }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || "Backend error"); }

      const data         = await res.json();
      const assistantMsg = { role: "assistant", text: data.final_output || "Done." };
      const dag          = data.current_dag || {};

      const { data: insertData, error: insertErr } = await supabase
        .from("chat_history")
        .insert([{ user_id: user?.id ?? null, user_email: user?.email ?? null, user_prompt: trimmed, final_output: data.final_output ?? "", current_dag: dag }])
        .select("id,user_prompt,final_output,current_dag,created_at").single();

      if (insertErr) setError("Saved but failed to persist history.");

      const newRow = { id: insertData?.id ?? Date.now(), prompt: trimmed, final_output: data.final_output || "", current_dag: dag, createdAt: insertData?.created_at ?? new Date().toISOString() };
      setChats((prev) => [newRow, ...prev]);
      setAnimateNew(true);
      setActiveChat((prev) => ({ id: newRow.id, messages: [...(prev?.messages ?? [userMsg]), assistantMsg], dag }));
    } catch (err) {
      setError(err.message || "Something went wrong.");
      setActiveChat((prev) => ({ ...prev, messages: [...(prev?.messages ?? []), { role: "assistant", text: "⚠ " + (err.message || "Error") }] }));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); } };

  const currentDag      = activeChat?.dag ?? {};
  const currentMessages = activeChat?.messages ?? [];
  const dagTasks        = normalizeDagTasks(currentDag);

  return (
    <>
      <canvas id="synapse-canvas" />
      <nav>
        <a href="/" className="nav-logo">Synapse<span>AI</span></a>
        <button className="btn-signout" onClick={() => supabase.auth.signOut()}>Sign out</button>
      </nav>

      <main className={`db-layout ${sidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>

        <div className="sidebar-toggle-strip" onClick={() => setSidebarOpen((v) => !v)} title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}>
          <div className="toggle-strip-lines">
            <span /><span /><span />
          </div>
          <div className={`toggle-arrow ${sidebarOpen ? "arrow-left" : "arrow-right"}`}>
            {sidebarOpen ? "‹" : "›"}
          </div>
        </div>

        <aside className={`db-sidebar ${sidebarOpen ? "sidebar-visible" : "sidebar-hidden"}`}>
          <div className="sidebar-header">
            <span className="sidebar-title">History</span>
            <button className="btn-new-chat" onClick={handleNewChat} title="New Chat">+</button>
          </div>
          <div className="sidebar-list">
            {chats.length === 0 && <div className="sidebar-empty">No history yet</div>}
            {chats.map((chat) => (
              <div
                key={chat.id}
                className={`sidebar-item ${activeChat?.id === chat.id ? "active" : ""}`}
                onClick={() => openHistoryChat(chat)}
              >
                <div className="sidebar-item-inner">
                  <div className="sidebar-item-title">
                    {chat.prompt.slice(0, 32)}{chat.prompt.length > 32 ? "…" : ""}
                  </div>
                  <div className="sidebar-item-time">
                    {new Date(chat.createdAt).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </div>
                </div>
                <button
                  className={`btn-delete ${deletingId === chat.id ? "deleting" : ""}`}
                  onClick={(e) => handleDelete(e, chat.id)}
                  title="Delete chat"
                  disabled={deletingId === chat.id}
                >
                  {deletingId === chat.id
                    ? <span className="delete-spinner" />
                    : <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
                        <path d="M1 1L11 11M11 1L1 11" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                      </svg>}
                </button>
              </div>
            ))}
          </div>
        </aside>

        <section className="db-chat">
          <div className="db-chat-header">
            <div className="db-header-left">
              <span className="live-dot" />
              <span className="db-chat-title">
                {activeChat
                  ? currentMessages[0]?.text.slice(0, 46) + (currentMessages[0]?.text.length > 46 ? "…" : "")
                  : "New Chat"}
              </span>
            </div>
            {loading && <span className="db-processing-badge">● Processing</span>}
          </div>

          <div className="db-chat-body">
            <ChatArea messages={currentMessages} loading={loading} animateLastAssistant={animateNew} />
          </div>

          {!chatIsComplete && (
            <div className="db-chat-input-wrap">
              {error && <div className="error-message">{error}</div>}
              <div className="db-input-row">
                <textarea
                  ref={textareaRef}
                  className="prompt-input db-textarea"
                  placeholder="Describe your task… (Enter to send)"
                  value={prompt} rows={2}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={loading}
                />
                <button className="btn-send" onClick={handleSubmit} disabled={loading || !prompt.trim()}>
                  {loading
                    ? <span className="send-spinner" />
                    : <svg viewBox="0 0 20 20" fill="none" width="16" height="16">
                        <path d="M2 10L18 2L11 18L9 11L2 10Z" fill="currentColor" />
                      </svg>}
                </button>
              </div>
            </div>
          )}

          {chatIsComplete && !loading && (
            <div className="db-chat-locked">
              {error && <div className="error-message" style={{ marginBottom: "0.4rem" }}>{error}</div>}
              <div className="locked-msg">
                <span className="locked-icon">✓</span>
                Session complete
                <button className="btn-new-chat-inline" onClick={handleNewChat}>+ New chat</button>
              </div>
            </div>
          )}
        </section>

        <section className="db-dag">
          <div className="db-dag-header">
            <div className="db-header-left">
              <span className="db-dag-title">Execution DAG</span>
              {dagTasks.length > 0 && (
                <span className="dag-live-badge"><span className="dag-live-dot" />LIVE</span>
              )}
            </div>
            <span className="dag-node-count">
              {dagTasks.length > 0 ? `${dagTasks.length} node${dagTasks.length !== 1 ? "s" : ""}` : "-"}
            </span>
          </div>
          <div className="db-dag-body">
            <DagGraph dag={currentDag} />
          </div>
        </section>

      </main>

      <style>{`
        /* ── Layout ── */
        .db-layout {
          position: relative; z-index: 1;
          display: grid;
          height: 100vh; padding-top: 60px; overflow: hidden;
          transition: grid-template-columns 0.3s cubic-bezier(0.4,0,0.2,1);
        }
        .db-layout.sidebar-open  { grid-template-columns: 14px 200px 1fr 340px; }
        .db-layout.sidebar-closed { grid-template-columns: 14px 0px 1fr 1fr; }

        /* ── Sidebar toggle strip ── */
        .sidebar-toggle-strip {
          display: flex; flex-direction: column; align-items: center; justify-content: center;
          gap: 10px; width: 14px; background: rgba(6,7,12,0.98);
          border-right: 1px solid rgba(0,245,212,0.15); cursor: pointer; z-index: 10;
          transition: background 0.18s, border-color 0.18s; position: relative; overflow: hidden;
        }
        .sidebar-toggle-strip:hover { background: rgba(0,245,212,0.06); border-color: rgba(0,245,212,0.4); }
        .sidebar-toggle-strip:hover .toggle-strip-lines span { background: rgba(0,245,212,0.7); }
        .sidebar-toggle-strip:hover .toggle-arrow { color: var(--cyan); opacity: 1; }
        .toggle-strip-lines { display: flex; flex-direction: column; gap: 3px; align-items: center; }
        .toggle-strip-lines span { display: block; width: 4px; height: 4px; border-radius: 50%; background: rgba(0,245,212,0.25); transition: background 0.18s, transform 0.18s; }
        .toggle-arrow { font-size: 10px; color: rgba(0,245,212,0.3); opacity: 0.6; transition: color 0.18s, opacity 0.18s; line-height: 1; user-select: none; }

        /* ── Sidebar ── */
        .db-sidebar {
          display: flex; flex-direction: column;
          background: rgba(6,7,12,0.98); border-right: 1px solid var(--border);
          overflow: hidden; transition: width 0.3s cubic-bezier(0.4,0,0.2,1), opacity 0.25s ease;
        }
        .db-sidebar.sidebar-hidden { width: 0; opacity: 0; pointer-events: none; border-right: none; }
        .db-sidebar.sidebar-visible { width: 200px; opacity: 1; }
        .sidebar-header { display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 0.75rem 0.6rem; border-bottom: 1px solid var(--border); flex-shrink: 0; white-space: nowrap; }
        .sidebar-title { font-family: var(--font-display); font-size: 0.58rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--muted); }
        .btn-new-chat { width: 22px; height: 22px; border-radius: 50%; border: 1px solid var(--border); background: transparent; color: var(--cyan); font-size: 0.9rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.18s; }
        .btn-new-chat:hover { background: rgba(0,245,212,0.12); border-color: var(--cyan); box-shadow: 0 0 10px rgba(0,245,212,0.3); }
        .sidebar-list { flex: 1; overflow-y: auto; padding: 0.4rem; display: flex; flex-direction: column; gap: 0.2rem; }
        .sidebar-list::-webkit-scrollbar { width: 2px; }
        .sidebar-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
        .sidebar-empty { font-family: var(--font-mono); font-size: 0.65rem; color: var(--muted); padding: 0.85rem 0.4rem; text-align: center; }
        .sidebar-item { display: flex; align-items: center; gap: 0.3rem; padding: 0.5rem 0.5rem; border: 1px solid transparent; border-radius: 7px; cursor: pointer; transition: all 0.16s; white-space: nowrap; }
        .sidebar-item:hover { background: rgba(0,245,212,0.05); border-color: rgba(0,245,212,0.15); }
        .sidebar-item.active { background: rgba(0,245,212,0.09); border-color: rgba(0,245,212,0.32); }
        .sidebar-item-inner { flex: 1; min-width: 0; }
        .sidebar-item-title { font-family: var(--font-mono); font-size: 0.68rem; color: var(--white); line-height: 1.35; margin-bottom: 0.15rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .sidebar-item.active .sidebar-item-title { color: var(--cyan); }
        .sidebar-item-time { font-family: var(--font-mono); font-size: 0.56rem; color: var(--muted); }
        .btn-delete { flex-shrink: 0; width: 18px; height: 18px; border-radius: 4px; border: none; background: transparent; color: var(--muted); display: flex; align-items: center; justify-content: center; cursor: pointer; opacity: 0; transition: all 0.15s; }
        .sidebar-item:hover .btn-delete { opacity: 1; }
        .btn-delete:hover { background: rgba(255,80,80,0.15); color: #ff6b6b; }
        .btn-delete.deleting { opacity: 1; }
        .delete-spinner { width: 8px; height: 8px; border: 1.5px solid transparent; border-top-color: var(--muted); border-radius: 50%; animation: spin 0.6s linear infinite; }

        /* ── Chat panel ── */
        .db-chat { display: flex; flex-direction: column; background: rgba(5,5,8,0.92); border-right: 1px solid var(--border); overflow: hidden; }
        .db-chat-header { padding: 0.7rem 1rem; border-bottom: 1px solid var(--border); flex-shrink: 0; background: rgba(7,8,14,0.65); backdrop-filter: blur(10px); display: flex; align-items: center; justify-content: space-between; }
        .db-header-left { display: flex; align-items: center; gap: 0.45rem; }
        .live-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; background: #00f5d4; box-shadow: 0 0 5px #00f5d4; animation: livePulse 2s ease-in-out infinite; }
        @keyframes livePulse { 0%,100% { opacity: 1; box-shadow: 0 0 5px #00f5d4; } 50% { opacity: 0.35; box-shadow: 0 0 12px #00f5d4; } }
        .db-chat-title { font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.06em; color: var(--muted); }
        .db-processing-badge { font-family: var(--font-mono); font-size: 0.55rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--cyan); border: 1px solid rgba(0,245,212,0.28); border-radius: 20px; padding: 0.15rem 0.5rem; animation: badgeFade 1.3s ease-in-out infinite; }
        @keyframes badgeFade { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
        .db-chat-body { flex: 1; overflow-y: auto; padding: 1rem 1.25rem; scroll-behavior: smooth; }
        .db-chat-body::-webkit-scrollbar { width: 2px; }
        .db-chat-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        /* Welcome */
        .chat-welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; text-align: center; padding: 2rem; gap: 0.6rem; }
        .chat-welcome-icon { font-size: 2rem; color: var(--cyan); opacity: 0.35; margin-bottom: 0.4rem; }
        .chat-welcome-title { font-family: var(--font-display); font-size: 0.9rem; font-weight: 700; letter-spacing: 0.08em; color: var(--white); }
        .chat-welcome-sub { font-family: var(--font-mono); font-size: 0.7rem; color: var(--muted); max-width: 340px; line-height: 1.6; }

        /* Messages */
        .chat-messages { display: flex; flex-direction: column; gap: 0.7rem; }
        .bubble-animate { animation: bubblePop 0.25s cubic-bezier(0.34,1.56,0.64,1) both; }
        @keyframes bubblePop { from { opacity: 0; transform: translateY(8px) scale(0.97); } to { opacity: 1; transform: translateY(0) scale(1); } }
        .chat-bubble { max-width: 80%; border-radius: 12px; padding: 0.6rem 0.85rem; font-family: var(--font-mono); font-size: 0.78rem; line-height: 1.55; border: 1px solid transparent; }
        .chat-bubble.user { align-self: flex-end; text-align: right; background: rgba(0,158,206,0.13); border-color: rgba(0,158,206,0.32); }
        .chat-bubble.assistant { align-self: flex-start; text-align: left; background: rgba(0,245,212,0.06); border-color: rgba(0,245,212,0.2); }
        .bubble-role-label { font-size: 0.52rem; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.25rem; font-weight: 700; }
        .chat-bubble.user .bubble-role-label { color: rgba(0,158,206,0.75); }
        .chat-bubble.assistant .bubble-role-label { color: var(--cyan-dim); }
        .bubble-text { color: var(--white); white-space: pre-wrap; }
        .cursor-blink { display: inline-block; color: var(--cyan); animation: cursorBlink 0.5s step-end infinite; margin-left: 1px; }
        @keyframes cursorBlink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

        /* Typing dots */
        .typing-bubble { display: flex; align-items: center; gap: 4px; padding: 0.7rem 0.9rem; min-width: 52px; }
        .typing-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--cyan); animation: typingBounce 1.2s ease infinite; }
        .typing-dot:nth-child(2) { animation-delay: 0.18s; }
        .typing-dot:nth-child(3) { animation-delay: 0.36s; }
        @keyframes typingBounce { 0%,60%,100% { transform: translateY(0); opacity: 0.35; } 30% { transform: translateY(-5px); opacity: 1; } }

        /* Input area */
        .db-chat-input-wrap { padding: 0.6rem 0.85rem; border-top: 1px solid var(--border); background: rgba(7,8,14,0.88); flex-shrink: 0; }
        .db-input-row { display: flex; align-items: flex-end; gap: 0.5rem; }
        .db-textarea { flex: 1; min-height: 50px; max-height: 140px; resize: vertical; border-radius: 9px; font-size: 0.77rem; }
        .btn-send { width: 38px; height: 38px; border-radius: 9px; border: 1px solid var(--cyan); background: var(--cyan); color: var(--bg); display: flex; align-items: center; justify-content: center; cursor: pointer; flex-shrink: 0; transition: all 0.18s; }
        .btn-send:hover:not(:disabled) { background: var(--white); box-shadow: 0 0 18px rgba(0,245,212,0.4); }
        .btn-send:disabled { opacity: 0.38; cursor: not-allowed; }
        .send-spinner { width: 12px; height: 12px; border: 2px solid transparent; border-top-color: var(--bg); border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Locked footer */
        .db-chat-locked { padding: 0.65rem 1rem; border-top: 1px solid var(--border); background: rgba(7,8,14,0.88); flex-shrink: 0; }
        .locked-msg { display: flex; align-items: center; gap: 0.45rem; font-family: var(--font-mono); font-size: 0.7rem; color: var(--muted); }
        .locked-icon { color: var(--cyan); opacity: 0.75; font-size: 0.8rem; }
        .btn-new-chat-inline { margin-left: auto; padding: 0.28rem 0.7rem; border-radius: 7px; border: 1px solid rgba(0,245,212,0.32); background: rgba(0,245,212,0.07); color: var(--cyan); font-family: var(--font-mono); font-size: 0.68rem; cursor: pointer; transition: all 0.18s; }
        .btn-new-chat-inline:hover { background: rgba(0,245,212,0.16); border-color: var(--cyan); box-shadow: 0 0 10px rgba(0,245,212,0.2); }

        /* ── DAG panel ── */
        .db-dag { display: flex; flex-direction: column; background: rgba(6,7,12,0.98); overflow: hidden; }
        .db-dag-header { display: flex; align-items: center; justify-content: space-between; padding: 0.7rem 0.9rem; border-bottom: 1px solid var(--border); flex-shrink: 0; background: rgba(7,8,14,0.65); backdrop-filter: blur(10px); }
        .db-dag-title { font-family: var(--font-display); font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); }
        .dag-live-badge { display: flex; align-items: center; gap: 0.28rem; font-family: var(--font-mono); font-size: 0.5rem; letter-spacing: 0.15em; color: #00f5d4; border: 1px solid rgba(0,245,212,0.28); border-radius: 20px; padding: 0.12rem 0.4rem; }
        .dag-live-dot { width: 4px; height: 4px; border-radius: 50%; background: #00f5d4; animation: livePulse 1.5s ease-in-out infinite; }
        .dag-node-count { font-family: var(--font-mono); font-size: 0.58rem; color: var(--cyan); letter-spacing: 0.1em; }
        .db-dag-body { flex: 1; overflow-y: auto; padding: 0.6rem 0.5rem 2rem; position: relative; }
        .db-dag-body::-webkit-scrollbar { width: 2px; }
        .db-dag-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        /* DAG empty */
        .dag-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; text-align: center; gap: 0.4rem; padding: 1.5rem; }
        .dag-empty-icon { font-size: 1.6rem; color: var(--muted); opacity: 0.35; margin-bottom: 0.35rem; }
        .dag-empty p { font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted); }
        .dag-empty-sub { font-size: 0.62rem !important; opacity: 0.55; }

        /* DAG visual */
        .dag-visual-wrap { position: relative; min-height: 100%; overflow: auto; }
        .dag-svg-overlay { position: absolute; top: 0; left: 0; width: var(--dag-width, 100%); height: var(--dag-height, 100%); pointer-events: none; z-index: 0; }
        .dag-edge { stroke: var(--cyan, #00f5d4); stroke-width: 1; fill: none; stroke-dasharray: 3 3; opacity: 0.55; animation: dashFlow 20s linear infinite; }
        @keyframes dashFlow { to { stroke-dashoffset: -1000; } }
        .dag-nodes-container { position: relative; z-index: 1; }
        .dag-node { position: absolute; background: rgba(7,8,14,0.95); border: 1px solid rgba(0,245,212,0.22); border-radius: 6px; padding: 0.35rem 0.5rem; width: 120px; min-height: 58px; max-width: 120px; display: flex; flex-direction: column; justify-content: center; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.3); transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s; }
        .dag-node:hover { transform: translateY(-2px); border-color: var(--cyan, #00f5d4); box-shadow: 0 4px 12px rgba(0,245,212,0.15); }
        .dag-node-header { margin-bottom: 0.15rem; }
        .dag-node-id { font-family: var(--font-mono); font-size: 0.6rem; font-weight: 700; color: var(--cyan, #00f5d4); letter-spacing: 0.04em; }
        .dag-node-desc { font-family: var(--font-mono); font-size: 0.54rem; color: var(--white); opacity: 0.7; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

        /* ── Markdown content styles ── */
        .md-content { white-space: normal; }
        .md-content .md-h1 {
          font-family: var(--font-display);
          font-size: 1.05rem; font-weight: 800;
          color: var(--cyan); letter-spacing: 0.04em;
          margin: 0.9rem 0 0.45rem;
          padding-bottom: 0.3rem;
          border-bottom: 1px solid rgba(0,245,212,0.2);
          line-height: 1.3;
        }
        .md-content .md-h2 {
          font-family: var(--font-display);
          font-size: 0.88rem; font-weight: 700;
          color: #7ff5e8; letter-spacing: 0.04em;
          margin: 0.75rem 0 0.35rem;
          padding-left: 0.5rem;
          border-left: 2px solid rgba(0,245,212,0.5);
          line-height: 1.35;
        }
        .md-content .md-h3 {
          font-family: var(--font-display);
          font-size: 0.78rem; font-weight: 600;
          color: rgba(0,245,212,0.75); letter-spacing: 0.05em;
          text-transform: uppercase;
          margin: 0.6rem 0 0.28rem;
          line-height: 1.35;
        }
        .md-content .md-p {
          margin: 0.22rem 0;
          color: var(--white);
          line-height: 1.65;
          font-size: 0.78rem;
        }
        .md-content .md-ul,
        .md-content .md-ol {
          margin: 0.25rem 0 0.25rem 1.1rem;
          padding: 0;
        }
        .md-content .md-ul { list-style: none; }
        .md-content .md-ul .md-li {
          position: relative;
          padding-left: 0.9rem;
          margin: 0.2rem 0;
          color: var(--white);
          font-size: 0.78rem;
          line-height: 1.6;
        }
        .md-content .md-ul .md-li::before {
          content: "▸";
          position: absolute; left: 0;
          color: var(--cyan); font-size: 0.6rem;
          top: 0.22rem;
        }
        .md-content .md-ol { list-style: decimal; }
        .md-content .md-ol .md-li {
          margin: 0.2rem 0;
          color: var(--white);
          font-size: 0.78rem;
          line-height: 1.6;
          padding-left: 0.2rem;
        }
        .md-content .md-ol .md-li::marker { color: var(--cyan); font-weight: 700; }
        .md-content .md-inline-code {
          font-family: var(--font-mono);
          font-size: 0.72rem;
          background: rgba(0,245,212,0.1);
          border: 1px solid rgba(0,245,212,0.22);
          border-radius: 3px;
          padding: 0.05em 0.35em;
          color: #7ff5e8;
        }
        .md-content .md-code-block {
          margin: 0.55rem 0;
          border: 1px solid rgba(0,245,212,0.18);
          border-radius: 7px;
          overflow: hidden;
          background: rgba(0,0,0,0.35);
        }
        .md-content .md-code-lang {
          font-family: var(--font-mono);
          font-size: 0.52rem; letter-spacing: 0.14em; text-transform: uppercase;
          color: var(--cyan); opacity: 0.65;
          padding: 0.28rem 0.6rem;
          background: rgba(0,245,212,0.07);
          border-bottom: 1px solid rgba(0,245,212,0.12);
        }
        .md-content .md-code-block pre {
          margin: 0; padding: 0.6rem 0.75rem;
          font-family: var(--font-mono);
          font-size: 0.7rem; line-height: 1.55;
          color: #c8ffe9; overflow-x: auto;
          white-space: pre;
        }
        .md-content .md-blockquote {
          margin: 0.4rem 0;
          padding: 0.35rem 0.75rem;
          border-left: 2px solid rgba(0,245,212,0.4);
          background: rgba(0,245,212,0.04);
          border-radius: 0 5px 5px 0;
        }
        .md-content .md-blockquote p { margin: 0.1rem 0; color: rgba(255,255,255,0.7); font-style: italic; }
        .md-content .md-hr {
          border: none;
          border-top: 1px solid rgba(0,245,212,0.18);
          margin: 0.6rem 0;
        }
        .md-content .md-spacer { height: 0.3rem; }
        .md-content strong { color: #fff; font-weight: 700; }
        .md-content em { color: rgba(0,245,212,0.85); font-style: italic; }
        .md-content del { opacity: 0.45; text-decoration: line-through; }

        /* First child margin fix */
        .md-content > *:first-child { margin-top: 0 !important; }

        /* Responsive */
        @media (max-width: 860px) {
          .db-layout { grid-template-columns: 1fr !important; grid-template-rows: auto auto 1fr auto; overflow-y: auto; }
          .sidebar-toggle-strip { display: none; }
          .db-sidebar { width: 100% !important; height: 180px; border-right: none; border-bottom: 1px solid var(--border); }
          .db-sidebar.sidebar-hidden { height: 0; width: 100% !important; }
          .db-dag { height: 340px; border-top: 1px solid var(--border); }
        }
      `}</style>
    </>
  );
} 