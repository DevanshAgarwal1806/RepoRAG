### USER QUERY: What logic is used to order and position nodes in a directed acyclic graph layout to minimize overlap and optimize visual clarity?

### CODEBASE CONTEXT

PRIMARY MATCH: `dfs`

File: `server/sample_repository/server/src/graph/nodes/evaluator.py`

Code:
```
def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True 
                
        rec_stack.remove(node)
        return False
```

PRIMARY MATCH: `buildDagLayout`

File: `server/sample_repository/client/src/Dashboard.jsx`

Code:
```
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
```

PRIMARY MATCH: `normalizeDagTasks`

File: `server/sample_repository/client/src/Dashboard.jsx`

Code:
```
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
```

PRIMARY MATCH: `DagGraph`

File: `server/sample_repository/client/src/Dashboard.jsx`

Code:
```
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
```

PRIMARY MATCH: `DAGVisualization`

File: `server/sample_repository/client/src/LandingPage.jsx`

Code:
```
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
```

PRIMARY MATCH: `build_synapse_graph`

File: `server/sample_repository/server/src/graph/workflow.py`

Code:
```
def build_synapse_graph():
    builder = StateGraph(SynapseState)

    # Nodes
    builder.add_node("generator", generate_dag)
    builder.add_node("evaluator", evaluate_dag)
    builder.add_node("executor", execute_task)
    builder.add_node("reflector", reflect_on_execution)
    builder.add_node("synthesizer", synthesize_output)  

    # Entry point
    builder.set_entry_point("generator")

    # Phase 1 loop
    builder.add_edge("generator", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        route_evaluation,
        {
            "generate": "generator",
            "execute": "executor",
        }
    )

    # Phase 3 loop: executor → reflector → router → synthesizer → back or done
    builder.add_edge("executor", "reflector")
    builder.add_conditional_edges(
        "reflector",
        route_execution,
        {
            "next_task": "executor",   # success, still tasks left
            "retry":     "executor",   # failure, retry same task
            "done":      "synthesizer",          # all tasks complete
            "fail":      "synthesizer",          # task exhausted retries
        }
    )

    #Synthesizer always ends the DAG
    builder.add_edge("synthesizer",END)
    return builder.compile()
```

PRIMARY MATCH: `check_for_cycles`

File: `server/sample_repository/server/src/graph/nodes/evaluator.py`

Code:
```
def check_for_cycles(tasks: list) -> bool:
    """
    Returns True if a cycle is detected in the task dependencies, False otherwise.
    Uses Depth-First Search (DFS).
    """
    graph = {task["id"]: task.get("dependencies", []) for task in tasks}
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True 
                
        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False
```

