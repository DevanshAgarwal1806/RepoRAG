### USER QUERY: What does this codebase do?

### CODEBASE CONTEXT

PRIMARY MATCH: `compute_embeddings`

File: `server/sample_repository/backend/core_engine/fast_math.cpp`

Code:
```
std::vector<float> compute_embeddings(const std::vector<std::string>& texts) {
    const int EMBEDDING_DIM = 768;
    std::vector<float> embeddings(EMBEDDING_DIM, 0.0f);
    
    if (texts.empty()) {
        return embeddings;
    }

    // Iterate through all provided texts to build a composite embedding
    for (size_t t = 0; t < texts.size(); ++t) {
        const std::string& text = texts[t];
        
        for (size_t i = 0; i < text.length(); ++i) {
            // Distribute the ASCII value across the 768 dimensions using sine/cosine waves
            // This creates a deterministic, non-uniform distribution of floats
            int dim_index = (i * 31 + t) % EMBEDDING_DIM;
            float char_weight = static_cast<float>(text[i]);
            
            if (i % 2 == 0) {
                embeddings[dim_index] += std::sin(char_weight * 0.1f);
            } else {
                embeddings[dim_index] += std::cos(char_weight * 0.1f);
            }
        }
    }

    // Normalize the final embedding vector using L2 normalization
    float sum_squares = 0.0f;
    for (float val : embeddings) {
        sum_squares += val * val;
    }
    
    if (sum_squares > 0.0f) {
        float magnitude = std::sqrt(sum_squares);
        for (float& val : embeddings) {
            val /= magnitude;
        }
    }

    return embeddings;
}
```

PRIMARY MATCH: `Dashboard`

File: `server/sample_repository/frontend/modern_dash/Dashboard.tsx`

Code:
```
export const Dashboard: React.FC<Props> = ({ streamEndpoint }) => {
    const [events, setEvents] = useState<TelemetryEvent[]>([]);
    const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        setConnectionStatus('connecting');
        const ws = new WebSocket(streamEndpoint);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log(`Successfully connected to ${streamEndpoint}`);
            setConnectionStatus('connected');
        };

        ws.onmessage = (messageEvent) => {
            try {
                const newEvent: TelemetryEvent = JSON.parse(messageEvent.data);
                setEvents(prevEvents => {
                    // Keep only the latest 100 events in memory to prevent browser lag
                    const updated = [...prevEvents, newEvent];
                    return updated.length > 100 ? updated.slice(updated.length - 100) : updated;
                });
            } catch (error) {
                console.error("Failed to parse incoming telemetry data:", error);
            }
        };

        ws.onerror = (error) => {
            console.error("WebSocket encountered an error:", error);
            setConnectionStatus('disconnected');
        };

        ws.onclose = () => {
            console.log("WebSocket connection closed.");
            setConnectionStatus('disconnected');
        };

        // Cleanup function on component unmount
        return () => {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.close();
            }
        };
    }, [streamEndpoint]);

    return (
        <div className="dashboard-container">
            <h1>Live Telemetry Analytics</h1>
            <div className={`status-indicator ${connectionStatus}`}>
                Status: {connectionStatus}
            </div>
            <p>Active Events in Memory: {events.length}</p>
            {/* Charting components would be rendered here using the 'events' state */}
        </div>
    );
};
```

PRIMARY MATCH: `isValidEvent`

File: `server/sample_repository/frontend/modern_dash/types.ts`

Code:
```
export function isValidEvent(event: TelemetryEvent): boolean {
    return event.sensorReadings.length > 0 && event.timestamp > 0;
}
```

PRIMARY MATCH: `UserWidget`

File: `server/sample_repository/frontend/legacy_portal/UserWidget.jsx`

Code:
```
const UserWidget = ({ username, avatarUrl }) => {
    return (
        <div className="widget-card">
            <img src={avatarUrl} alt={`${username}'s avatar`} />
            <h3>@{username}</h3>
        </div>
    );
};
```

