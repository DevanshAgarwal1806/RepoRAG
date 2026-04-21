### USER QUERY: Show me the enterprise Java class responsible for ingesting streams.

### CODEBASE CONTEXT

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

PRIMARY MATCH: `detect_outliers`

File: `server/sample_repository/backend/py_analytics/anomaly_detector.py`

Code:
```
def detect_outliers(data_stream: list[float], threshold: float = 2.5) -> list[int]:
    """
    Scans a time-series data stream and identifies indices of anomalous data points
    using a simple z-score thresholding mechanism.
    
    Args:
        data_stream: A list of float values representing sensor readings.
        threshold: The z-score limit for an anomaly.
        
    Returns:
        List of indices where outliers occur.
    """
    if not data_stream:
        return []
    
    mean = sum(data_stream) / len(data_stream)
    variance = sum((x - mean) ** 2 for x in data_stream) / len(data_stream)
    std_dev = variance ** 0.5
    
    outliers = []
    for i, val in enumerate(data_stream):
        if std_dev > 0 and abs(val - mean) / std_dev > threshold:
            outliers.append(i)
            
    return outliers
```

PRIMARY MATCH: `StreamProcessor.normalizeSchema`

File: `server/sample_repository/backend/java_service/src/main/java/com/polydata/StreamProcessor.java`

Code:
```
private String normalizeSchema(String payload) {
        String processedPayload = payload;

        // Ensure a timestamp exists; if not, inject the current system time
        if (!TIMESTAMP_PATTERN.matcher(processedPayload).find()) {
             processedPayload = processedPayload.replaceFirst("\\{", "{\"timestamp\": " + System.currentTimeMillis() + ", ");
        }

        // Strip out deprecated legacy fields (Mock implementation)
        processedPayload = processedPayload.replace("\"legacy_user_id\": null,", "");
        
        return processedPayload;
    }
```

PRIMARY MATCH: `isValidEvent`

File: `server/sample_repository/frontend/modern_dash/types.ts`

Code:
```
export function isValidEvent(event: TelemetryEvent): boolean {
    return event.sensorReadings.length > 0 && event.timestamp > 0;
}
```

PRIMARY MATCH: `mask_payload`

File: `server/sample_repository/backend/core_engine/crypto_utils.c`

Code:
```
void mask_payload(const char* input, char* output, char key) {
    int i = 0;
    while (input[i] != '\0') {
        output[i] = input[i] ^ key;
        i++;
    }
    output[i] = '\0';
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

PRIMARY MATCH: `StreamProcessor.ingestAndRoute`

File: `server/sample_repository/backend/java_service/src/main/java/com/polydata/StreamProcessor.java`

Code:
```
public boolean ingestAndRoute(String rawData) {
        if (rawData == null || rawData.isEmpty()) {
            LOGGER.warning("Received null or empty raw data packet. Dropping payload.");
            return false;
        }

        try {
            // 1. Basic Structural Validation
            if (!rawData.trim().startsWith("{") || !rawData.trim().endsWith("}")) {
                throw new IllegalArgumentException("Malformed JSON payload: Must be a valid JSON object.");
            }

            // 2. Schema Normalization
            String normalizedData = normalizeSchema(rawData);

            // 3. Routing to Analytics Queue
            return routeToQueue(normalizedData);

        } catch (IllegalArgumentException e) {
            LOGGER.log(Level.WARNING, "Validation error during ingestion: " + e.getMessage());
            return false;
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Unexpected error routing data packet", e);
            return false;
        }
    }
```

PRIMARY MATCH: `StreamProcessor.routeToQueue`

File: `server/sample_repository/backend/java_service/src/main/java/com/polydata/StreamProcessor.java`

Code:
```
private boolean routeToQueue(String normalizedPayload) {
        // Mock queue routing logic
        LOGGER.info(String.format("Successfully routed payload [Hash: %d] to analytics queue.", normalizedPayload.hashCode()));
        return true;
    }
```

PRIMARY MATCH: `fetchLegacyUserData`

File: `server/sample_repository/frontend/legacy_portal/api_client.js`

Code:
```
export async function fetchLegacyUserData(userId) {
    try {
        const response = await fetch(`/api/v1/users/${userId}`);
        return await response.json();
    } catch (error) {
        console.error("Failed to fetch user:", error);
        return null;
    }
}
```

