### USER QUERY: Find references to z-score.

### CODEBASE CONTEXT

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

PRIMARY MATCH: `Dashboard`

File: `server/sample_repository/frontend/modern_dash/Dashboard.tsx`

Code:
```
export const Dashboard: React.FC<Props> = ({ streamEndpoint }) => {
    const [events, setEvents] = useState<TelemetryEvent[]>([]);

    useEffect(() => {
        // Mock WebSocket connection
        console.log(`Subscribing to ${streamEndpoint}`);
    }, [streamEndpoint]);

    return (
        <div className="dashboard-container">
            <h1>Live Telemetry Analytics</h1>
            <p>Active Events: {events.length}</p>
        </div>
    );
};
```

