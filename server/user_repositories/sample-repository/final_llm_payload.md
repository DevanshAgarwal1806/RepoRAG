### USER QUERY: odhvlfb

### CODEBASE CONTEXT

Function: `Dashboard`
File: `server/sample_repository/frontend/modern_dash/Dashboard.tsx`
Lines: 12-68

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

Function: `isValidEvent`
File: `server/sample_repository/frontend/modern_dash/types.ts`
Lines: 11-13

Code:
```
export function isValidEvent(event: TelemetryEvent): boolean {
    return event.sensorReadings.length > 0 && event.timestamp > 0;
}
```

Function: `UserWidget`
File: `server/sample_repository/frontend/legacy_portal/UserWidget.jsx`
Lines: 6-13

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

Function: `fetchLegacyUserData`
File: `server/sample_repository/frontend/legacy_portal/api_client.js`
Lines: 5-13

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
