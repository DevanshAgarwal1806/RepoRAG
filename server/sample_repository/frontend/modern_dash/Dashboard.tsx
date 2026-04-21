import React, { useState, useEffect } from 'react';
import { TelemetryEvent } from './types';

interface Props {
    streamEndpoint: string;
}

/**
 * Main React component for visualizing real-time telemetry data.
 * Subscribes to a WebSocket endpoint to render live charts.
 */
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
