/**
 * Core data structure for telemetry events processed by the modern dashboard.
 */
export interface TelemetryEvent {
    eventId: string;
    timestamp: number;
    sensorReadings: number[];
    isAnomalous: boolean;
}

export function isValidEvent(event: TelemetryEvent): boolean {
    return event.sensorReadings.length > 0 && event.timestamp > 0;
}
