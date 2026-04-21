### USER QUERY: Where is the logic for finding statistical deviations in sensor data?

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

PRIMARY MATCH: `isValidEvent`

File: `server/sample_repository/frontend/modern_dash/types.ts`

Code:
```
export function isValidEvent(event: TelemetryEvent): boolean {
    return event.sensorReadings.length > 0 && event.timestamp > 0;
}
```

PRIMARY MATCH: `StreamProcessor.ingestAndRoute`

File: `server/sample_repository/backend/java_service/src/main/java/com/polydata/StreamProcessor.java`

Code:
```
public boolean ingestAndRoute(String rawData) {
        if (rawData == null || rawData.isEmpty()) {
            return false;
        }
        // Normalization logic would go here
        return true;
    }
```

