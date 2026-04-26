package com.polydata;

import java.util.regex.Pattern;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * Enterprise service class responsible for ingesting high-throughput data streams.
 */
public class StreamProcessor {

    private static final Logger LOGGER = Logger.getLogger(StreamProcessor.class.getName());
    
    // Pattern to quickly check for the existence of a timestamp field in the raw JSON
    private static final Pattern TIMESTAMP_PATTERN = Pattern.compile("\"timestamp\"\\s*:\\s*\\d+");

    /**
     * Ingests a raw data packet, normalizes the schema, and routes it to the analytics queue.
     * @param rawData The incoming JSON payload as a string.
     * @return boolean indicating successful ingestion.
     */
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

    /**
     * Normalizes the incoming JSON payload to match the internal enterprise telemetry schema.
     * In a production environment, this would likely utilize Jackson or Gson.
     */
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

    /**
     * Simulates publishing the normalized payload to an event bus (e.g., Kafka or RabbitMQ).
     */
    private boolean routeToQueue(String normalizedPayload) {
        // Mock queue routing logic
        LOGGER.info(String.format("Successfully routed payload [Hash: %d] to analytics queue.", normalizedPayload.hashCode()));
        return true;
    }
}