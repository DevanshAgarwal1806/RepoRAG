#!/bin/bash

queries=(
    "Where is the logic for finding statistical deviations in sensor data?"
    "Find the code that encrypts or obfuscates payloads before they are sent."
    "How do we generate 768-dimensional vectors from text?"
    "Find the TypeScript interface definition for a telemetry event."
    "Show me the enterprise Java class responsible for ingesting streams."
    "Locate the React component that renders the live analytics charts."
    "Where is the entry point for getting user profiles from the v1 API?"
    "Find the C++ header file for fast math operations."
    "Show me the JSX file used for the legacy user avatar card."
    "Search for ingestAndRoute."
    "Find references to z-score."
)

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_repository>"
    exit 1
fi

repo_path="$1"
output_dir="${repo_path}_output"

mkdir -p "$output_dir"

i=1

for query in "${queries[@]}"; do
    echo "Processing query: $query"
    filename="${output_dir}/output${i}.txt"
    python server/run_pipeline.py --repo "$repo_path" --query "$query" > "$filename"
    echo "Finished processing query $i"
    ((i++))
done