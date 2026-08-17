#!/usr/bin/env bash
set -e
# Pulsar2 AX650 compile for FastEnhancer CoreNet.
# Usage: bash compile.sh 16k   or   bash compile.sh 48k
MODEL="${1:-16k}"
CFG="$(dirname "$0")/${MODEL}/pulsar2_config.json"
echo "Compiling FastEnhancer ${MODEL}..."
docker run --rm --network host -v "$(pwd)":/workspace pulsar2:7.0 \
    pulsar2 build --config "/workspace/${CFG#./}"
echo "Done: model.axmodel"
