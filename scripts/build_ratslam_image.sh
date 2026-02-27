#!/usr/bin/env bash
set -euo pipefail

IMG_TAG="ratslam:melodic"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

docker build -f "$ROOT_DIR/docker/Dockerfile.ratslam" -t "$IMG_TAG" "$ROOT_DIR"
echo "\nBuilt image: $IMG_TAG"

