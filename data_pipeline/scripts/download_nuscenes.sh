#!/usr/bin/env bash
# nuScenes v1.0-mini (10 scenes, ~4 GB) — enough to run the full pipeline on
# real data and to render camera-projected trajectories.
#
# nuScenes requires accepting its terms and signing in, so the archive cannot
# be fetched non-interactively. Steps:
#   1. Create a free account at https://www.nuscenes.org/sign-up
#   2. Go to nuScenes -> Downloads -> "Mini" (v1.0-mini.tgz)
#   3. Put the downloaded file next to this repo and run:
#        bash scripts/download_nuscenes.sh /path/to/v1.0-mini.tgz
set -euo pipefail
ARCHIVE="${1:-}"
DEST="${2:-data/nuscenes}"

if [[ -z "$ARCHIVE" ]]; then
  echo "usage: bash scripts/download_nuscenes.sh /path/to/v1.0-mini.tgz [dest]"
  echo "download it from https://www.nuscenes.org/nuscenes#download (Mini split)"
  exit 1
fi

mkdir -p "$DEST"
echo "extracting $ARCHIVE -> $DEST (this takes a few minutes) ..."
tar -xzf "$ARCHIVE" -C "$DEST"

if [[ -d "$DEST/v1.0-mini" ]]; then
  echo "ok: $DEST/v1.0-mini"
  echo "now run: python scripts/run_pipeline.py --source nuscenes"
else
  echo "warning: expected $DEST/v1.0-mini — check the archive layout"
fi
