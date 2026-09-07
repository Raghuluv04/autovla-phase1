#!/usr/bin/env bash
# BDD-X explanation annotations (Kim et al., ECCV 2018). Annotations only —
# a few MB of CSV, no video — which is all that is needed to build and
# evaluate the natural-language justification data in Milestone 4.
set -euo pipefail
DEST="${1:-data/bddx}"
mkdir -p "$DEST"
URL="https://raw.githubusercontent.com/JinkyuKimUCB/BDD-X-dataset/master/BDD-X-Annotations_v1.csv"
echo "fetching BDD-X annotations ..."
curl -fL "$URL" -o "$DEST/BDD-X-Annotations_v1.csv"
wc -l "$DEST/BDD-X-Annotations_v1.csv"
echo "ok: $DEST"
