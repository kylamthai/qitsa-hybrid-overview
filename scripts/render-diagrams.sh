#!/usr/bin/env bash
# Render QITSA pipeline Mermaid diagrams locally.
set -euo pipefail

if ! command -v mmdc >/dev/null 2>&1; then
  echo "mermaid-cli (mmdc) not found. Install with: npm install -g @mermaid-js/mermaid-cli"
  exit 1
fi

INPUT="docs/pipeline-overview.mmd"
OUT1="docs/pipeline-overview.png"
OUT2="docs/pipeline-overview@2x.png"

mmdc -i "$INPUT" -o "$OUT1" --scale 1
mmdc -i "$INPUT" -o "$OUT2" --scale 2

echo "Generated:"
ls -la "$OUT1" "$OUT2"
