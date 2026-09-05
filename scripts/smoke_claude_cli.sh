#!/bin/bash
# Smoke test: can the Claude Code binary run headless from a compute node using the existing login?
CLAUDE_BIN=/mmfs1/data/home/yding/.vscode-server/extensions/anthropic.claude-code-2.1.233-linux-x64/resources/native-binary/claude
cd /mmfs1/data/group/pmc050/yding/gad_reasoning
echo "host=$(hostname) whoami=$(whoami) HOME=$HOME"
timeout 180 "$CLAUDE_BIN" -p 'Reply with exactly this JSON and nothing else: {"ok": true, "sum": 2+3 evaluated as an integer}' \
  --output-format json --model claude-sonnet-4-6 --max-turns 1 2>&1 | head -c 2000
echo; echo "exit=$?"
