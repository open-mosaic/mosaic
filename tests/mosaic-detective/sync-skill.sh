#!/bin/bash
# Copy the detective into the Claude Code skills directory.
set -e
DEST=~/.claude/skills/mosaic-detective
mkdir -p "$DEST/scripts" "$DEST/.claude"
cp SKILL.md fault-signatures.md metrics-reference.md mosaic_queries.py "$DEST/"
cp scripts/*.py "$DEST/scripts/"
cp skill-settings.json "$DEST/.claude/settings.local.json"
echo "synced to $DEST"
