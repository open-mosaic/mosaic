#!/bin/bash
# Copy the detective into the Claude Code skills directory.
set -e
DEST=~/.claude/skills/mosaic-detective
mkdir -p "$DEST/scripts"
cp SKILL.md fault-signatures.md metrics-reference.md mosaic_queries.py "$DEST/"
cp scripts/*.py "$DEST/scripts/"
echo "synced to $DEST"
