FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Claude Code reads the skill from the installed location, not the repo, and
# runs with that directory as its working directory. The build does what
# sync-skill.sh does on a workstation.
ENV SKILL=/root/.claude/skills/mosaic-detective
RUN mkdir -p $SKILL/scripts $SKILL/.claude
COPY SKILL.md fault-signatures.md metrics-reference.md mosaic_queries.py $SKILL/
COPY scripts/ $SKILL/scripts/
COPY skill-settings.json $SKILL/.claude/settings.local.json

COPY kowalski.py mosaic_queries.py ./
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1
EXPOSE 8500
CMD ["python3", "kowalski.py"]
