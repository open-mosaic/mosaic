# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Discord bot: say 'Kowalski, analysis' in a channel, get a diagnosis back.

The chat platform is an adapter around run_detective(); a Slack version would
reuse that function and swap only this file's event handling.
"""

import os
import subprocess

import discord

SKILL_DIR = os.path.expanduser("~/.claude/skills/mosaic-detective")
TRIGGER = "kowalski, analysis"


def run_detective():
    """Invoke Claude headlessly and return its diagnosis (shared with the receiver)."""
    result = subprocess.run(
        ["claude", "-p", "Kowalski, analysis"],
        capture_output=True, text=True, timeout=900,
        cwd=SKILL_DIR,
        env={**os.environ, "PROMETHEUS_HOST": "bravo", "PROMETHEUS_PORT": "9090"})
    return result.stdout


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Kowalski online as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if TRIGGER in message.content.lower():
        await message.channel.send("On it — analysing the cluster...")
        report = run_detective()
        # Discord caps messages at 2000 chars; chunk if needed.
        for i in range(0, len(report), 1900):
            await message.channel.send(report[i:i + 1900])


if __name__ == "__main__":
    token = os.environ["DISCORD_BOT_TOKEN"]
    client.run(token)
