# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Kowalski: Discord control plus autonomous Grafana alert handling, one process.

The alert receiver runs in a daemon thread alongside the Discord client, so the
two share a plain module-level mode flag -- no state file, no control port.

Modes (what happens when a Grafana alert arrives):
    armed     announce the alert, then diagnose it and post the report
    notify    announce the alert; wait for a human to ask for a diagnosis
    disarmed  log it in the terminal only, post nothing

Commands (all prefixed "Kowalski,"):
    analysis          run a diagnosis now -- works in every mode
    arm / on          switch to armed
    notify / standby  switch to notify
    disarm / off      switch to disarmed
    status            show the current mode
    help              list commands
"""

import asyncio
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import discord

# --- config --------------------------------------------------------------

SKILL_DIR = os.path.expanduser("~/.claude/skills/mosaic-detective")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
PORT = 8500
COOLDOWN_SECONDS = 180
TIMEOUT = 900
PREFIX = "kowalski"

ARMED, NOTIFY, DISARMED = "armed", "notify", "disarmed"

# Starting mode. Set to NOTIFY or DISARMED if you would rather not diagnose
# automatically from boot -- but note an unattended restart would then leave
# autonomous diagnosis silently off.
MODE = ARMED

MODE_BLURB = {
    ARMED: "**ARMED** — alerts are diagnosed automatically",
    NOTIFY: "**NOTIFY** — alerts are announced here; ask for a diagnosis when you want one",
    DISARMED: "**DISARMED** — alerts are logged locally only",
}

_last_fired = 0.0
_greeted = False

# Guards against two diagnoses overlapping -- a manual summon during an alert
# run, or two people summoning at once. Both paths are in this process, so one
# lock covers everything.
_run_lock = threading.Lock()


# --- detective -----------------------------------------------------------

def run_detective():
    """Invoke Claude headlessly. Returns the report, an error string, or None
    if a run is already in progress."""
    if not _run_lock.acquire(blocking=False):
        return None
    try:
        env = dict(os.environ)
        env.setdefault("PROMETHEUS_HOST", "localhost")
        env.setdefault("PROMETHEUS_PORT", "9090")
        try:
            result = subprocess.run(
                ["claude", "-p", "Kowalski, analysis"],
                capture_output=True, text=True, timeout=TIMEOUT,
                cwd=SKILL_DIR, env=env)
        except subprocess.TimeoutExpired:
            return f"Detective timed out after {TIMEOUT}s."
        except FileNotFoundError:
            return "Could not run `claude` -- not on PATH for this process."

        # A non-zero exit is how an expired session surfaces; without this it
        # returns empty stdout and looks like a silent success.
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[:600]
            return f"Detective failed (exit {result.returncode}):\n{detail}"
        return result.stdout.strip() or "Detective produced no output."
    finally:
        _run_lock.release()


def chunk(text, size=1900):
    """Split for Discord's 2000-char cap, breaking on newlines where possible."""
    out = []
    while text:
        if len(text) <= size:
            out.append(text)
            break
        cut = text.rfind("\n", 0, size)
        if cut < size // 2:
            cut = size
        out.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return out


# --- alert receiver (daemon thread) --------------------------------------

def post_to_discord(text):
    if not DISCORD_WEBHOOK:
        print("  no DISCORD_WEBHOOK set, skipping post")
        return
    parts = chunk(text)
    for i, part in enumerate(parts):
        req = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=json.dumps({"content": part}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "Kowalski/1.0"})
        try:
            urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as e:
            print(f"  Discord rejected the post: {e.code} — {e.read().decode()[:200]}")
            return
        except Exception as e:
            print(f"  Discord post failed: {e}")
            return
        if i < len(parts) - 1:
            time.sleep(0.5)      # stay under the webhook rate limit
    print(f"  posted to Discord ({len(parts)} message(s))")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global _last_fired

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else ""
        print(f"\n[{datetime.now():%H:%M:%S}] alert received")

        title, status = "unknown", ""
        try:
            payload = json.loads(body)
            status = payload.get("status", "")
            title = payload.get("title", status or "unknown")
            print(f"  alert: {title}")
        except json.JSONDecodeError:
            print("  (body was not JSON)")

        # Acknowledge immediately. The diagnosis runs on its own thread below,
        # because holding this connection open for the ~90s a run takes makes
        # Grafana time out and retry, producing bursts of duplicate alerts.
        self.send_response(200)
        self.end_headers()

        # A resolved notification means the cluster recovered. Diagnosing that
        # spends a full model invocation to report that nothing is wrong.
        if status == "resolved":
            print("  resolved notification — ignoring")
            self.wfile.write(b"resolved\n")
            return

        if MODE == DISARMED:
            print("  disarmed — logged locally, nothing posted")
            self.wfile.write(b"disarmed\n")
            return

        # Cooldown applies to notify as well, so a flapping alert does not
        # spam the channel with announcements.
        now = time.time()
        if now - _last_fired < COOLDOWN_SECONDS:
            print(f"  within cooldown ({int(COOLDOWN_SECONDS - (now - _last_fired))}s left)")
            self.wfile.write(b"cooldown\n")
            return
        _last_fired = now

        banner = f"⚠️ **Alert fired** — {title}"

        if MODE == NOTIFY:
            self.wfile.write(b"notified\n")
            print("  notify mode — announced, not diagnosing")
            threading.Thread(
                target=post_to_discord, daemon=True,
                args=(f"{banner}\nOn standby. Say `Kowalski, analysis` "
                      f"if you want this diagnosed.",)).start()
            return

        self.wfile.write(b"received\n")
        threading.Thread(target=self._diagnose, args=(banner,), daemon=True).start()

    def _diagnose(self, banner):
        """Announce, run the detective, post the report. Runs off the HTTP
        handler thread so the webhook response is not held open."""
        # Announce first: a diagnosis takes about a minute, and silence for
        # that long looks like nothing happened.
        post_to_discord(f"{banner}\nAye aye, Skipper. Diagnosing now — report to follow.")

        print("  running detective...")
        report = run_detective()
        if report is None:
            print("  a diagnosis is already running, skipping")
            post_to_discord("A diagnosis is already in progress — this alert is covered by it.")
            return

        post_to_discord(report)
        print("=" * 60)
        print(report)
        print("=" * 60)

    def log_message(self, *args):
        pass


def start_receiver():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Alert receiver listening on port {PORT}")


# --- discord bot ---------------------------------------------------------

HELP = (
    "**Kowalski commands**\n"
    "`Kowalski, analysis` — diagnose the cluster now (works in any mode)\n"
    "`Kowalski, arm` — announce alerts and diagnose them automatically\n"
    "`Kowalski, notify` — announce alerts, diagnose only on request\n"
    "`Kowalski, disarm` — log alerts locally, post nothing\n"
    "`Kowalski, status` — show the current mode\n"
    "`Kowalski, help` — this message"
)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def parse_command(content):
    text = content.strip().lower()
    if not text.startswith(PREFIX):
        return None
    rest = text[len(PREFIX):].lstrip(" ,:-")
    return rest.split()[0].strip("!?.,") if rest else None


def ack(text):
    """Prefix every acknowledged order, because of course."""
    return f"Aye aye, Skipper.\n{text}"


@client.event
async def on_ready():
    global _greeted
    print(f"Kowalski online as {client.user} ({MODE})")

    # on_ready fires again on every gateway reconnect -- greet only once, or a
    # flaky connection turns into channel spam.
    if _greeted:
        return
    _greeted = True

    await asyncio.to_thread(
        post_to_discord,
        f"🐧 **Kowalski reporting for duty.**\n"
        f"Status: {MODE_BLURB[MODE]}\n"
        f"Awaiting orders — `Kowalski, help` for the full list.")


@client.event
async def on_message(message):
    global MODE

    if message.author == client.user:
        return
    # Ignore our own webhook posts, so an announcement can never trigger a run.
    if message.webhook_id:
        return

    command = parse_command(message.content)
    if command is None:
        return

    if command in ("arm", "on"):
        MODE = ARMED
        await message.channel.send(ack(MODE_BLURB[ARMED]))

    elif command in ("notify", "standby"):
        MODE = NOTIFY
        await message.channel.send(ack(MODE_BLURB[NOTIFY]))

    elif command in ("disarm", "off"):
        MODE = DISARMED
        await message.channel.send(ack(MODE_BLURB[DISARMED]))

    elif command == "status":
        busy = " — diagnosis in progress" if _run_lock.locked() else ""
        await message.channel.send(ack(MODE_BLURB[MODE] + busy))

    elif command == "help":
        await message.channel.send(ack(HELP))

    elif command == "analysis":
        await message.channel.send(ack("On it — analysing the cluster..."))
        # subprocess.run blocks for ~a minute; running it on the event loop
        # would stall gateway heartbeats and can disconnect the bot.
        report = await asyncio.to_thread(run_detective)
        if report is None:
            await message.channel.send("A diagnosis is already running — hold on.")
            return
        for part in chunk(report):
            await message.channel.send(part)

    else:
        # Deliberately not acknowledged -- a failure should not look like an
        # accepted order.
        await message.channel.send(f"Unknown command `{command}`. Try `Kowalski, help`.")


if __name__ == "__main__":
    token = os.environ["DISCORD_BOT_TOKEN"]
    start_receiver()
    client.run(token)