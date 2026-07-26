# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Receive a Grafana alert webhook, run the Mosaic Detective, record the report."""

import json
import os
import subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.error
import urllib.request

SKILL_DIR = os.path.expanduser("~/.claude/skills/mosaic-detective")
REPORT_DIR = os.path.expanduser("~/mosaic/tests/mosaic-detective/reports")



DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")

import time

_last_fired = 0.0
COOLDOWN_SECONDS = 180

def post_to_discord(text):
    print(f"  webhook set: {bool(DISCORD_WEBHOOK)}, len={len(DISCORD_WEBHOOK)}")
    if not DISCORD_WEBHOOK:
        return
    # Discord caps messages at 2000 chars.
    body = json.dumps({"content": text[:1990]}).encode()
    req = urllib.request.Request(
        DISCORD_WEBHOOK, data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "Kowalski/1.0"})
    try:
        urllib.request.urlopen(req, timeout=10)
        print("  posted to Discord")
    except urllib.error.HTTPError as e:
        print(f"  Discord failed: {e.code} — {e.read().decode()}")
    except Exception as e:
        print(f"  Discord post failed: {e}")




def run_detective():
    result = subprocess.run(
        ["claude", "-p", "Kowalski, analysis"],
        capture_output=True, text=True, timeout=900,
        cwd=SKILL_DIR,
        env={**os.environ, "PROMETHEUS_HOST": "bravo", "PROMETHEUS_PORT": "9090"})
    return result.stdout


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else ""
        print(f"\n[{datetime.now():%H:%M:%S}] alert received")
        global _last_fired
        now = time.time()
        if now - _last_fired < COOLDOWN_SECONDS:
            remaining = int(COOLDOWN_SECONDS - (now - _last_fired))
            print(f"  within cooldown ({remaining}s left), ignoring")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"cooldown\n")
            return
        _last_fired = now
        try:
            payload = json.loads(body)
            print(f"  alert: {payload.get('title', payload.get('status', 'unknown'))}")
        except json.JSONDecodeError:
            print("  (body was not JSON)")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"received\n")

        print("  running detective...")
        report = run_detective()

        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, f"report-{datetime.now():%Y%m%d-%H%M%S}.md")
        with open(path, "w") as f:
            f.write(report)
        print(f"  report written to {path}")
        post_to_discord(f"**Kowalski, analysis!**\n{report}")
        print("=" * 60)
        print(report)
        print("=" * 60)

    def log_message(self, *args):
        pass



if __name__ == "__main__":
    port = 8500
    print(f"Kowalski alert receiver listening on port {port}")
    print("Waiting for webhooks (Ctrl+C to stop)...")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
