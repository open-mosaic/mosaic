---
icon: fontawesome/solid/box-open
title: Fault Detective Deployment
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Fault Detective deployment

Installing, verifying and tuning the Fault Detective. What it is and why it is built this way is
covered in [Fault Detective](mosaic-detective.md).

## Deployment

Six steps and one measured number. Everything after them is optional.

### Before you start

A running profiler stack with Prometheus scraping, and a live workload — with nothing running,
every check reads as a fault. Docker on the head node. A Discord server you can add applications
to. A Claude plan supporting headless use (Pro, Max, Team or Enterprise), and outbound access
from the head node to the Claude API and to Discord.

### 1. Discord credentials

The bot token is how Kowalski listens; the webhook is how it posts.

At [discord.com/developers/applications](https://discord.com/developers/applications), create an
application, then under **Bot** reset and copy the token. That is `DISCORD_BOT_TOKEN`, shown
once. On the same page enable **Message Content Intent** — without it the bot connects, appears
online, never sees a message, and reports no error. Invite it via **OAuth2 → URL Generator**,
scope `bot`, permissions View Channels, Send Messages, Read Message History.

For `DISCORD_WEBHOOK`, in the channel diagnoses should land in: **Edit Channel → Integrations →
Webhooks → New Webhook**. Diagnoses post there regardless of where the trigger phrase was typed.

### 2. Claude token

Interactive login does not work in a container and a Keychain session cannot be copied into
Linux. Run this on a machine with a browser, not over SSH on the head node:

```bash
claude setup-token
```

Same browser flow as `/login`, prints a token valid for one year, saved nowhere. Copy it
straight across and write the expiry somewhere a human will see it — when it lapses the receiver
posts the failure to the channel, which only helps if someone is reading.

### 3. Configure

Everything runs from `tests/mosaic-detective/`:

```bash
cd tests/mosaic-detective
cp .env.local.example .env.local
```

| Variable | What it is |
|---|---|
| `DISCORD_BOT_TOKEN` | Bot token, for the interactive chat commands |
| `DISCORD_WEBHOOK` | Webhook URL, for posting diagnoses |
| `PROMETHEUS_HOST` | For running outside the container. Compose overrides it |
| `PROMETHEUS_PORT` | As above |
| `CLAUDE_CODE_OAUTH_TOKEN` | Headless authentication for Claude Code |
| `TZ` | Timezone for timestamps in reports |

The file is gitignored and holds live credentials. Keep it that way.

### 4. Set one number

`fault-signatures.md` holds a healthy per-rank throughput figure from the reference cluster.
Replace it with yours. With your normal workload running and no faults injected:

```promql
rate(nccl_profiler_collective_bytes_total[5m])
```

One series per rank. Divide by a million for MB/s and take the median. That is the only value
you have to change, and the agent is materially worse without it. See [Concrete numbers beat
abstract procedures](#concrete-numbers-beat-abstract-procedures).

### 5. Build and run

The compose file expects the profiler stack's network to already exist. Check the name with
`docker network ls` before building.

```bash
docker compose up -d --build
docker compose logs -f
```

`restart: unless-stopped` is set, so the system survives a host reboot. You should see a boot
greeting in Discord.

### 6. Alerting

Follow [Grafana Alerting](alerting.md) for the three rules, then create a webhook contact point
pointing at the container by name on port 8500 and hit **Test**.

Between the boot greeting, a status command in chat, and a successful **Test**, you have proved
the container is up, the credentials work, and Grafana can reach it. That is deployed.

## Going further

None of this is needed to run the system. In rough order of what pays back fastest.

### Prove the whole loop with an injected fault

The steps above stop short of the agent actually running. An injected fault is the only thing
that exercises alert evaluation, invocation, authentication and diagnosis together. Killing the
collector is the fastest unambiguous check: inject, leave it alone, then restore. Have a deadman
timer and a tested restore path first.

### Finish the calibration

Beyond the single throughput figure, add healthy GPU clock ranges and a note of any recurring
oscillation in steady state. The reference cluster swings by about a third between two values in
normal operation, a batching artifact of the collective, and the agent reports it as a fault
unless told otherwise. Yours will have its own. See [Write down what normal looks
like](#write-down-what-normal-looks-like).

### Run the skill interactively

For development, or to try the skill without the bot in the way:

```bash
./sync-skill.sh
```

Copies the skill files, `mosaic_queries.py`, the query scripts and the permissions file into
`~/.claude/skills/mosaic-detective/`. Set `PROMETHEUS_HOST` to reach your cluster, then ask for
an analysis in Claude Code.

The container builds its own copy at image build time, so this is for local use only. Either way
the agent reads the installed copy: an edit you have not synced or rebuilt has not taken effect.

### Tune the knowledge base against your own faults

Once the loop runs, accuracy is a knowledge base problem rather than a code one. [Best
practices](#best-practices) is what came out of building it, and [Design the evaluation before
tuning](#design-the-evaluation-before-tuning) is the part to read first.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Posts `Detective failed (exit ...)` | Usually an expired token. Regenerate with `claude setup-token` |
| Posts `Could not run claude` | The `claude` binary is not on `PATH` in the container |
| A second alert during a fault produces nothing | Global three minute cooldown, not per-rule |
| Bot is online but ignores the trigger phrase | Message Content Intent not enabled |
| Agent replies but does not use the skill | Skill not discovered. `SKILL.md` must open with its YAML front matter on line 1 |
| Skill edits have no effect | Not synced, or the image not rebuilt |
| `docker compose up` fails on the network | Profiler network absent or named differently |
| Alerts never arrive | Contact point pointing at the wrong host, or the container not on the profiler network |
| Alerts arrive minutes late | Group wait left at the Grafana default |
| Duplicate diagnoses for one fault | Cooldown shorter than the alert's repeat interval |
| Reports truncated mid sentence | Message chunking not splitting at newlines |
| Discord rejects posts with error 1010 | Missing `User-Agent` header on webhook requests |
| Timestamps wrong | `TZ` not set in `.env.local` |
| Everything reads as an anomaly | Detection threshold overridden, or no workload running |

After every rebuild, check the mode. It lives in memory and resets to `ARMED`, so a container
that restarts while you were deliberately disarmed comes back diagnosing on its own.

## Best practices

### Calibrate before you trust it

The method transfers between clusters. The numbers do not. Every figure in the knowledge base
came from one network bound configuration; the [Limitations](mosaic-detective.md#limitations)
explain what that distorts.

Do not try to predict what your signatures will be. The predicted ones were wrong often enough
that predicting them stopped being worth doing. Inject the fault, watch what moves, write it
down.

### Write down what normal looks like

Healthy clusters are not flat, and the agent needs telling which movement is expected.

The reference cluster swings by about a third between two values in steady state, a batching
artifact of the collective. A short window rate oscillated depending on whether it caught ten or
eleven counter increments, so anything under about ten percent was noise. One component had
genuinely higher natural variance than its three peers, enough that any "one component differs"
rule would have flagged its healthy jitter.

Each needed an explicit entry saying this is normal, with numbers. Find your equivalents during
calibration and write them down before writing a single fault signature. This part of the
knowledge base earns its keep faster than any other.

### Compare two windows, never one

A wide window dilutes a fault that started thirty seconds ago: the average barely moves and the
check reads healthy. A narrow window on its own is too noisy to trust.

Compute both. A one minute rate against a five minute rate is self baselining, detecting change
against the cluster's own recent history rather than a fixed number. If the narrow window sits
meaningfully below the wide one, something is degrading now. Both being equally low is also
degradation, once the fault has run long enough to drag the wide one down — that is where you
need your calibrated healthy figure.

### The reference documents are the program

`SKILL.md` and its reference files are not documentation about the agent's behavior, they are
the behavior. The model is the interpreter and those files are the source.

A factually wrong sentence in a markdown file is a bug, and it does not present as an error. The
skill once diagnosed a killed rank and named the wrong one, on the wrong node, with reasoning
that read perfectly. The cause was one sentence in `fault-signatures.md` claiming that a dying
rank's counter stops ten to twenty seconds before its peers. It does not — the abort propagates
within a single scrape interval. The model reasoned correctly from a false premise it had been
handed.

Debug the knowledge base before you debug the prompt, and confirm the edit landed in the copy
the running system reads.

### It never tells you an instruction is impossible

Not once, across the whole build, did the agent say it had been asked to find something that was
not in the data. It improvised something plausible instead, every time.

So the failure mode is not a crash. It is a confident wrong answer that looks exactly like a
right one, and the only way to find those is to check output against ground truth you injected
yourself. That is the real argument for the fault injection harness: it is not scaffolding for
the AI part, it is the only reason you can tell whether the AI part works.

### Concrete numbers beat abstract procedures

The skill missed injected packet loss three times running. The cause was a prose tidy up that
removed the one concrete number in the reference document, the healthy per-rank baseline. With
no number to compare against, there was nothing to compare against.

"Check whether throughput is degraded" performs noticeably worse than a worked example showing
what healthy looks like and what the fault did to it. The trade is real — those numbers are what
make the agent work and what make it site specific. Keep them, labelled as observations from one
cluster rather than thresholds.

### Teach it to abstain, and score abstention as correct

Some questions cannot be answered from the available data. Which rank died is one at typical
scrape resolution; where a network fault sits is another, on a bulk synchronous ring.

An agent will answer anyway unless told not to, and vague hedging does not work. Name the
specific inference it must not make, in the specific circumstance, and state that "unresolved at
this resolution" is a correct verdict. Scope it narrowly: a broader prohibition against naming
nodes caused a suspected false negative, because removing that signal also removed the evidence
that degradation existed at all. Prohibit the conclusion, not the observation.

Make sure the evaluation accepts abstention as correct, or you will tune the agent straight back
out of the behavior you just taught it.

### Design the evaluation before tuning

The obvious eval design leaks. You tune `SKILL.md` and the reference files until the agent
recognizes the faults you documented into it, which is close to tautological.

What helps:

- Hold out variants. Tune on a fault at one component, evaluate at another. Better still, hold
  back an entire fault type entirely, which tells you what the agent does with something
  unfamiliar.
- Decide what correct means up front. Fault type only, or type plus location? The second is much
  harder and much more meaningful.
- Small n lies. Four out of five against three out of five swings on one draw. A harness that
  injects a random fault, waits, invokes the agent, scores it and restores makes twenty trials
  cheap.
- Export raw metric windows per trial, not screenshots. A playbook wants screenshots, an eval
  wants labeled data.
- The grader prompt is load bearing. Scoring with a second headless call works, but that prompt
  needs as much care as the skill.

With a non deterministic component in the loop, revert to the last state that scored well rather
than iterating forward from a broken one.
