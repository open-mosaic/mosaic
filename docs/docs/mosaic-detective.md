---
icon: fontawesome/solid/magnifying-glass
title: Fault Detective
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Mosaic Fault Detective

An agent skill that diagnoses cluster faults from profiler metrics. Kowalski[^1] is the bot
that runs it.

A Grafana alert fires, Kowalski invokes Claude Code headlessly against the skill, the agent
investigates through a fixed set of read-only scripts, and the diagnosis lands in chat.
Nobody is in the loop.

[^1]: Named after the analyst penguin from the *Madagascar* films, whose catchphrase is
    "Kowalski, analysis!" That is also the phrase that triggers the skill.

Three layers, deliberately kept apart. Detection is deterministic: Grafana rules decide
something is wrong. Diagnosis is agentic: Claude decides what to look at and what it means.
Perception is fixed: the agent sees the cluster only through scripts you wrote and tested.

![Kowalski architecture](images/kowalski-architecture.png)

## How it works

### The loop

| Stage | Component |
|---|---|
| Alert evaluates and fires | Grafana alert rule |
| Alert delivered as a webhook | Grafana contact point |
| Mode and cooldown checked | `kowalski.py` HTTP receiver |
| Agent invoked headlessly | `claude -p` subprocess |
| Skill loaded, investigation planned | Claude Code with the `mosaic-detective` skill |
| Metrics queried | Scripts in `scripts/`, via `mosaic_queries.py` |
| Diagnosis posted | `kowalski.py` chat client |

Everything from the receiver down runs in one container on the head node. An injected fault
reaches a posted diagnosis in about two minutes, most of it the agent's investigation.

### Grafana alert rules

+Alert rules in Grafana trigger a diagnosis, these rules are designed to trigger when a failure occurs, not to diagnose.
No information about the trigger is passed to Kowalski for the diagnosis

[Grafana Alerting](alerting.md) has the expressions and how to calibrate them.


### `kowalski.py`

One process running the chat bot and the HTTP receiver, sharing an in-memory mode flag:
`ARMED` diagnoses and posts, `NOTIFY` announces the alert and waits to be asked, `DISARMED`
logs locally and posts nothing. It starts `ARMED` and returns there on every restart.

`Kowalski, analysis` runs a diagnosis in any mode, `DISARMED` included. Disarming stops
autonomous diagnosis, not the bot.

The rest is defensive:

- A three minute cooldown after any alert. It is global rather than per-rule, so a second,
  different alert arriving inside the window is dropped.
- A non-blocking lock around the invocation. A manual summon during an alert run is told it is
  already covered rather than starting a second agent.
- Diagnosis runs on a daemon thread and the handler returns immediately, otherwise Grafana
  times out and retries. A run takes around ninety seconds; the subprocess is capped at fifteen
  minutes.
- Resolved alerts are dropped. Diagnosing one spends a full invocation to report that nothing
  is wrong.
- Posts from the bot's own webhook are ignored, so an announcement cannot trigger a run.
- Reports split at newlines rather than at the character limit, which silently truncated every
  autonomous report until it was fixed.
- A non-zero exit from `claude -p` is caught and posted. Without that an expired token returns
  empty `stdout` and reads as a successful run with nothing to say.

### The skill

A Claude Code Agent Skill loaded as a unit: `SKILL.md` with the trigger and the diagnostic
procedure, alongside `fault-signatures.md` and `metrics-reference.md`. Those files are not
documentation about what the agent does. They are what it does. See [The reference documents
are the program](#the-reference-documents-are-the-program).

The skill runs from its installed location, not the repository. `sync-skill.sh` copies it
into place, and an unsynced edit has not taken effect.

### Wrapper scripts

The agent does not write queries. It picks among four read-only scripts in `scripts/`:

| Script | Question it answers |
|---|---|
| `check_collector_health.py` | Is the observability pipeline itself up? |
| `list_metrics.py` | What metrics exist, and which are trusted? |
| `metric_timeline.py` | How has one metric moved over a window? |
| `compare_ranks.py` | Do the ranks or links differ from each other? |

`metric_timeline.py` scans for rate-of-change transitions, catching both a frozen counter and
a gauge stepping down. `compare_ranks.py` groups by rank or link, comparing means for gauges
and deltas for counters.

### `mosaic_queries.py`

Connections, query construction, typed results, and the metric names the wrappers use held as
constants. The Metric Reference in [Profiler Design](design.md) covers what each metric
measures. No diagnostic logic lives here: plumbing is verified by unit tests and judgment
against injected faults, and mixing them makes it hard to tell whether a bad diagnosis came
from a bad query or a bad inference.

## Design decisions

### Guided, not caged

The agent picks the investigative move. It does not pick the metric, which is fixed inside
each wrapper in tested code. A doctor ordering a blood panel does not get to redefine what the
blood panel measures.

The boundary is deliberately left open. On one kill-rank run the agent went outside the
wrappers to query process counts, checking whether the processes had actually exited. It was
right, and the diagnosis was better for it. Wrappers-only is blind to anything you did not
anticipate; open query access is more capable and can fabricate. The current position takes
the safe side.

### Detection deterministic, diagnosis agentic

Alert rules are cheap, fast and predictable, and run continuously. Agent invocations are none
of those. The deterministic gate means the expensive part only runs once something simple has
established there is a reason to look.

It also keeps the [Fault Playbook](fault-playbook.md) useful on its own. A human with the
playbook and no agent reaches the same answers.

### Fault injection stays outside

The injection harness is not packaged with the detective, and the detective has no write path
to the cluster. Every script it can run is a query.

### Services are reached by name

The container joins the profiler stack's Docker network, so Grafana reaches it by container
name and it reaches Prometheus by container name. No addresses in runtime configuration.

## Limitations

- Discord is the only chat platform currently supported. Slack is planned; see [Extending to
  another chat platform](#extending-to-another-chat-platform).
- Some faults cannot be localized, and the skill says so rather than guessing. A network fault
  cannot be pinned to a node on a bulk-synchronous ring, and a killed rank cannot be identified
  at typical scrape resolution because the abort propagates within one scrape interval. The
  [Fault Playbook](fault-playbook.md) covers why.
- Combination faults are untested. With the collector down it should report that it cannot see
  the cluster and stop, but that has not been exercised.
- Signatures are calibrated to one cluster: two nodes, four GPUs, 1 GbE, network bound. Several
  findings follow from that bottleneck rather than from the profiler. A GPU clock clamp is the
  clearest case: the clocks move and the alert rule sees it, but throughput does not, because
  the network is the constraint. On a compute bound cluster that reverses. The method
  transfers; the numbers do not.

## Deployment

Six steps and one measured number. Everything after them is optional.

### Before you start

A running profiler stack with Prometheus scraping, and a live workload. With nothing running,
every check reads as a fault. Docker on the head node. A Discord server you can add
applications to. A Claude plan supporting headless use (Pro, Max, Team or Enterprise), and
outbound access from the head node to the Claude API and to Discord.

### 1. Discord credentials

Two credentials from two places. The bot token is how Kowalski listens; the webhook is how it
posts.

At [discord.com/developers/applications](https://discord.com/developers/applications), create
an application, then under **Bot** reset and copy the token. This is `DISCORD_BOT_TOKEN`, and
it is shown once. On the same page, enable **Message Content Intent** under Privileged Gateway
Intents. Without it the bot connects, appears online, and never sees a message; the trigger
phrase never fires and nothing reports an error. Invite it via **OAuth2 → URL Generator**,
scope `bot`, permissions View Channels, Send Messages, Read Message History.

For `DISCORD_WEBHOOK`, in the channel diagnoses should land in: **Edit Channel → Integrations →
Webhooks → New Webhook**. Diagnoses post to that channel regardless of where the trigger phrase
was typed.

### 2. Claude token

Interactive login does not work in a container and a Keychain session cannot be copied into
Linux. Run this on a machine with a browser, not over SSH on the head node:

```bash
claude setup-token
```

Same browser flow as `/login`, prints a token valid for one year. It is not saved anywhere, so
copy it straight across.

Write the expiry somewhere a human will see it. When it lapses the receiver catches the
non-zero exit and posts it, so the failure is visible in the channel rather than silent, but
only to whoever is reading the channel.

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

The compose file expects the profiler stack's network to already exist. Check the name matches
with `docker network ls` before building.

```bash
docker compose up -d --build
docker compose logs -f
```

`restart: unless-stopped` is set, so the system survives a host reboot. You should see a boot
greeting in Discord.

### 6. Alerting

Follow [Grafana Alerting](alerting.md) for the three rules, then create a webhook contact point
pointing at the container by name on port 8500. Hit **Test**, which confirms Grafana can reach
the receiver without waiting for a real fault.

Between the boot greeting, a status command in chat, and a successful **Test**, you have proved
the container is up, the credentials work, and Grafana can reach it. That is deployed.

## Going further

None of this is needed to run the system. In rough order of what pays back fastest.

### Prove the whole loop with an injected fault

The steps above stop short of the agent actually running. An injected fault is the only thing
that exercises alert evaluation, invocation, authentication and diagnosis together.

Have a deadman timer and a tested restore path first. The failure you have not planned for is
the one where the injection outlives the terminal you launched it from. Killing the collector is
the fastest unambiguous check: inject, leave it alone, then restore.

### Finish the calibration

Beyond the single throughput figure, the useful additions are healthy GPU clock ranges and a note
of any recurring oscillation in steady state. The reference cluster swings by about a third
between two values in normal operation, a batching artifact of the collective, and the agent
needs telling that this is expected or it reports it. Yours will have its own. See [Write down
what normal looks like](#write-down-what-normal-looks-like).

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
practices](#best-practices) is what came out of building it, and [Design the evaluation
before tuning](#design-the-evaluation-before-tuning) is the part to read before changing
anything.

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
came from one network bound configuration; the [Limitations](#limitations) explain what that
distorts.

Do not try to predict what your signatures will be. The predicted ones were wrong often
enough that predicting them stopped being worth doing. Inject the fault, watch what moves, and
write that down.

### Write down what normal looks like

Healthy clusters are not flat, and the agent needs to be told which movement is expected.

The reference cluster swings by about a third between two values in steady state, a batching
artifact of the collective. A short window rate oscillated depending on whether it caught ten
or eleven counter increments, so anything under about ten percent was noise. One component had
genuinely higher natural variance than its three peers, enough that any "one component differs"
rule would have flagged its healthy jitter.

Each needed an explicit entry saying this is normal, with numbers. Find your cluster's
equivalents during calibration and write them down before writing a single fault signature.
This part of the knowledge base earns its keep faster than any other.

### Compare two windows, never one

A wide window dilutes a fault that started thirty seconds ago: the average barely moves and the
check reads healthy. A narrow window on its own is too noisy to trust.

Compute both and compare. A one minute rate against a five minute rate is self baselining, so
it detects change relative to the cluster's own recent history rather than against a fixed
number. If the narrow window sits meaningfully below the wide one, something is degrading right
now.

Both windows being equally low is also degradation, once the fault has run long enough to drag
the wide one down. That is where you need your calibrated healthy figure.

### The reference documents are the program

`SKILL.md` and its reference files are not documentation about the agent's behavior, they are
the behavior. The model is the interpreter and those files are the source.

A factually wrong sentence in a markdown file is a bug, and it does not present as an error. It
presents as a confident, well reasoned, wrong answer.

The skill once diagnosed a killed rank and named the wrong one, on the wrong node, with
reasoning that read perfectly. The cause was one sentence in `fault-signatures.md` claiming
that a dying rank's counter stops ten to twenty seconds before its peers. It does not. The abort
propagates within a single scrape interval. The model reasoned correctly from a false premise it
had been handed.

Debug the knowledge base before you debug the prompt. And confirm with a diff or a grep that the
edit landed in the copy the running system reads, not the one in the repository.

### It never tells you an instruction is impossible

Not once, across the whole build, did the agent say it had been asked to find something that was
not in the data. It improvised something plausible instead, every time.

So the failure mode is not a crash. It is a confident wrong answer that looks exactly like a
right one, and the only way to find those is to check output against ground truth you injected
yourself.

Which is the real argument for the fault injection harness. The injection tooling is not
scaffolding for the AI part. It is the only reason you can tell whether the AI part works.

### Concrete numbers beat abstract procedures

The skill missed injected packet loss three times running. The cause was a prose tidy up of the
reference document that removed the one concrete number in it, the healthy per-rank baseline.
With no number to compare against, there was nothing to compare against.

"Check whether throughput is degraded" performs noticeably worse than a worked example showing
what healthy looks like and what the fault did to it.

The trade is real: the concrete numbers are what make the agent work, and they are what make it
site specific. Keep them, and label them as observations from one cluster rather than
thresholds.

### Teach it to abstain, and score abstention as correct

Some questions cannot be answered from the available data. Which rank died is one at typical
scrape resolution. Where a network fault sits is another, on a bulk synchronous ring.

An agent will answer anyway unless told not to, and vague hedging does not work. What works is
an explicit, narrowly scoped prohibition: name the specific inference it must not make, in the
specific circumstance, and state that "unresolved at this resolution" is a correct verdict.

Scope it narrowly. A broader prohibition against naming nodes caused a suspected false negative,
because removing that signal also removed the evidence that degradation existed at all. Prohibit
the conclusion, not the observation.

And make sure the evaluation accepts abstention as correct, or you will tune the agent straight
back out of the behavior you just taught it. The eval ground truth initially named a specific
dead rank, which scored the correct abstention as a failure.

### Design the evaluation before tuning

The obvious eval design leaks. You tune `SKILL.md` and the reference files until the agent
recognizes the faults you documented into it, which is close to tautological. The score will be
real and it will not support a claim about generalization.

What helps:

- Hold out variants. Tune on a fault at one component, evaluate at another. Better still, hold
  back an entire fault type and never write it into the knowledge base, which tells you what the
  agent does when it meets something unfamiliar.
- Decide what correct means up front. Fault type only, or type plus location? The second is much
  harder and much more meaningful, and sometimes the correct answer is that location is not
  determinable.
- Small n lies. Four out of five against three out of five swings on one draw. A harness that
  injects a random fault, waits, invokes the agent, scores it and restores makes twenty trials
  cheap.
- Export raw metric windows per trial, not screenshots. A playbook wants screenshots, an eval
  wants labeled data.
- The grader prompt is load bearing. Scoring with a second headless call works, but that prompt
  needs as much care as the skill.

With a non deterministic component in the loop, revert to the last state that scored well rather
than iterating forward from a broken one. Two changes interacting is expensive to untangle.

## Extending to another chat platform

Slack is planned. The coupling is small and sits at both ends: the diagnosis is text and one
function posts it, and the interactive commands are a thin surface over the same mode flag. A
second platform means another implementation of each, selected by which credentials are
configured, so an existing deployment behaves exactly as it does now.

Formatting is the work. Diagnoses come out as Markdown, and platforms differ in dialect, in
message length caps, and in how long content is best delivered. Each needs its own formatter.

Transport is not a problem. Posting needs no inbound connectivity, and platforms offering an
outbound WebSocket mode for interactive features can be driven from behind a firewall.

## Open questions

- Where should the line sit between wrapper only access and open queries?
- Can anything separate a throttle induced abort from a deliberate rank kill? It needs the
  pre-abort trajectory, which snapshot based tooling discards.
- Does any of this survive a compute bound cluster or a real fabric? Every "invisible to the
  profiler" finding may reverse.
- What does the agent do with a fault type it has never seen?
- Historical and offset queries are the obvious next tooling gap.
