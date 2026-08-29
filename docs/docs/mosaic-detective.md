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

[^1]: Named after the legendary analyst penguin from the *Madagascar* films, whose catchphrase is
    "Kowalski, analysis!" That is also the phrase that triggers the skill.



![Kowalski architecture](images/kowalski-architecture.png)

Installing it is covered in [Fault Detective Deployment](mosaic-detective-deployment.md).

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
reaches a posted diagnosis in about two minutes.

### Grafana alert rules

Alert rules decide that something is wrong, not what is wrong. Nothing from the alert reaches
the agent beyond the rule title, which is used for the chat message.

[Grafana Alerting](alerting.md) has the expressions and how to calibrate them.

### `kowalski.py`

One process running the chat bot and the HTTP receiver, sharing an in-memory mode flag:
`ARMED` diagnoses and posts, `NOTIFY` announces the alert and waits to be asked, `DISARMED`
posts nothing. It starts `ARMED` and returns there on every restart.

`Kowalski, analysis` runs a diagnosis in any mode, `DISARMED` included. Disarming stops
autonomous diagnosis, not the bot.

A global three minute cooldown follows any alert, and a non-blocking lock means a manual summon
during a run is told it is already covered rather than starting a second agent. Diagnosis runs
on a daemon thread so the handler can return immediately, or Grafana times out and retries. Resolved alerts are dropped, and
a non-zero exit from `claude -p` is caught and posted rather than read as a successful run with
nothing to say.

### The skill

A Claude Code Agent Skill loaded as a unit: `SKILL.md` with the trigger and the diagnostic
procedure, alongside `fault-signatures.md` and `metrics-reference.md`. Those files are not
documentation about what the agent does. They are what it does. See [The reference documents
are the program](mosaic-detective-deployment.md#the-reference-documents-are-the-program).

The skill runs from its installed location, not the repository. `sync-skill.sh` copies it into
place, and an unsynced edit has not taken effect.

### Wrapper scripts

The agent does not write queries. It picks among four read-only scripts in `scripts/`:

| Script | Question it answers |
|---|---|
| `check_collector_health.py` | Is the observability pipeline itself up? |
| `list_metrics.py` | What metrics exist, and which are trusted? |
| `metric_timeline.py` | How has one metric moved over a window? |
| `compare_ranks.py` | Do the ranks or links differ from each other? |

`metric_timeline.py` scans for rate-of-change transitions, catching both a frozen counter and a
gauge stepping down. `compare_ranks.py` groups by rank or link, comparing means for gauges and
deltas for counters.

### `mosaic_queries.py`

Connections, query construction, typed results, and the metric names the wrappers use held as
constants. No diagnostic logic lives here, so a bad diagnosis traces to a bad query or a bad
inference, never both at once.

## Design decisions

### Guided, not caged

The agent picks the investigative move. It does not pick the metric, which is fixed inside each
wrapper in tested code. A doctor ordering a blood panel does not get to redefine what the blood
panel measures.

The boundary is deliberately left open. On one kill-rank run the agent went outside the wrappers
to query process counts, checking whether the processes had actually exited. It was right, and
the diagnosis was better for it. Wrappers-only is blind to anything you did not anticipate; open
query access is more capable and can fabricate. The current position takes the safe side.

### Detection deterministic, diagnosis agentic

Alert rules are cheap, fast and predictable, and run continuously. Agent invocations are none of
those. The deterministic gate means the expensive part only runs once something simple has
established there is a reason to look.

### Why not a decision tree

The [Fault Playbook](fault-playbook.md) is that tree, and for the faults written into it a
deterministic implementation is faster and cheaper. Run that first.

Every branch of a tree is a fault someone already knew about. The agent picks its next query
from what the last one returned, so it can investigate one nobody encoded — on the kill-rank run
above it went outside the wrappers to count processes, a step no branch contains. A tree falls
off the end and returns nothing; the agent returns what it ruled out and what evidence would
settle the rest.

Maintenance splits the same way. A new metric means rewriting branches. A new signature is a
paragraph.
### Fault injection stays outside

The injection harness is not packaged with the detective, and the detective has no write path to
the cluster. Every script it can run is a query.


## Limitations

- Discord is the only chat platform currently supported. Slack is planned; see [Extending to
  another chat platform](#extending-to-another-chat-platform).
- Some faults cannot be localized, and the skill says so rather than guessing. A network fault
  cannot be pinned to a node on a bulk-synchronous ring, and a killed rank cannot be identified
  at typical scrape resolution because the abort propagates within one scrape interval. The
  [Fault Playbook](fault-playbook.md) covers why.
- Combination faults are untested.
- Signatures are calibrated to one cluster: two nodes, four GPUs, 1 GbE, network bound. A GPU
  clock clamp is the clearest case , the clocks move and the alert rule sees it, but throughput
  does not, because the network is the constraint. On a compute bound cluster that reverses. The
  method transfers; the numbers do not.
## Extending to another chat platform

Slack is planned. The coupling is small and sits at both ends: the diagnosis is text and one
function posts it, and the interactive commands are a thin surface over the same mode flag. A
second platform is another implementation of each, selected by which credentials are configured.

Formatting is the work, since platforms differ in Markdown dialect and message length caps.
Transport is not: posting needs no inbound connectivity, and an outbound WebSocket mode can be
driven from behind a firewall.

## Future improvements

- Historical and offset queries are the obvious next tooling gap.
- An MCP server instead of wrapper scripts, giving the agent a typed tool surface rather than a
  shell command per verb.
- Very basic write access, so the agent can act on a diagnosis rather than only report it,
  restarting a rank or a node being the obvious first case.
- Widening the line between wrapper only access and open queries, so the agent can ask questions
  the wrappers do not anticipate.
- Separating a throttle induced abort from a deliberate rank kill. It needs the pre-abort
  trajectory, which snapshot based tooling discards.
- Handling a fault type the agent has never seen.
