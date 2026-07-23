# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
# Fault-injection harness for the OpenMosaic profiler stack.
#
# Breaks a running multi-GPU collective workload in reproducible ways so each
# failure's metric signature can be recorded. Advances issue #31; usable as an
# integration-test fixture.
#
#  SETUP
#   1. cp config.mk.example config.mk   and edit it for your cluster
#   2. run `make` ON THE HEAD NODE (the node you launch the workload from)
#   3. see README.md for prerequisites (SSH, sudo, docker)
#
# Every site-specific value lives in config.mk. Hardcode nothing here.

-include config.mk

# Defaults (override in config.mk, NOT here)
# Cluster topology
NODES      ?= node-a node-b
IFACE      ?= eth0
# RANK_HOSTS: rank -> node.  RANK_GPUS: rank -> local GPU index.
RANK_HOSTS ?= 0:node-a 1:node-a 2:node-b 3:node-b
RANK_GPUS  ?= 0:0 1:1 2:0 3:1

# Observability
PROM           ?= http://localhost:9090
COLLECTOR_HOST ?= node-a
COLLECTOR_CT   ?= otel-lgtm
COLLECTOR_PROC ?= otelcol-contrib
THROUGHPUT_METRIC ?= nccl_profiler_collective_bytes_total

# Workload (reference launcher, see config.mk.example). Only WORKLOAD_PROC is
# needed to detect the job; WORKLOAD_CMD/LAUNCHER are for `make workload`.
WORKLOAD_CMD      ?=
WORKLOAD_PROC     ?= all_reduce_perf
WORKLOAD_LAUNCHER ?= mpirun
LOG               ?= workload-$(shell date -u +%Y%m%dT%H%M%SZ).log

# Fault parameters (all overridable on the command line). NODE defaults to the
# first non-collector node. CLK: clamp (MHz). MS: delay. PCT: loss.
# DEADMAN: auto-restore window (s), 0 disables. INTERVAL: watch period.
NODE     ?= $(word 1,$(filter-out $(COLLECTOR_HOST),$(NODES)))
CLK      ?= 800
MS       ?= 20
PCT      ?= 1
DEADMAN  ?= 300
INTERVAL ?= 10

# SSH: -t so sudo can prompt. SSH_Q: quiet, no-sudo checks.
SSH      ?= ssh -t -o LogLevel=ERROR
SSH_Q    ?= ssh -o BatchMode=yes -o LogLevel=ERROR

# Internals
rank_host = $(word 2,$(subst :, ,$(filter $(1):%,$(RANK_HOSTS))))
rank_gpu  = $(word 2,$(subst :, ,$(filter $(1):%,$(RANK_GPUS))))

# Where each node records the PGIDs of its armed timers. restore kills exactly
# these groups, never a pattern match, so it can only cancel timers this harness
# armed.
PIDFILE  ?= /tmp/mosaic-fault-timers.pgids

# arm_deadman: runs `sleep DEADMAN; <cmd>` detached and records its PGID so
# restore can kill the group later. Usage: $(call arm_deadman,<cmd>)
arm_deadman = setsid sh -c "sleep $(DEADMAN); $(1)" >/dev/null 2>&1 & echo \$$! >> $(PIDFILE)

.DEFAULT_GOAL := help   

# Refuse to run against a missing or placeholder config.mk, so a half-edited
# config fails loudly instead of silently targeting placeholder hosts.
_check-config:
	@if [ ! -f config.mk ]; then \
	  echo "ERROR: config.mk not found. Run: cp config.mk.example config.mk  then edit it."; \
	  exit 1; \
	fi
	@bad=""; \
	for v in $(NODES) $(COLLECTOR_HOST) $(IFACE); do \
	  case "$$v" in \
	    node-a|node-b|eth0|otel-lgtm) bad="$$bad $$v" ;; \
	  esac; \
	done; \
	if [ -n "$$bad" ]; then \
	  echo "ERROR: config.mk still has placeholder value(s):$$bad"; \
	  echo "       Edit config.mk with your real cluster values (see config.mk.example)."; \
	  echo "       If you edited it and still see this, check for DUPLICATE lines,"; \
	  echo "       a later placeholder line overrides your value (last assignment wins)."; \
	  exit 1; \
	fi

.PHONY: help status workload kill-workload watch restore \
        inject-netem-delay inject-netem-loss \
        inject-kill-collector inject-kill-rank inject-slow-gpu

help:
	@echo "OpenMosaic fault-injection harness   (run on the head node)"
	@echo
	@echo "  make status                         what is running / injected, per node"
	@echo "  make workload                       start the workload (WORKLOAD_CMD)"
	@echo "  make kill-workload                  stop it everywhere"
	@echo "  make watch                          live per-rank throughput"
	@echo
	@echo "  make inject-slow-gpu RANK=n [CLK=mhz]   fault 1  clamp one GPU clock"
	@echo "  make inject-netem-loss PCT=p            fault 2  packet loss on NODE"
	@echo "  make inject-netem-delay MS=n            fault 2  added latency on NODE"
	@echo "  make inject-kill-rank RANK=n            fault 3  kill one rank (aborts job)"
	@echo "  make inject-kill-collector              fault 4  kill the OTel collector"
	@echo "  make restore                            clear every fault on every node"
	@echo
	@echo "  DEADMAN=<s>   auto-restore window (default $(DEADMAN)s); DEADMAN=0 disables"
	@echo
	@echo "Note: rate()[1m] on the throughput metric aliases by ~one scrape-worth"
	@echo "per window (often ~10%). Treat sub-10% moves as noise; measure faults"
	@echo "against a same-run baseline, not a fixed number."

status:
	@echo "== workload ranks (proc: $(WORKLOAD_PROC))"
	@for n in $(NODES); do \
	  printf '  %-10s %s\n' "$$n" "$$($(SSH_Q) $$n 'pgrep -xc $(WORKLOAD_PROC) || true')"; done
	@echo "== qdisc"
	@for n in $(NODES); do \
	  printf '  %-10s %s\n' "$$n" "$$($(SSH_Q) $$n tc qdisc show dev $(IFACE) | head -1)"; done
	@echo "== armed deadmen (auto-restore timers)"
	@for n in $(NODES); do \
	  s=$$($(SSH_Q) $$n 'if [ -s $(PIDFILE) ]; then echo "may be pending, run make restore to clear"; else echo none; fi'); \
	  printf '  %-10s %s\n' "$$n" "$$s"; done

# Runs WORKLOAD_CMD (from config.mk) detached. Replace it, or this target, for
# your own workload.
#
# Must be one long-lived process, not a restart loop: a loop rebuilds the
# communicator each iteration and resets the profiler's two-window
# stabilization, so nothing exports.
#
# Refuses a second concurrent job: two jobs emit series with identical
# hostname+rank labels that silently interleave.
workload: _check-config
	@if [ -z "$(strip $(WORKLOAD_CMD))" ]; then \
	  echo "ERROR: WORKLOAD_CMD is empty. Set it in config.mk (see config.mk.example)."; exit 1; fi
	@for n in $(NODES); do \
	  if [ "$$($(SSH_Q) $$n 'pgrep -xc $(WORKLOAD_PROC) || true')" != "0" ]; then \
	    echo "ERROR: workload already running on $$n. Stop it first: make kill-workload"; exit 1; fi; done
	@echo "== workload start $$(date -u +%FT%TZ)"
	@nohup sh -c '$(WORKLOAD_CMD)' > $(LOG) 2>&1 &
	@echo "   log: $(LOG)"
	@echo "   allow time for the profiler's two-window stabilization before judging metrics"
	@echo "   rate()[1m] ramps for ~60s after start: early readings understate throughput"

kill-workload:
	@echo "== kill workload $$(date -u +%FT%TZ)"
	@pkill -f $(WORKLOAD_LAUNCHER) 2>/dev/null || true
	@sleep 5
	@for n in $(NODES); do \
	  printf '  %-10s %s rank(s) remaining\n' "$$n" "$$($(SSH_Q) $$n 'pgrep -xc $(WORKLOAD_PROC) || true')"; done

# Live per-rank throughput. A single reading can't tell a ramp from a fault; a
# trend can.
watch:
	@echo "== watch (Ctrl-C to stop). Compare against your own steady-state baseline."
	@while true; do \
	  printf '%s  ' "$$(date -u +%T)"; \
	  curl -s '$(PROM)/api/v1/query' --data-urlencode \
	    'query=rate($(THROUGHPUT_METRIC)[1m])' \
	  | jq -j '[.data.result[] | "\(.metric.hostname)/\(.metric.rank)=\(.value[1]|tonumber/1e6*10|round/10)"] | sort | join("  ")'; \
	  echo; \
	  sleep $(INTERVAL); \
	done

# fault 2  network latency / loss
# inject-netem-delay MS=n and inject-netem-loss PCT=p share the _netem injector
# below. For a raw expression: make _netem NETEM="delay 50ms limit 5000"
#
# The deadman is armed before the fault in the same root shell, so one sudo
# covers both. Note you reach NODE over the interface you are degrading.
#
# NOTE netem's default queue is `limit 1000` packets. Exceed the bandwidth-delay
# product and the queue overflows, turning a `delay` into heavy loss. Find the
# usable range by bisection.
_netem: _check-config
	@if [ -z "$(NETEM)" ]; then echo "ERROR: use make inject-netem-loss PCT=p or inject-netem-delay MS=n"; exit 1; fi
	@echo "== inject netem [$(NETEM)] on $(NODE)  $$(date -u +%FT%TZ)"
	@if [ "$(DEADMAN)" = "0" ]; then \
	  echo "   *** NO DEADMAN. Persists until 'make restore'. You reach $(NODE) over $(IFACE)."; \
	  $(SSH) $(NODE) 'sudo sh -c "tc qdisc replace dev $(IFACE) root netem $(NETEM); tc qdisc show dev $(IFACE) | head -1"'; \
	else \
	  $(SSH) $(NODE) 'sudo sh -c "nohup setsid sh -c \"sleep $(DEADMAN); tc qdisc del dev $(IFACE) root\" >/dev/null 2>&1 & echo \$$! >> $(PIDFILE); tc qdisc replace dev $(IFACE) root netem $(NETEM); tc qdisc show dev $(IFACE) | head -1"'; \
	  echo "   deadman: auto-restore in $(DEADMAN)s"; \
	fi

inject-netem-delay: NETEM = delay $(MS)ms
inject-netem-delay: _netem

inject-netem-loss: NETEM = loss $(PCT)%
inject-netem-loss: _netem

# fault 4  kill the OTel collector
# pkill -x (exact name), NOT pkill -f: a workload launched with
# NCCL_PROFILER_PLUGIN=otel has "otel" in its command line.
#
# Kills only the collector, leaving Prometheus/Grafana (same container) alive,
# so the fault is observable while it happens.
#
# Signature: profiler series stop, up{job="<collector>"} -> 0, other exporters
# stay 1, workload keeps running. A metrics gap is not a cluster failure.
inject-kill-collector: _check-config
	@echo "== inject kill-collector [$(COLLECTOR_PROC)] on $(COLLECTOR_HOST)  $$(date -u +%FT%TZ)"
	@if [ "$(DEADMAN)" = "0" ]; then \
	  echo "   *** NO DEADMAN. Collector stays dead until 'make restore'."; \
	  $(SSH) $(COLLECTOR_HOST) 'sudo pkill -x $(COLLECTOR_PROC)'; \
	else \
	  $(SSH) $(COLLECTOR_HOST) 'sudo sh -c "nohup setsid sh -c \"sleep $(DEADMAN); docker restart $(COLLECTOR_CT)\" >/dev/null 2>&1 & echo \$$! >> $(PIDFILE); pkill -x $(COLLECTOR_PROC)"'; \
	  echo "   deadman: auto-restart of $(COLLECTOR_CT) in $(DEADMAN)s"; \
	fi
	@sleep 2
	@printf '   %s %s process(es) remaining (expect 0)\n' \
	  "$$($(SSH_Q) $(COLLECTOR_HOST) 'pgrep -xc $(COLLECTOR_PROC) || true')" "$(COLLECTOR_PROC)"

# fault 3  kill one rank
# Resolves rank -> gpu -> uuid -> compute PID so it hits exactly that rank;
# pgrep -f on the binary would also match the launcher and kill the whole job.
#
# NCCL has no fault tolerance, so one dead rank aborts everything.
# Not self-healing, recover with `make workload`. No deadman (nothing to undo).
#
# Inverse of fault 4: here the collector stays up and all GPUs go idle.
inject-kill-rank: _check-config
	@if [ -z "$(RANK)" ]; then echo "ERROR: RANK required, e.g. make inject-kill-rank RANK=3"; exit 1; fi
	$(eval H := $(call rank_host,$(RANK)))
	$(eval G := $(call rank_gpu,$(RANK)))
	@if [ -z "$(H)" ]; then echo "ERROR: RANK=$(RANK) not in RANK_HOSTS"; exit 1; fi
	@echo "== inject kill-rank $(RANK) -> $(H) gpu $(G)  $$(date -u +%FT%TZ)"
	@$(SSH_Q) $(H) 'U=$$(nvidia-smi -i $(G) --query-gpu=uuid --format=csv,noheader); \
	  P=$$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F", " -v u="$$U" "\$$2==u {print \$$1}"); \
	  if [ -z "$$P" ]; then echo "  no compute process on $(H) gpu $(G)"; exit 1; fi; \
	  echo "  killing pid $$P on $(H)"; kill -9 $$P'
	@echo "  job aborted. recover with: make workload"

# fault 1  clamp one GPU's clock 
# Locks one GPU to CLK MHz with `nvidia-smi -lgc`; the deadman runs -rgc.
#
# The only asymmetric fault: it hits one GPU, not all ranks. On a network-bound
# workload the clamped GPU still finishes before the link is ready, so
# throughput barely moves and the signature lives only in DCGM, not the
# profiler. Clamp too hard and the job aborts; find the range by bisection.
inject-slow-gpu: _check-config
	@if [ -z "$(RANK)" ]; then echo "ERROR: RANK required, e.g. make inject-slow-gpu RANK=3"; exit 1; fi
	$(eval H := $(call rank_host,$(RANK)))
	$(eval G := $(call rank_gpu,$(RANK)))
	@if [ -z "$(H)" ]; then echo "ERROR: RANK=$(RANK) not in RANK_HOSTS"; exit 1; fi
	@echo "== inject slow-gpu rank $(RANK) -> $(H) gpu $(G) @ $(CLK)MHz  $$(date -u +%FT%TZ)"
	@if [ "$(DEADMAN)" = "0" ]; then \
	  echo "   *** NO DEADMAN. Clock stays clamped until 'make restore'."; \
	  $(SSH) $(H) 'sudo sh -c "nvidia-smi -i $(G) -lgc $(CLK),$(CLK)"'; \
	else \
	  $(SSH) $(H) 'sudo sh -c "nohup setsid sh -c \"sleep $(DEADMAN); nvidia-smi -i $(G) -rgc\" >/dev/null 2>&1 & echo \$$! >> $(PIDFILE); nvidia-smi -i $(G) -lgc $(CLK),$(CLK)"'; \
	  echo "   deadman: auto-release in $(DEADMAN)s"; \
	fi
	@$(SSH_Q) $(H) 'nvidia-smi -i $(G) --query-gpu=clocks.max.sm,clocks.gr --format=csv,noheader' | sed "s/^/   now: /"

# Restore 
# Idempotent: safe when nothing is injected, safe to run twice. Clears netem,
# resets all GPU clocks, cancels timers, restarts the collector if dead.
#
# ssh output is not piped and stderr is not suppressed: a pipe swallows sudo's
# password prompt and you type blind.
restore:
	@echo "== restore $$(date -u +%FT%TZ)"
	@for n in $(NODES); do \
	  echo "  -- $$n (netem + gpu clocks + timers)"; \
	  $(SSH) $$n 'sudo sh -c "if tc qdisc show dev $(IFACE) | grep -q netem; then tc qdisc del dev $(IFACE) root && echo \"    netem cleared\"; else echo \"    no netem present\"; fi; for g in \$$(nvidia-smi --query-gpu=index --format=csv,noheader); do nvidia-smi -i \$$g -rgc >/dev/null 2>&1; done; echo \"    gpu clocks released\"; if [ -f $(PIDFILE) ]; then while read g; do kill -TERM -\$$g 2>/dev/null; done < $(PIDFILE); rm -f $(PIDFILE); fi; echo \"    pending timers cancelled\""'; \
	done
	@echo "  -- $(COLLECTOR_HOST) collector"
	@if [ "$$($(SSH_Q) $(COLLECTOR_HOST) 'pgrep -xc $(COLLECTOR_PROC) || true')" = "0" ]; then \
	  $(SSH_Q) $(COLLECTOR_HOST) 'docker restart $(COLLECTOR_CT) >/dev/null' && \
	  echo "    $(COLLECTOR_PROC) was dead, $(COLLECTOR_CT) restarted"; \
	else \
	  echo "    $(COLLECTOR_PROC) healthy, left alone"; \
	fi
	@echo "  NOTE restore != recovery: clearing a heavy fault removes the cause,"
	@echo "       but the job may take minutes to recover or need 'make workload'."
	@echo "  NOTE 'docker restart' preserves the TSDB; 'docker compose down' destroys it."
