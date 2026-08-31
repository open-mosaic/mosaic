<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Grafana alerting config for Kowalski

`alert-rules.yaml` — three rules in folder `Mosaic`, group `mosaic-10s`:
throughput drop, GPU clock divergence, collector down. Each routes directly
to the `Kowalski` contact point via `notification_settings`, so no
notification policy is needed.

`contact-points.yaml` — the `Kowalski` webhook, pointing at `kowalski:8500`.
That resolves as-is if the receiver runs on the same Docker network as
Grafana; change the URL if it runs elsewhere.

Both reference `datasourceUid: prometheus`, which is the fixed UID the
otel-lgtm image provisions, so they import unchanged against the official
deployment. The thresholds do not travel — they came from a two node, four
GPU cluster on 1 GbE. See [alerting.md](../../../docs/docs/alerting.md) for
how to calibrate them.

To install, copy both into `/etc/grafana/provisioning/alerting/` and restart
Grafana, or import through Alerting > Alert rules > Import.

Provisioned rules are read-only in the Grafana UI, and alerting provisioning
is not re-read from disk on an interval, so a changed file needs a restart or
`POST /api/admin/provisioning/alerting/reload`.