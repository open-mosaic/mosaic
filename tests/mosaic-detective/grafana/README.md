# Grafana alerting config for Kowalski

Exported from a running Grafana 12.4.0 with:

    curl -s -u admin:admin \
      "http://<grafana>/api/v1/provisioning/alert-rules/export?format=yaml"
    curl -s -u admin:admin \
      "http://<grafana>/api/v1/provisioning/contact-points/export?format=yaml"

`alert-rules.yaml` — three rules in folder `Mosaic`, group `mosaic-10s`:
throughput drop, GPU clock divergence, collector down. Each routes directly
to the `Kowalski` contact point via `notification_settings`, so no
notification policy is needed.

`contact-points.yaml` — the `Kowalski` webhook. The URL is environment
specific and must be changed to wherever the receiver is listening.

To install, copy both into Grafana's provisioning directory
(`/etc/grafana/provisioning/alerting/`) and restart, or import through
Alerting > Alert rules > Import.

Note: file-provisioned rules are read-only in the Grafana UI.
