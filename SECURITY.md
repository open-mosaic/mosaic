<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Security Policy

We take security seriously. This document describes how to report vulnerabilities and what you can expect from us.

## Reporting a Vulnerability

**Do not report security vulnerabilities in public GitHub issues.** Public issues can expose users before a fix is available.

To report a security vulnerability:

1. Open a **private** security advisory on GitHub: **Security** → **Report a vulnerability**.
2. **Include**:
   - Description of the vulnerability and affected component (profiler plugin, exporters, deployment configs, etc.)
   - Steps to reproduce
   - Impact (e.g. information disclosure, denial of service, privilege escalation)
   - Suggested fix or mitigation, if any
   - Your name/handle for acknowledgment (optional)
3. **Allow time for response** – We aim to acknowledge within **5 business days** and will keep you updated on triage and remediation.

**Please do not disclose the issue publicly until we have had a chance to address it and, if applicable, release a fix or advisory.**

## Supported Versions

We provide security updates for:

| Version           | Supported   |
|-------------------|-------------|
| Latest release    | Yes         |
| Previous minor    | Best effort |
| Older             | No          |

When in doubt, upgrade to the latest release.

## Scope

**In scope:**

- The Mosaic profiler plugin
- OpenTelemetry export and configuration
- Deployment and exporter components in this repository
- Dependency vulnerabilities in our declared dependencies

**Out of scope:**

- Third-party observability backends (Grafana, Prometheus, etc.) except where we ship config or code for them
- NCCL, CUDA, or other upstream libraries beyond our use of their APIs
- General hardening or best-practice guidance (we welcome docs PRs for those)

## What to Expect

- **Acknowledgment** – We will confirm receipt of your report.
- **Triage** – We will assess severity and impact and decide on fixes and advisories.
- **Updates** – We will communicate progress and, with your permission, credit you in any advisory or release notes.
- **Fix and disclosure** – We will work toward a fix and coordinate disclosure timing with you when possible.

Thank you for helping keep Mosaic and its users safe.
