# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Query layer for the Mosaic Detective skill."""

from __future__ import annotations

import os

import requests

from dataclasses import dataclass


@dataclass
class RangeSeries:
    labels: dict
    values: list

@dataclass
class InstantSample:
    labels: dict
    value: tuple


class MosaicQueryError(RuntimeError):
    """Raised for any transport, HTTP, or Prometheus-level query failure."""


class MosaicClient:
    def __init__(self, base_url: str | None = None, timeout: float = 10.0,
                 session: requests.Session | None = None):
        if base_url is None:
            host = os.getenv("PROMETHEUS_HOST", "localhost")
            port = os.getenv("PROMETHEUS_PORT", "9090")
            base_url = f"http://{host}:{port}"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session if session is not None else requests.Session()

    def query_range(self, query: str, start, end, step: str = "15s") -> dict:
        return self._get("/api/v1/query_range", {
            "query": query,
            "start": start,
            "end": end,
            "step": step,
        })

    def query_instant(self, query: str, time=None) -> dict:
        return self._get("/api/v1/query", {"query": query, "time": time})

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise MosaicQueryError(f"cannot reach Prometheus at {self.base_url}: {exc}") from exc

        if resp.status_code != 200:
            raise MosaicQueryError(
                f"Prometheus API error at {self.base_url}: HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise MosaicQueryError(f"invalid JSON from Prometheus at {self.base_url}: {exc}") from exc

        if payload.get("status") != "success":
            raise MosaicQueryError(f"Prometheus query failed at {self.base_url}: {payload.get('error')}")

        return payload.get("data", {})


def parse_range(data):
    series_list = []
    for entry in data.get("result", []):
        labels = entry["metric"]
        values = [(float(ts), float(val)) for ts, val in entry["values"]]
        series_list.append(RangeSeries(labels, values))
    return series_list

def parse_instant(data):
    series_list = []
    for entry in data.get("result",[]):

        labels = entry["metric"]
        ts, val = entry["value"] 
        value = (float(ts), float(val))    

        series_list.append(InstantSample(labels, value))
        
    return series_list
