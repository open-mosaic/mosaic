import pytest
import mosaic_queries as mq
import requests

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class FakeSession:
    def __init__(self, response):
        self._response = response

    def get(self, url, params=None, timeout=None):
        return self._response


def test_get_returns_data_on_success():
    body = {"status": "success", "data": {"resultType": "vector", "result": [1, 2, 3]}}
    client = mq.MosaicClient(session=FakeSession(FakeResponse(json_data=body)))

    result = client._get("/api/v1/query")

    assert result == {"resultType": "vector", "result": [1, 2, 3]}


def test_get_raises_on_connection_failure():
    class DeadSession:
        def get(self, url, params=None, timeout=None):
            raise requests.exceptions.ConnectionError("refused")
    client = mq.MosaicClient(session=DeadSession())

    with pytest.raises(mq.MosaicQueryError):
        client._get("/api/v1/query")


def test_get_raises_on_http_failure():
    client = mq.MosaicClient(session=FakeSession(FakeResponse(500)))

    with pytest.raises(mq.MosaicQueryError):
        client._get("/api/v1/query")


def test_get_raises_on_prometheus_error_status():
    body = {"status": "error", "error": "bad query"}
    client = mq.MosaicClient(session=FakeSession(FakeResponse(json_data=body)))

    with pytest.raises(mq.MosaicQueryError):
        client._get("/api/v1/query")

def test_query_range_calls_get_with_correct_params():
    client = mq.MosaicClient(session=FakeSession(FakeResponse(json_data={})))

    captured = {}
    def fake_get(path, params=None):
        captured["path"] = path
        captured["params"] = params
        return {"result": "sentinel"}
    client._get = fake_get

    result = client.query_range("up", start=100, end=200, step="15s")

    assert captured["path"] == "/api/v1/query_range"
    assert captured["params"] == {"query": "up", "start": 100, "end": 200, "step": "15s"}
    assert result == {"result": "sentinel"}

def test_query_instant_calls_get_with_correct_params():
    client = mq.MosaicClient(session=FakeSession(FakeResponse(json_data={})))

    captured = {}
    def fake_get(path, params=None):
        captured["path"] = path
        captured["params"] = params
        return {"result": "sentinel"}
    client._get = fake_get

    result = client.query_instant("up", 67)

    assert captured["path"] == "/api/v1/query"
    assert captured["params"] == {"query": "up", "time": 67}
    assert result == {"result": "sentinel"}


def test_parse_range_builds_typed_series():
    raw = {
        "resultType": "matrix",
        "result": [
            {"metric": {"__name__": "up", "rank": "0"},
             "values": [[1710000000, "1"], [1710000015, "0"]]},
        ],
    }

    result = mq.parse_range(raw)

    assert result == [
        mq.RangeSeries(
            labels={"__name__": "up", "rank": "0"},
            values=[(1710000000.0, 1.0), (1710000015.0, 0.0)],
        )
    ]


def test_parse_instant_sample():
    raw = {
        "resultType": "vector",
        "result": [
            {"metric": {"__name__": "up", "rank": "0"},
             "value": [1710000000, "1"]},
        ],
    }
    result = mq.parse_instant(raw)

    assert result == [
        mq.InstantSample(
            labels={"__name__": "up", "rank": "0"},
            value=(1710000000.0, 1.0),
        )
    ]