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