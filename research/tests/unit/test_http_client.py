import pytest

from prepump.net.client import BannedError, HttpClient, HttpError, NotFoundError, Response
from prepump.net.ratelimit import TokenBucket


def make_client(responses, **kwargs):
    calls = []

    def transport(url, params, timeout):
        calls.append((url, params))
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        return item

    slept = []
    client = HttpClient(
        bucket=TokenBucket(rate_per_second=1000.0, sleeper=lambda s: None),
        transport=transport,
        sleeper=slept.append,
        **kwargs,
    )
    return client, calls, slept


def ok(content=b"ok"):
    return Response(200, content, {}, "http://x")


def status(code, headers=None):
    return Response(code, b"", headers or {}, "http://x")


def test_success_returns_content():
    client, calls, _ = make_client([ok(b"payload")])
    assert client.get("http://x").content == b"payload"
    assert client.stats.requests == 1


def test_429_is_retried_then_succeeds():
    client, calls, slept = make_client([status(429), ok()], max_retries=3)
    responses = [status(429), ok()]
    sequence = iter(responses)
    client.transport = lambda url, params, timeout: next(sequence)
    assert client.get("http://x").status == 200
    assert client.stats.retries == 1
    assert client.stats.rate_limited == 1
    assert slept, "a 429 must actually back off, not spin"


def test_retry_after_header_wins_over_computed_backoff():
    sequence = iter([status(429, {"Retry-After": "30"}), ok()])
    client, _, slept = make_client([ok()], max_retries=3)
    client.transport = lambda url, params, timeout: next(sequence)
    client.get("http://x")
    assert slept[0] >= 30


def test_ban_is_never_retried():
    client, calls, _ = make_client([status(418)], max_retries=5)
    with pytest.raises(BannedError):
        client.get("http://x")
    assert len(calls) == 1


def test_policy_denial_is_never_retried():
    client, calls, _ = make_client([status(403)], max_retries=5)
    with pytest.raises(BannedError):
        client.get("http://x")
    assert len(calls) == 1


def test_retries_are_bounded():
    client, calls, _ = make_client([status(503)], max_retries=2)
    with pytest.raises(HttpError):
        client.get("http://x")
    assert len(calls) == 3  # initial attempt plus two retries


def test_backoff_is_capped():
    client, _, slept = make_client([status(503)], max_retries=6, backoff_base_seconds=1.0, backoff_max_seconds=5.0)
    with pytest.raises(HttpError):
        client.get("http://x")
    assert max(slept) <= 5.0


def test_404_is_not_an_error_condition_to_retry():
    client, calls, _ = make_client([status(404)], max_retries=3)
    with pytest.raises(NotFoundError):
        client.get("http://x")
    assert len(calls) == 1
    assert client.stats.not_found == 1


def test_transport_exception_is_retried_then_raises():
    attempts = []

    def transport(url, params, timeout):
        attempts.append(url)
        raise ConnectionError("reset")

    client = HttpClient(
        bucket=TokenBucket(rate_per_second=1000.0, sleeper=lambda s: None),
        transport=transport,
        sleeper=lambda s: None,
        max_retries=2,
    )
    with pytest.raises(HttpError):
        client.get("http://x")
    assert len(attempts) == 3


def test_file_urls_work_offline(tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"local")
    client = HttpClient(bucket=TokenBucket(rate_per_second=1000.0, sleeper=lambda s: None))
    assert client.get(f"file://{target}").content == b"local"
    with pytest.raises(NotFoundError):
        client.get(f"file://{tmp_path / 'missing.bin'}")
