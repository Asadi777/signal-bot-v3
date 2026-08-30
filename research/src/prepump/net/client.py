"""HTTP client with bounded retries and hard stops.

Deliberate behaviours:

* 429 and 5xx are retried with exponential backoff plus jitter, up to a bounded
  number of attempts. ``Retry-After`` wins over the computed backoff.
* 418 (Binance ban) and 403 are **not** retried. Hammering a host that has just
  banned us is how a short throttle becomes a long one, and a 403 from the
  egress policy will never become a 200 by trying again.
* ``file://`` URLs are served from the local filesystem, so the exact same
  collector code paths run in a fully offline fixture mode.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urlparse, unquote

from prepump.logging import get_logger
from prepump.net.ratelimit import TokenBucket

log = get_logger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
FATAL_STATUS = {403, 418, 451}


class HttpError(Exception):
    def __init__(self, message: str, *, status: int | None = None, url: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.url = url


class BannedError(HttpError):
    """Non-retryable refusal: rate-limit ban, policy denial or geo block."""


class NotFoundError(HttpError):
    """The resource does not exist at the source (a 404, not a failure)."""


@dataclass
class Response:
    status: int
    content: bytes
    headers: dict
    url: str


@dataclass
class HttpStats:
    requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    not_found: int = 0
    bytes_downloaded: int = 0
    rate_limit_wait_seconds: float = 0.0
    backoff_wait_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "retries": self.retries,
            "rate_limited_responses": self.rate_limited,
            "not_found": self.not_found,
            "bytes_downloaded": self.bytes_downloaded,
            "rate_limit_wait_seconds": round(self.rate_limit_wait_seconds, 3),
            "backoff_wait_seconds": round(self.backoff_wait_seconds, 3),
        }


def _file_transport(url: str, params: dict | None, timeout: float) -> Response:
    path = Path(unquote(urlparse(url).path))
    if not path.exists() or path.is_dir():
        return Response(status=404, content=b"", headers={}, url=url)
    return Response(status=200, content=path.read_bytes(), headers={}, url=url)


def _requests_transport(url: str, params: dict | None, timeout: float) -> Response:
    import requests  # imported lazily so offline runs need no network stack

    resp = requests.get(url, params=params, timeout=timeout)
    return Response(status=resp.status_code, content=resp.content, headers=dict(resp.headers), url=resp.url)


@dataclass
class HttpClient:
    bucket: TokenBucket
    max_retries: int = 5
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 60.0
    timeout_seconds: float = 60.0
    transport: "callable | None" = None
    sleeper: "callable" = time.sleep
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    stats: HttpStats = field(default_factory=HttpStats)

    def get(self, url: str, params: dict | None = None) -> Response:
        """Fetch ``url``. Raises NotFoundError on 404, BannedError on a hard stop."""
        transport = self.transport or (_file_transport if url.startswith("file://") else _requests_transport)
        attempt = 0
        while True:
            self.stats.rate_limit_wait_seconds += self.bucket.acquire()
            self.stats.requests += 1
            try:
                resp = transport(url, params, self.timeout_seconds)
            except Exception as exc:  # transport-level failure: retryable
                if attempt >= self.max_retries:
                    raise HttpError(f"transport failure after {attempt} retries: {exc}", url=url) from exc
                attempt += 1
                self._sleep_backoff(attempt, None)
                continue

            if resp.status == 200:
                self.stats.bytes_downloaded += len(resp.content)
                return resp
            if resp.status == 404:
                self.stats.not_found += 1
                raise NotFoundError(f"not found: {_full_url(url, params)}", status=404, url=url)
            if resp.status in FATAL_STATUS:
                raise BannedError(
                    f"refused with {resp.status} (not retried): {_full_url(url, params)}",
                    status=resp.status,
                    url=url,
                )
            if resp.status in RETRYABLE_STATUS:
                if resp.status == 429:
                    self.stats.rate_limited += 1
                if attempt >= self.max_retries:
                    raise HttpError(
                        f"giving up after {attempt} retries, last status {resp.status}",
                        status=resp.status,
                        url=url,
                    )
                attempt += 1
                self._sleep_backoff(attempt, resp.headers.get("Retry-After"))
                continue
            raise HttpError(f"unexpected status {resp.status} for {_full_url(url, params)}", status=resp.status, url=url)

    def _sleep_backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = min(self.backoff_base_seconds * (2 ** (attempt - 1)), self.backoff_max_seconds)
        delay += self.rng.uniform(0, self.backoff_base_seconds)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        delay = min(delay, self.backoff_max_seconds)
        self.stats.retries += 1
        self.stats.backoff_wait_seconds += delay
        log.warning("retrying after backoff", extra={"attempt": attempt, "delay_seconds": round(delay, 3)})
        self.sleeper(delay)


def _full_url(url: str, params: dict | None) -> str:
    return f"{url}?{urlencode(params)}" if params else url


def build_client(settings, transport=None) -> HttpClient:
    return HttpClient(
        bucket=TokenBucket(rate_per_second=settings.rate_limit_rps),
        max_retries=settings.max_retries,
        backoff_base_seconds=settings.backoff_base_seconds,
        backoff_max_seconds=settings.backoff_max_seconds,
        timeout_seconds=settings.request_timeout_seconds,
        transport=transport,
    )
