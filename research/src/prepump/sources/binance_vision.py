"""Binance public data archive (``data.binance.vision``).

Chosen as the primary backfill path over the REST klines endpoint because:

* one request returns a whole month instead of 1000 candles;
* it is a static file host, so a long backfill cannot earn an API ban;
* files carry a published SHA256, making a run verifiable rather than merely
  repeatable;
* delisted symbols remain downloadable, which is the only free way to satisfy
  the survivorship-bias rule of Master Spec §65.3.

Two robustness details matter here. The CSV has shipped both with and without a
header row, and the time columns have shipped in different units across
datasets and years; both are detected per file rather than assumed, and what
was detected is recorded in the raw sidecar.
"""

from __future__ import annotations

import csv
import io
import zipfile
from xml.etree import ElementTree

from prepump.io.checksums import sha256_bytes
from prepump.logging import get_logger
from prepump.net.client import HttpClient, NotFoundError
from prepump.normalize.ohlcv import RawKline
from prepump.normalize.schema import AvailabilityMethod
from prepump.normalize.symbols import SymbolSpec
from prepump.sources.base import FetchResult, FetchUnit, SourceDescriptor
from prepump.timeutils import detect_epoch_ms, detect_time_unit, month_range

log = get_logger(__name__)

MARKET_PATHS = {
    "spot": "spot",
    "linear_perpetual": "futures/um",
    "inverse_perpetual": "futures/cm",
}

# Column order of the kline CSV, stable across spot and futures dumps.
_COLUMNS = 12


class BinanceVisionSource:
    def __init__(self, client: HttpClient, base_url: str, interval: str = "1m") -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.interval = interval
        self.descriptor = SourceDescriptor(
            name="binance_vision_monthly_klines",
            exchange="binance",
            # A closed candle is usable the moment it closes; the archive file
            # merely republishes it later. Publication delay is a property of
            # our access path, not of when the market knew the value, so the
            # earliest defensible availability is derived from the close.
            availability_method=AvailabilityMethod.DERIVED_FROM_CLOSE,
            availability_delta_ms=1,
            # Plausible spread of exchange dissemination latency. Not measured
            # for historical data, and deliberately not claimed as measured.
            availability_uncertainty_ms=2000,
            notes="Monthly zipped CSV dumps; SHA256 published alongside each file.",
        )

    def supports(self, spec: SymbolSpec) -> bool:
        return spec.exchange == "binance" and spec.market_type in MARKET_PATHS

    def _market_path(self, spec: SymbolSpec) -> str:
        return MARKET_PATHS[spec.market_type]

    def file_url(self, spec: SymbolSpec, month: str) -> str:
        name = f"{spec.symbol}-{self.interval}-{month}.zip"
        return f"{self.base_url}/data/{self._market_path(spec)}/monthly/klines/{spec.symbol}/{self.interval}/{name}"

    def plan(self, spec: SymbolSpec, start_ms: int, end_ms: int) -> list[FetchUnit]:
        return [
            FetchUnit(
                key=f"{spec.exchange}:{spec.market_type}:{spec.symbol}:{month}",
                url=self.file_url(spec, month),
                label=month,
            )
            for month in month_range(start_ms, end_ms)
        ]

    def fetch(self, spec: SymbolSpec, unit: FetchUnit) -> FetchResult:
        payload = self.client.get(unit.url).content
        digest = sha256_bytes(payload)
        published = self._published_checksum(unit.url)
        if published and published != digest:
            raise ValueError(
                f"checksum mismatch for {unit.url}: published {published[:12]}..., got {digest[:12]}..."
            )
        klines, meta = parse_kline_zip(payload)
        meta["checksum_verified"] = bool(published)
        return FetchResult(
            klines=klines,
            source_revision=f"sha256:{digest}",
            raw_bytes=payload,
            raw_filename=unit.url.rsplit("/", 1)[-1],
            meta={"url": unit.url, "sha256": digest, "month": unit.label, **meta},
        )

    def _published_checksum(self, file_url: str) -> str | None:
        """Fetch the sibling ``.CHECKSUM`` file; absence is recorded, not fatal."""
        try:
            text = self.client.get(file_url + ".CHECKSUM").content.decode().strip()
        except NotFoundError:
            log.warning("no published checksum", extra={"url": file_url})
            return None
        return text.split()[0] if text else None


def parse_kline_zip(payload: bytes) -> tuple[list[RawKline], dict]:
    """Parse a Binance kline zip into raw klines plus what we detected."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected exactly one CSV in archive, found {names}")
        raw_text = archive.read(names[0]).decode("utf-8")

    rows = list(csv.reader(io.StringIO(raw_text)))
    rows = [r for r in rows if r and any(cell.strip() for cell in r)]
    if not rows:
        return [], {"csv_name": names[0], "rows": 0, "header_detected": False, "time_unit": None}

    header_detected = not _is_numeric(rows[0][0])
    if header_detected:
        rows = rows[1:]
    if not rows:
        return [], {"csv_name": names[0], "rows": 0, "header_detected": True, "time_unit": None}

    time_unit = detect_time_unit(int(float(rows[0][0])))
    klines = [_row_to_kline(row) for row in rows]
    return klines, {
        "csv_name": names[0],
        "rows": len(klines),
        "header_detected": header_detected,
        "time_unit": time_unit,
    }


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _row_to_kline(row: list[str]) -> RawKline:
    if len(row) < _COLUMNS - 1:
        raise ValueError(f"unexpected kline row width {len(row)}: {row[:4]}...")
    return RawKline(
        open_time_ms=detect_epoch_ms(int(float(row[0]))),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        base_volume=float(row[5]),
        close_time_ms=detect_epoch_ms(int(float(row[6]))),
        quote_volume=float(row[7]),
        trade_count=int(float(row[8])),
        taker_buy_base=float(row[9]),
        taker_buy_quote=float(row[10]),
    )


def list_prefix(client: HttpClient, base_url: str, prefix: str) -> tuple[list[str], list[str]]:
    """List an archive prefix, returning (sub-prefixes, object keys).

    Used to discover which symbols exist in the archive, including symbols that
    no longer trade and are therefore absent from the live REST metadata.
    """
    sub_prefixes: list[str] = []
    keys: list[str] = []
    marker: str | None = None
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        body = client.get(base_url.rstrip("/") + "/", params=params).content
        root = ElementTree.fromstring(body)
        ns = {"s3": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        find = (lambda tag: f"s3:{tag}") if ns else (lambda tag: tag)
        sub_prefixes.extend(el.text for el in root.findall(f".//{find('CommonPrefixes')}/{find('Prefix')}", ns) if el.text)
        keys.extend(el.text for el in root.findall(f".//{find('Contents')}/{find('Key')}", ns) if el.text)
        truncated = root.findtext(find("IsTruncated"), default="false", namespaces=ns)
        if truncated.lower() != "true":
            return sub_prefixes, keys
        marker = root.findtext(find("NextMarker"), namespaces=ns) or (keys[-1] if keys else None)
        if not marker:
            return sub_prefixes, keys
