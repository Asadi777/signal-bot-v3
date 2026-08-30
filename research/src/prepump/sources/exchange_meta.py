"""Exchange symbol and market metadata.

Purpose (Kickoff §C): capture what is needed to reconstruct universe
membership later. Today's snapshot is not history, so the collector is
append-only by collection date: a series of dated snapshots is the only
honest approximation of "which symbols existed when" that a live metadata
endpoint can give us.

Where a field the spec asks for does not exist at the source, the row records
an explicit gap instead of a guess. Binance spot metadata, for example,
carries no listing date at all; the futures endpoint does.
"""

from __future__ import annotations

import json

import pyarrow as pa

from prepump.io.checksums import sha256_bytes
from prepump.net.client import HttpClient
from prepump.normalize.symbols import SymbolSpec, make_spec
from prepump.timeutils import utc_now_ms
from prepump.version import COLLECTOR_VERSION, METADATA_SCHEMA_VERSION

TS = pa.timestamp("ms", tz="UTC")

METADATA_SCHEMA = pa.schema(
    [
        pa.field("exchange", pa.string(), nullable=False),
        pa.field("market_type", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("symbol_canonical", pa.string(), nullable=False),
        pa.field("base_asset", pa.string(), nullable=False),
        pa.field("quote_asset", pa.string(), nullable=False),
        pa.field("settle_asset", pa.string(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
        pa.field("contract_type", pa.string(), nullable=True),
        pa.field("listing_time", TS, nullable=True),
        pa.field("listing_time_method", pa.string(), nullable=False),
        pa.field("delisting_time", TS, nullable=True),
        pa.field("tick_size", pa.float64(), nullable=True),
        pa.field("step_size", pa.float64(), nullable=True),
        pa.field("min_notional", pa.float64(), nullable=True),
        pa.field("contract_specs_json", pa.string(), nullable=False),
        pa.field("known_gaps", pa.list_(pa.string()), nullable=False),
        pa.field("collected_at", TS, nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("collector_version", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
    ]
)

# Values the exchanges use; kept verbatim rather than mapped to a project
# vocabulary, because a lossy status mapping would be invisible later.
LISTING_METHOD_EXCHANGE = "EXCHANGE_PROVIDED"
LISTING_METHOD_UNAVAILABLE = "UNAVAILABLE_AT_SOURCE"


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filter_value(filters: list[dict], filter_type: str, key: str) -> float | None:
    for entry in filters or []:
        if entry.get("filterType") == filter_type:
            return _num(entry.get(key))
    return None


def collect_binance_spot(client: HttpClient, base_url: str, run_id: str) -> list[dict]:
    body = client.get(base_url.rstrip("/") + "/api/v3/exchangeInfo").content
    payload = json.loads(body)
    revision = f"sha256:{sha256_bytes(body)}"
    now = utc_now_ms()
    rows = []
    for item in payload.get("symbols", []):
        spec = make_spec("binance", "spot", item["symbol"], item.get("baseAsset"), item.get("quoteAsset"))
        rows.append(
            _row(
                spec,
                status=item.get("status", "UNKNOWN"),
                contract_type=None,
                settle_asset=None,
                listing_time=None,
                listing_time_method=LISTING_METHOD_UNAVAILABLE,
                tick_size=_filter_value(item.get("filters", []), "PRICE_FILTER", "tickSize"),
                step_size=_filter_value(item.get("filters", []), "LOT_SIZE", "stepSize"),
                min_notional=_filter_value(item.get("filters", []), "NOTIONAL", "minNotional"),
                specs=item,
                gaps=[
                    "listing_time: spot exchangeInfo exposes no listing/onboard date",
                    "delisting_time: delisted symbols disappear from this endpoint entirely",
                ],
                collected_at=now,
                source="binance_spot_exchange_info",
                revision=revision,
                run_id=run_id,
            )
        )
    return rows


def collect_binance_futures(client: HttpClient, base_url: str, run_id: str) -> list[dict]:
    body = client.get(base_url.rstrip("/") + "/fapi/v1/exchangeInfo").content
    payload = json.loads(body)
    revision = f"sha256:{sha256_bytes(body)}"
    now = utc_now_ms()
    rows = []
    for item in payload.get("symbols", []):
        contract_type = item.get("contractType") or ""
        market_type = "linear_perpetual" if contract_type.upper() == "PERPETUAL" else "futures"
        spec = make_spec("binance", market_type, item["symbol"], item.get("baseAsset"), item.get("quoteAsset"))
        onboard = item.get("onboardDate")
        delivery = item.get("deliveryDate")
        rows.append(
            _row(
                spec,
                status=item.get("status") or item.get("contractStatus") or "UNKNOWN",
                contract_type=contract_type or None,
                settle_asset=item.get("marginAsset"),
                listing_time=int(onboard) if onboard else None,
                listing_time_method=LISTING_METHOD_EXCHANGE if onboard else LISTING_METHOD_UNAVAILABLE,
                tick_size=_filter_value(item.get("filters", []), "PRICE_FILTER", "tickSize"),
                step_size=_filter_value(item.get("filters", []), "LOT_SIZE", "stepSize"),
                min_notional=_filter_value(item.get("filters", []), "MIN_NOTIONAL", "notional"),
                specs=item,
                gaps=[] if onboard else ["listing_time: onboardDate absent for this contract"],
                collected_at=now,
                source="binance_futures_exchange_info",
                revision=revision,
                run_id=run_id,
                delisting_time=int(delivery) if delivery and int(delivery) < 4_102_444_800_000 else None,
            )
        )
    return rows


def collect_bybit(client: HttpClient, base_url: str, category: str, run_id: str) -> list[dict]:
    market_type = {"spot": "spot", "linear": "linear_perpetual", "inverse": "inverse_perpetual"}[category]
    body = client.get(base_url.rstrip("/") + "/v5/market/instruments-info", params={"category": category, "limit": 1000}).content
    payload = json.loads(body)
    if payload.get("retCode") not in (0, None):
        raise ValueError(f"bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
    revision = f"sha256:{sha256_bytes(body)}"
    now = utc_now_ms()
    rows = []
    for item in (payload.get("result") or {}).get("list", []):
        contract = item.get("contractType") or ""
        resolved = market_type
        if category == "linear" and contract and contract.upper() != "LINEARPERPETUAL":
            resolved = "futures"
        spec = make_spec("bybit", resolved, item["symbol"], item.get("baseCoin"), item.get("quoteCoin"))
        launch = item.get("launchTime")
        launch_ms = int(launch) if launch and str(launch).isdigit() and int(launch) > 0 else None
        price_filter = item.get("priceFilter") or {}
        lot_filter = item.get("lotSizeFilter") or {}
        rows.append(
            _row(
                spec,
                status=item.get("status", "UNKNOWN"),
                contract_type=contract or None,
                settle_asset=item.get("settleCoin"),
                listing_time=launch_ms,
                listing_time_method=LISTING_METHOD_EXCHANGE if launch_ms else LISTING_METHOD_UNAVAILABLE,
                tick_size=_num(price_filter.get("tickSize")),
                step_size=_num(lot_filter.get("qtyStep") or lot_filter.get("basePrecision")),
                min_notional=_num(lot_filter.get("minNotionalValue") or lot_filter.get("minOrderAmt")),
                specs=item,
                gaps=[] if launch_ms else ["listing_time: launchTime absent or zero"],
                collected_at=now,
                source=f"bybit_instruments_info_{category}",
                revision=revision,
                run_id=run_id,
            )
        )
    return rows


def _row(
    spec: SymbolSpec,
    *,
    status: str,
    contract_type: str | None,
    settle_asset: str | None,
    listing_time: int | None,
    listing_time_method: str,
    tick_size: float | None,
    step_size: float | None,
    min_notional: float | None,
    specs: dict,
    gaps: list[str],
    collected_at: int,
    source: str,
    revision: str,
    run_id: str,
    delisting_time: int | None = None,
) -> dict:
    return {
        "exchange": spec.exchange,
        "market_type": spec.market_type,
        "symbol": spec.symbol,
        "symbol_canonical": spec.canonical,
        "base_asset": spec.base_asset,
        "quote_asset": spec.quote_asset,
        "settle_asset": settle_asset,
        "status": status,
        "contract_type": contract_type,
        "listing_time": listing_time,
        "listing_time_method": listing_time_method,
        "delisting_time": delisting_time,
        "tick_size": tick_size,
        "step_size": step_size,
        "min_notional": min_notional,
        "contract_specs_json": json.dumps(specs, sort_keys=True, separators=(",", ":")),
        "known_gaps": gaps,
        "collected_at": collected_at,
        "source": source,
        "source_revision": revision,
        "collector_version": COLLECTOR_VERSION,
        "schema_version": METADATA_SCHEMA_VERSION,
        "run_id": run_id,
    }


def rows_to_table(rows: list[dict]) -> pa.Table:
    arrays = [pa.array([row[field.name] for row in rows], type=field.type) for field in METADATA_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=METADATA_SCHEMA)
