"""
Signal Bot — Phase 1 Watchlist Engine  v3.3
============================================
نسخه نهایی یکپارچه Phase 1

تغییرات اصلی نسبت به v3.2:
  1) Dedup fix واقعی:
     mark_sent فقط بعد از ارسال موفق Telegram اجرا می‌شود
  2) Version consistency:
     همه جا v3.3
  3) pump_4h:
     دیگر امتیاز مستقل نمی‌گیرد؛ فقط multiplier/validation
  4) Gate:
     نیازمند align شدن چند TF
  5) Memory safety:
     df_1h با finally cleanup می‌شود
  6) Hard invalidate:
     momentum خیلی قوی => setup رد می‌شود
  7) OI scoring:
     فقط وقتی absolute OI کافی باشد
  8) Weekly continuation risk:
     از هشدار صرف به penalty واقعی تبدیل شد

خروجی:
  🔴 Tier A Watch — ریسک برگشت بالا | تایید Phase 2 لازم است
  🟠 Tier B Watch — زیر نظر بگیر
  🟡 Tier C Watch — در رادار
  ⚪ No Trade
"""

import os
import sys
import time
import logging
import gc
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import ta


# ══════════════════════════════════════════════
# ENV CONFIG
# ══════════════════════════════════════════════
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("❌ TELEGRAM_TOKEN و TELEGRAM_CHAT_ID باید environment variable باشند")
    print("   export TELEGRAM_TOKEN='your_token'")
    print("   export TELEGRAM_CHAT_ID='your_chat_id'")
    sys.exit(1)


# ══════════════════════════════════════════════
# VERSION
# ══════════════════════════════════════════════
APP_VERSION = "v3.3"


# ══════════════════════════════════════════════
# TUNABLE PARAMETERS
# ══════════════════════════════════════════════
MIN_VOLUME_USDT = 100_000_000
MIN_LISTING_DAYS = 120
MIN_MA_DISTANCE = 25
WEEKLY_CANDLE_WARN = 80
SCAN_INTERVAL = 120

# Gate thresholds
GATE_PUMP_24H = 20.0
GATE_PUMP_4H = 15.0
GATE_PUMP_1H = 10.0

# Tier thresholds
TIER_A_MIN = 72
TIER_B_MIN = 52
TIER_C_MIN = 35

# 1h context constants
NEAR_HIGH_PCT = 0.98
UPPER_WICK_MIN = 0.04
MOMENTUM_SLOW_PCT = 2.0
MOMENTUM_FAST_PCT = 8.0

# OI minimum meaningful size
MIN_MEANINGFUL_OI = 5_000_000

# Dedup
DEDUP_WINDOW_MIN = 90


# ══════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# SAFE NUM
# ══════════════════════════════════════════════
def safe_num(x, default=None):
    try:
        v = float(x)
        if pd.isna(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


# ══════════════════════════════════════════════
# HTTP SESSION
# ══════════════════════════════════════════════
def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_session()


# ══════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════
def send_telegram(message: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }
        resp = SESSION.post(url, data=data, timeout=12)
        if resp.status_code == 200 and resp.json().get("ok"):
            return True
        logger.warning(f"Telegram rejected: {resp.status_code} — {resp.text[:120]}")
        return False
    except requests.exceptions.Timeout:
        logger.error("Telegram timeout")
    except Exception as e:
        logger.error(f"Telegram error: {e}")
    return False


# ══════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════
def _get_json(url: str, timeout: int = 12):
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            logger.warning("Rate limit — sleep 10s")
            time.sleep(10)
        raise


def get_bybit_symbols() -> dict:
    try:
        data = _get_json(
            "https://api.bybit.com/v5/market/instruments-info"
            "?category=linear&limit=1000"
        )
        out = {}
        if data.get("retCode") == 0:
            for item in data["result"]["list"]:
                if item.get("quoteCoin") == "USDT" and item.get("status") == "Trading":
                    out[item["symbol"]] = int(item.get("launchTime", 0))
        logger.info(f"Bybit: {len(out)} symbols")
        return out
    except Exception as e:
        logger.error(f"get_bybit_symbols: {e}")
        return {}


def get_aster_symbols() -> set:
    try:
        data = _get_json("https://fapi.asterdex.com/fapi/v1/exchangeInfo")
        out = {
            item["symbol"]
            for item in data.get("symbols", [])
            if item.get("quoteAsset") == "USDT" and item.get("status") == "TRADING"
        }
        logger.info(f"Aster: {len(out)} symbols")
        return out
    except Exception as e:
        logger.error(f"get_aster_symbols: {e}")
        return set()


def get_bybit_tickers_all() -> dict:
    try:
        data = _get_json("https://api.bybit.com/v5/market/tickers?category=linear")
        out = {}
        if data.get("retCode") == 0:
            for t in data["result"]["list"]:
                out[t["symbol"]] = {
                    "symbol": t["symbol"],
                    "price": float(t["lastPrice"]),
                    "volume_24h": float(t["turnover24h"]),
                    "price_24h_ago": float(t["prevPrice24h"]),
                    "open_interest": float(t.get("openInterest", 0)),
                    "funding_rate": float(t.get("fundingRate", 0)),
                }
        return out
    except Exception as e:
        logger.error(f"get_bybit_tickers_all: {e}")
        return {}


def get_bybit_klines(symbol: str, interval: str, limit: int = 60):
    try:
        data = _get_json(
            f"https://api.bybit.com/v5/market/kline"
            f"?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
        )
        if data.get("retCode") == 0:
            df = pd.DataFrame(
                data["result"]["list"],
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
            )
            df = df.astype(
                {
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "volume": float,
                }
            )
            return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        logger.debug(f"klines {symbol}/{interval}: {e}")
    return None


def get_long_short_ratio(symbol: str):
    try:
        data = _get_json(
            f"https://api.bybit.com/v5/market/account-ratio"
            f"?category=linear&symbol={symbol}&period=1h&limit=1"
        )
        if data.get("retCode") == 0 and data["result"]["list"]:
            item = data["result"]["list"][0]
            buy = float(item.get("buyRatio", 0.5))
            sell = float(item.get("sellRatio", 0.5))
            return buy / max(sell, 0.001)
    except Exception as e:
        logger.debug(f"ls_ratio {symbol}: {e}")
    return None


def get_oi_change_pct(symbol: str):
    try:
        data = _get_json(
            f"https://api.bybit.com/v5/market/open-interest"
            f"?category=linear&symbol={symbol}&intervalTime=1h&limit=25"
        )
        if data.get("retCode") == 0:
            items = data["result"]["list"]
            if len(items) >= 24:
                new_oi = float(items[0]["openInterest"])
                old_oi = float(items[-1]["openInterest"])
                if old_oi > 0:
                    return ((new_oi - old_oi) / old_oi) * 100
    except Exception as e:
        logger.debug(f"oi_change {symbol}: {e}")
    return None


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def calculate_moving_averages(df_daily):
    if df_daily is None or len(df_daily) < 30:
        return None

    cp = df_daily["close"].iloc[-1]

    def _dist(n):
        ma = df_daily["close"].iloc[-n:].mean()
        return round(ma, 8), round(((cp - ma) / ma) * 100, 1)

    ma7, d7 = _dist(7)
    ma14, d14 = _dist(14)
    ma30, d30 = _dist(30)

    return {
        "ma7": ma7,
        "ma14": ma14,
        "ma30": ma30,
        "dist7": d7,
        "dist14": d14,
        "dist30": d30,
    }


def check_weekly_history(symbol: str):
    df = get_bybit_klines(symbol, "W", limit=50)
    if df is None or len(df) < 2:
        return False, 0.0

    max_pump = max(
        ((r["close"] - r["open"]) / r["open"]) * 100
        for _, r in df.iterrows()
        if r["open"] > 0
    )
    del df
    return max_pump > WEEKLY_CANDLE_WARN, max_pump


def get_pump_pct(df, n_candles: int):
    if df is None or len(df) < n_candles + 1:
        return None
    old = df["close"].iloc[-(n_candles + 1)]
    new = df["close"].iloc[-1]
    return ((new - old) / old) * 100 if old > 0 else None


# ══════════════════════════════════════════════
# PUMP FETCHER
# ══════════════════════════════════════════════
def fetch_pump_data(symbol: str, ticker: dict, df_1h) -> dict:
    pump = {}

    p24 = ticker["price_24h_ago"]
    pump["24h"] = ((ticker["price"] - p24) / p24) * 100 if p24 > 0 else 0.0

    pump["1h"] = get_pump_pct(df_1h, 1)

    df4h = get_bybit_klines(symbol, "240", limit=2)
    pump["4h"] = get_pump_pct(df4h, 1)
    if df4h is not None:
        del df4h

    df7d = get_bybit_klines(symbol, "D", limit=8)
    pump["7d"] = get_pump_pct(df7d, 7)
    if df7d is not None:
        del df7d

    return pump


# ══════════════════════════════════════════════
# GATE
# ══════════════════════════════════════════════
def passes_gate(pump_data: dict) -> bool:
    p24 = pump_data.get("24h", 0) or 0
    p4h = pump_data.get("4h", 0) or 0
    p1h = pump_data.get("1h", 0) or 0

    return (
        (p24 >= 20 and p4h >= 5)
        or (p4h >= 15 and p1h >= 5)
        or (p24 >= 30)
    )


# ══════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════
def score_A_pump(pump_data: dict, ma_data: dict) -> tuple:
    pts = 0.0
    d = {}

    p24 = pump_data.get("24h") or 0.0
    d["pump_24h"] = f"{p24:+.1f}%"
    if p24 >= 80:
        pts += 10.0
    elif p24 >= 50:
        pts += 7.0
    elif p24 >= 30:
        pts += 4.0
    elif p24 >= 20:
        pts += 2.0

    p7d = pump_data.get("7d") or 0.0
    d["pump_7d"] = f"{p7d:+.1f}%"
    if p7d >= 100:
        pts += 3.0
    elif p7d >= 60:
        pts += 2.0
    elif p7d >= 30:
        pts += 1.0

    d7, d14, d30 = ma_data["dist7"], ma_data["dist14"], ma_data["dist30"]

    def _ma_pts(dist, max_pts):
        if dist >= 120:
            return max_pts
        elif dist >= 80:
            return max_pts * 0.75
        elif dist >= 50:
            return max_pts * 0.50
        elif dist >= 30:
            return max_pts * 0.25
        return 0.0

    pts += _ma_pts(d7, 3.3)
    pts += _ma_pts(d14, 3.3)
    pts += _ma_pts(d30, 3.4)

    p4h = pump_data.get("4h") or 0.0
    d["pump_4h"] = f"{p4h:+.1f}% (val)"
    if p4h >= 10:
        pts *= 1.10
    elif p4h >= 5:
        pts *= 1.05

    d["ma_dists"] = f"7d:{d7:+.0f}% 14d:{d14:+.0f}% 30d:{d30:+.0f}%"
    return min(pts, 25.0), d


def score_B_exhaustion(df_1h) -> tuple:
    pts = 0.0
    d = {}

    if df_1h is None or len(df_1h) < 26:
        return 0.0, {"note": "داده کم"}

    close = df_1h["close"]
    cur = safe_num(close.iloc[-1], 0.0)

    rsi = safe_num(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
    if rsi is not None:
        d["RSI"] = round(rsi, 1)
        if rsi >= 88:
            pts += 7.0
        elif rsi >= 80:
            pts += 5.0
        elif rsi >= 75:
            pts += 3.0
        elif rsi >= 70:
            pts += 1.5
    else:
        d["RSI"] = "N/A"

    stoch_k = safe_num(ta.momentum.StochRSIIndicator(close).stochrsi_k().iloc[-1])
    if stoch_k is not None:
        d["StochRSI"] = round(stoch_k * 100, 1)
        if stoch_k >= 0.97:
            pts += 5.0
        elif stoch_k >= 0.92:
            pts += 3.5
        elif stoch_k >= 0.85:
            pts += 2.0
    else:
        d["StochRSI"] = "N/A"

    bb = ta.volatility.BollingerBands(close, window=20)
    bb_high = safe_num(bb.bollinger_hband().iloc[-1])
    if bb_high is not None and bb_high > 0 and cur:
        bb_pct = ((cur - bb_high) / bb_high) * 100
        d["BB_above"] = f"{bb_pct:+.1f}%"
        if bb_pct >= 8:
            pts += 5.0
        elif bb_pct >= 4:
            pts += 3.0
        elif bb_pct >= 0:
            pts += 1.5
    else:
        d["BB_above"] = "N/A"

    ema21 = safe_num(ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1])
    if ema21 is not None and ema21 > 0 and cur:
        ema_dist = ((cur - ema21) / ema21) * 100
        d["EMA21_dist"] = f"{ema_dist:+.1f}%"
        if ema_dist >= 60:
            pts += 4.0
        elif ema_dist >= 35:
            pts += 2.5
        elif ema_dist >= 20:
            pts += 1.0
    else:
        d["EMA21_dist"] = "N/A"

    if len(df_1h) >= 25:
        avg_vol = safe_num(df_1h["volume"].iloc[-25:-1].mean(), 0.0)
        cur_vol = safe_num(df_1h["volume"].iloc[-1], 0.0)
        if avg_vol and avg_vol > 0:
            vol_ratio = cur_vol / avg_vol
            d["Vol_ratio"] = f"{vol_ratio:.1f}x"
            if vol_ratio >= 8:
                pts += 4.0
            elif vol_ratio >= 5:
                pts += 2.5
            elif vol_ratio >= 3:
                pts += 1.0
        else:
            d["Vol_ratio"] = "N/A"

    try:
        hist = ta.trend.MACD(close).macd_diff()
        hist_cur = safe_num(hist.iloc[-1])
        hist_prev = safe_num(hist.iloc[-2]) if len(hist) >= 2 else hist_cur
        if hist_cur is not None:
            if hist_cur < 0:
                pts += 2.0
                d["MACD"] = "منفی"
            elif hist_prev is not None and hist_cur < hist_prev * 0.5:
                pts += 1.0
                d["MACD"] = "در حال ضعیف شدن"
            else:
                d["MACD"] = "صعودی"
        else:
            d["MACD"] = "N/A"
    except Exception:
        d["MACD"] = "N/A"

    return min(pts, 25.0), d


def score_C_crowd(ticker: dict, ls_ratio, oi_change) -> tuple:
    pts = 0.0
    d = {}

    funding = ticker.get("funding_rate", 0) * 100
    d["Funding"] = f"{funding:.4f}%"
    if funding >= 0.30:
        pts += 8.0
    elif funding >= 0.15:
        pts += 5.5
    elif funding >= 0.08:
        pts += 3.0
    elif funding >= 0.03:
        pts += 1.0

    if ls_ratio is not None:
        d["LS_ratio"] = round(ls_ratio, 2)
        if ls_ratio >= 2.5:
            pts += 7.0
        elif ls_ratio >= 2.0:
            pts += 5.0
        elif ls_ratio >= 1.5:
            pts += 3.0
        elif ls_ratio >= 1.2:
            pts += 1.0
    else:
        d["LS_ratio"] = "N/A"

    if oi_change is not None:
        d["OI_change"] = f"{oi_change:+.1f}%"
        p24 = ((ticker["price"] - ticker["price_24h_ago"]) / ticker["price_24h_ago"]) * 100
        oi_abs = ticker.get("open_interest", 0)

        if oi_abs > MIN_MEANINGFUL_OI:
            if oi_change >= 30 and p24 >= 20:
                pts += 5.0
            elif oi_change >= 30:
                pts += 2.5
            elif oi_change >= 15 and p24 >= 10:
                pts += 3.0
            elif oi_change >= 15:
                pts += 1.5
            elif oi_change >= 5:
                pts += 0.5
    else:
        d["OI_change"] = "N/A"

    return min(pts, 20.0), d


def score_D_context(df_1h) -> tuple:
    pts = 0.0
    d = {}

    if df_1h is None or len(df_1h) < 20:
        return 0.0, {"note": "داده کم"}

    close = df_1h["close"]
    high = df_1h["high"]

    c_cur = safe_num(close.iloc[-1])
    c_prev = safe_num(close.iloc[-5])
    slow_momentum = False

    if c_cur is not None and c_prev and c_prev != 0:
        momentum = ((c_cur - c_prev) / c_prev) * 100
        d["momentum_4c"] = f"{momentum:+.1f}%"
        if momentum <= MOMENTUM_SLOW_PCT:
            pts += 5.0
            slow_momentum = True
        elif momentum <= 5.0:
            pts += 2.5
    else:
        d["momentum_4c"] = "N/A"

    last5 = df_1h.iloc[-5:]
    body_top = last5[["open", "close"]].max(axis=1)
    uw_raw = (last5["high"] - body_top) / last5["high"].replace(0, np.nan)
    uw_ratio = safe_num(uw_raw.mean(), 0.0)
    d["upper_wick_avg"] = f"{uw_ratio*100:.1f}%" if uw_ratio is not None else "N/A"

    strong_wick = False
    if uw_ratio is not None:
        if uw_ratio >= UPPER_WICK_MIN:
            pts += 4.0
            strong_wick = True
        elif uw_ratio >= UPPER_WICK_MIN / 2:
            pts += 2.0

    last2 = df_1h.iloc[-2:]
    candle_range = last2["high"] - last2["low"]
    uw2 = last2["high"] - last2[["open", "close"]].max(axis=1)
    uw2_ratio = safe_num((uw2 / candle_range.replace(0, np.nan)).mean(), 0.0)
    d["last2_wick_pct"] = f"{uw2_ratio*100:.1f}%" if uw2_ratio is not None else "N/A"

    if uw2_ratio is not None:
        if uw2_ratio >= 0.30:
            pts += 5.0
        elif uw2_ratio >= 0.15:
            pts += 2.5

    high_20 = safe_num(high.iloc[-20:].max())
    c_last = safe_num(close.iloc[-1])

    near_hi = (
        high_20 is not None
        and c_last is not None
        and c_last >= high_20 * NEAR_HIGH_PCT
    )
    d["near_20h_high"] = "بله ✓" if near_hi else "خیر"

    if near_hi and (strong_wick or slow_momentum):
        pts += 3.0
        d["near_high_confirmed"] = "تایید شد"
    elif near_hi:
        d["near_high_confirmed"] = "بدون تایید — امتیاز نگرفت"

    if len(df_1h) >= 7:
        vol_cur = safe_num(df_1h["volume"].iloc[-1])
        vol_prev = safe_num(df_1h["volume"].iloc[-4:-1].mean())
        if vol_cur is not None and vol_prev and vol_prev > 0:
            vol_trend = (vol_cur - vol_prev) / vol_prev * 100
            d["vol_trend"] = f"{vol_trend:+.1f}%"
            if vol_trend <= -20:
                pts += 3.0
            elif vol_trend <= -10:
                pts += 1.5
        else:
            d["vol_trend"] = "N/A"

    return min(pts, 20.0), d


def score_E_penalty(
    df_1h,
    ticker: dict,
    oi_change,
    ls_ratio,
    has_warn: bool = False,
) -> tuple:
    penalty = 0.0
    d = {}

    if df_1h is not None and len(df_1h) >= 10:
        close = df_1h["close"]

        c_last = safe_num(close.iloc[-1])
        c_prev = safe_num(close.iloc[-5])

        if c_last is not None and c_prev not in (None, 0):
            momentum = ((c_last - c_prev) / c_prev) * 100

            # HARD INVALIDATION
            if momentum >= MOMENTUM_FAST_PCT:
                return -10.0, {"invalidate": f"momentum {momentum:+.1f}%"}

            if momentum >= 5.0:
                penalty += 2.0
                d["pen_momentum"] = f"−2 (mom {momentum:+.1f}%)"

        last3_high = safe_num(df_1h["high"].iloc[-3:].max())
        high_20 = safe_num(df_1h["high"].iloc[-20:].max())
        if last3_high is not None and high_20 is not None and last3_high >= high_20 * 0.995:
            last3 = df_1h.iloc[-3:]
            body_top3 = last3[["open", "close"]].max(axis=1)
            uw3 = safe_num(
                ((last3["high"] - body_top3) / last3["high"].replace(0, np.nan)).mean()
            )
            if uw3 is not None and uw3 < 0.03:
                penalty += 3.0
                d["pen_newhigh"] = "−3 (high جدید بدون ریجکشن)"

        if len(df_1h) >= 5:
            vol_last2 = safe_num(df_1h["volume"].iloc[-2:].mean())
            vol_prev3 = safe_num(df_1h["volume"].iloc[-5:-2].mean())
            if (
                vol_last2 is not None
                and vol_prev3 is not None
                and vol_prev3 > 0
                and (vol_last2 / vol_prev3) > 1.3
            ):
                penalty += 2.0
                d["pen_vol_rising"] = "−2 (volume بالاست)"

    funding = ticker.get("funding_rate", 0) * 100
    if funding < 0:
        penalty += 1.5
        d["pen_funding"] = f"−1.5 (funding {funding:.4f}%)"
    elif funding == 0:
        penalty += 0.5
        d["pen_funding"] = "−0.5 (funding صفر)"

    if oi_change is not None and oi_change < -5:
        penalty += 1.0
        d["pen_oi"] = f"−1 (OI {oi_change:+.1f}%)"

    weak_ls = ls_ratio is not None and ls_ratio < 1.1
    weak_funding = funding < 0.02
    if weak_ls and weak_funding:
        penalty += 1.0
        d["pen_weak_crowd"] = "−1 (crowding ضعیف)"

    if has_warn:
        penalty += 2.0
        d["pen_weekly"] = "−2 (weekly continuation risk)"

    total = min(penalty, 10.0)
    return -total, d


# ══════════════════════════════════════════════
# TIER CLASSIFIER
# ══════════════════════════════════════════════
TIER_LABELS = {
    "A": "🔴 TIER A WATCH — ریسک برگشت بالا | تایید Phase 2 لازم است",
    "B": "🟠 TIER B WATCH — زیر نظر بگیر",
    "C": "🟡 TIER C WATCH — در رادار",
}


def classify_tier(score: float) -> tuple:
    if score >= TIER_A_MIN:
        return "A", TIER_LABELS["A"]
    elif score >= TIER_B_MIN:
        return "B", TIER_LABELS["B"]
    elif score >= TIER_C_MIN:
        return "C", TIER_LABELS["C"]
    return "N", None


# ══════════════════════════════════════════════
# DEDUP
# ══════════════════════════════════════════════
_dedup_cache: dict = {}


def should_send(symbol: str, tier: str) -> bool:
    now = time.time()
    window = DEDUP_WINDOW_MIN * 60
    entry = _dedup_cache.get(symbol)

    if entry is None:
        return True

    tier_rank = {"C": 1, "B": 2, "A": 3}
    if tier_rank.get(tier, 0) > tier_rank.get(entry["tier"], 0):
        return True

    return (now - entry["sent_at"]) >= window


def mark_sent(symbol: str, tier: str):
    _dedup_cache[symbol] = {"tier": tier, "sent_at": time.time()}


def clean_dedup():
    now = time.time()
    window = DEDUP_WINDOW_MIN * 60 * 2
    stale = [k for k, v in _dedup_cache.items() if now - v["sent_at"] > window]
    for k in stale:
        del _dedup_cache[k]


# ══════════════════════════════════════════════
# FORMAT MESSAGE
# ══════════════════════════════════════════════
def format_message(
    symbol,
    tier,
    tier_label,
    total_score,
    pump_data,
    age_days,
    ticker,
    ma_data,
    sa,
    sb,
    sc,
    sd,
    se,
    da,
    db,
    dc,
    dd,
    de,
    has_warn,
    max_weekly,
) -> str:
    score_clamped = max(0, min(100, round(total_score)))
    bar = "█" * round(score_clamped / 10) + "░" * (10 - round(score_clamped / 10))
    emoji = {"A": "🔴", "B": "🟠", "C": "🟡"}.get(tier, "⚪")

    pump_line = "  ".join(
        [
            f"1h:{pump_data.get('1h', 0):+.1f}%" if pump_data.get("1h") is not None else "1h:N/A",
            f"4h:{pump_data.get('4h', 0):+.1f}%" if pump_data.get("4h") is not None else "4h:N/A",
            f"24h:{pump_data.get('24h', 0):+.1f}%" if pump_data.get("24h") is not None else "24h:N/A",
            f"7d:{pump_data.get('7d', 0):+.1f}%" if pump_data.get("7d") is not None else "7d:N/A",
        ]
    )

    pen_notes = "  ".join(v for k, v in de.items() if k.startswith("pen_"))
    if not pen_notes:
        pen_notes = "بدون جریمه ✓"

    msg = f"{emoji} <b>{tier_label}</b>\n"
    msg += f"<b>{symbol}</b>  |  امتیاز: <b>{score_clamped}/100</b>\n"
    msg += f"[{bar}]\n\n"

    msg += f"💰 قیمت:    <b>${ticker['price']}</b>\n"
    msg += f"📈 پامپ:    {pump_line}\n"
    msg += f"📊 حجم:    ${ticker['volume_24h']/1e6:.0f}M  |  عمر: {age_days:.0f}d\n"
    msg += f"📉 Funding: {ticker['funding_rate']*100:.4f}%\n\n"

    msg += (
        f"📐 <b>MA:</b>  "
        f"7d:{ma_data['dist7']:+.0f}%  "
        f"14d:{ma_data['dist14']:+.0f}%  "
        f"30d:{ma_data['dist30']:+.0f}%\n\n"
    )

    msg += "🧮 <b>امتیاز تفکیک:</b>\n"
    msg += f"  A) Pump/Overext  {sa:5.1f}/25\n"
    msg += f"     {da.get('pump_24h','')} | {da.get('pump_4h','')} | {da.get('pump_7d','')} | MA: {da.get('ma_dists','')}\n"
    msg += f"  B) Exhaustion    {sb:5.1f}/25\n"
    msg += f"     RSI:{db.get('RSI','?')} | BB:{db.get('BB_above','?')} | StochRSI:{db.get('StochRSI','?')} | MACD:{db.get('MACD','?')}\n"
    msg += f"  C) Crowd         {sc:5.1f}/20\n"
    msg += f"     L/S:{dc.get('LS_ratio','?')} | OI:{dc.get('OI_change','?')} | Fund:{dc.get('Funding','?')}\n"
    msg += f"  D) 1h Context    {sd:5.1f}/20\n"
    msg += f"     mom:{dd.get('momentum_4c','?')} | wick:{dd.get('upper_wick_avg','?')} | near_hi:{dd.get('near_20h_high','?')}\n"
    msg += f"  E) Penalty      {se:5.1f}\n"
    msg += f"     {pen_notes}\n\n"

    msg += "📊 <b>Indicators:</b>\n"
    msg += (
        f"  RSI:{db.get('RSI','?')} | "
        f"Stoch:{db.get('StochRSI','?')} | "
        f"EMA21:{db.get('EMA21_dist','?')}\n"
    )
    msg += f"  Vol:{db.get('Vol_ratio','?')} | wick2:{dd.get('last2_wick_pct','?')}"
    if dd.get("near_high_confirmed"):
        msg += f" | {dd['near_high_confirmed']}"
    msg += "\n\n"

    if has_warn:
        msg += f"⚠️ <b>هشدار:</b> کندل هفتگی max {max_weekly:.0f}% — ریسک continuation!\n\n"

    msg += "⏳ <i>Phase 1 Watch — بدون Entry/SL — تایید Phase 2 لازم است</i>\n"
    msg += f"🕔 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    return msg


# ══════════════════════════════════════════════
# ANALYZE SYMBOL
# ══════════════════════════════════════════════
def analyze_symbol(symbol: str, launch_time: int, aster_symbols: set, ticker: dict):
    df_1h = None
    try:
        if symbol not in aster_symbols:
            return

        now_ms = int(time.time() * 1000)
        age_days = (now_ms - launch_time) / (1000 * 86400)
        if age_days < MIN_LISTING_DAYS:
            return

        if ticker["volume_24h"] < MIN_VOLUME_USDT:
            return

        if ticker["price_24h_ago"] <= 0:
            return

        df_daily = get_bybit_klines(symbol, "D", limit=32)
        ma_data = calculate_moving_averages(df_daily)
        if df_daily is not None:
            del df_daily

        if ma_data is None:
            return

        if max(ma_data["dist7"], ma_data["dist14"], ma_data["dist30"]) < MIN_MA_DISTANCE:
            return

        df_1h = get_bybit_klines(symbol, "60", limit=60)
        pump_data = fetch_pump_data(symbol, ticker, df_1h)

        if not passes_gate(pump_data):
            return

        has_warn, max_weekly = check_weekly_history(symbol)
        ls_ratio = get_long_short_ratio(symbol)
        oi_change = get_oi_change_pct(symbol)

        sa, da = score_A_pump(pump_data, ma_data)
        sb, db = score_B_exhaustion(df_1h)
        sc, dc = score_C_crowd(ticker, ls_ratio, oi_change)
        sd, dd = score_D_context(df_1h)
        se, de = score_E_penalty(df_1h, ticker, oi_change, ls_ratio, has_warn=has_warn)

        if "invalidate" in de:
            logger.info(f"INVALID {symbol} — {de['invalidate']}")
            return

        gc.collect()

        raw_score = sa + sb + sc + sd + se
        total_score = round(max(0.0, raw_score), 1)

        tier, tier_label = classify_tier(total_score)
        if tier == "N":
            logger.info(f"NO TRADE  {symbol:>12}  {total_score:.1f}")
            return

        if not should_send(symbol, tier):
            return

        msg = format_message(
            symbol,
            tier,
            tier_label,
            total_score,
            pump_data,
            age_days,
            ticker,
            ma_data,
            sa,
            sb,
            sc,
            sd,
            se,
            da,
            db,
            dc,
            dd,
            de,
            has_warn,
            max_weekly,
        )

        ok = send_telegram(msg)

        # فقط بعد از ارسال موفق
        if ok:
            mark_sent(symbol, tier)

        status = "✅" if ok else "⚠️ send failed — dedup NOT marked"
        logger.info(
            f"{status} {tier}  {symbol:>12}  {total_score:.1f}/100"
            f"  A={sa:.0f} B={sb:.0f} C={sc:.0f} D={sd:.0f} E={se:.0f}"
        )

    except Exception as e:
        logger.error(f"analyze {symbol}: {e}", exc_info=False)

    finally:
        if df_1h is not None:
            del df_1h


# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
def main():
    logger.info(f"Phase 1 Watchlist Engine {APP_VERSION} شروع شد")
    send_telegram(
        f"🚀 <b>Watchlist Engine {APP_VERSION} — Phase 1</b>\n"
        "✅ Token از env variable\n"
        "✅ Multi-TF aligned gate\n"
        "✅ 4h validation as multiplier\n"
        "✅ Hard invalidation logic\n"
        "✅ Weekly risk penalty\n"
        "✅ Smart dedup\n"
        "⏳ بدون Entry/SL — این فقط Watchlist است\n"
        "در حال اسکن بازار..."
    )

    while True:
        try:
            logger.info("══ اسکن شروع شد ══")
            bybit_symbols = get_bybit_symbols()
            aster_symbols = get_aster_symbols()
            all_tickers = get_bybit_tickers_all()

            for symbol, launch_time in bybit_symbols.items():
                ticker = all_tickers.get(symbol)
                if ticker:
                    analyze_symbol(symbol, launch_time, aster_symbols, ticker)
                time.sleep(0.4)

            del all_tickers
            gc.collect()
            clean_dedup()
            logger.info(f"══ اسکن تموم شد — {SCAN_INTERVAL}s صبر ══")
            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            logger.info("متوقف شد.")
            break
        except Exception as e:
            logger.error(f"main loop: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
