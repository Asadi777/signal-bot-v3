# Phase 0A — Research Bootstrap Plan v1 (Plan Only, Pre-Implementation)

> **Date:** 2026-08-30
> **Author:** Claude (implementation planning seat)
> **Owner / Approver:** Majid
> **Responds to:** `codex_phase0a_kickoff_task_v1.md` → «Required first response — plan only»
> **Authoritative context:** `crypto_pre_pump_master_spec_handoff_v5_seat1.md` (Master Spec v5 + Addendum A–D)
> **Controlling execution doc:** `crypto_pre_pump_feasibility_charter_v1.md`
> **Status:** `AWAITING APPROVAL` — هیچ کدی نوشته نشده و نباید نوشته شود تا تأیید Majid.

---

## 0. خلاصه در یک نگاه

Phase 0A فقط یک چیز را ثابت می‌کند: **آیا می‌توانیم dataset پژوهشیِ بازتولیدپذیر و Point-in-Time-Correct بسازیم؟**
این فاز مدل نمی‌سازد، Label نمی‌زند، Onset تشخیص نمی‌دهد و محصول نمی‌سازد.

خروجی نهایی: یک اسکلت Python تمیز + ۲ collector + یک dataset نمونه‌ی کوچک با schema دوزمانه + گزارش کیفیت داده + Data Source Matrix نسبتاً verify‌شده + فهرست صریح gapها.

سه انحراف پیشنهادی نسبت به متن Kickoff، که در بخش‌های مربوطه توضیح داده شده‌اند و **نیاز به تأیید دارند**:

1. Backfill اصلی از **آرشیو عمومی فایل‌محور Binance** انجام شود، نه REST pagination. دلیل: هم rate-limit را حذف می‌کند، هم تنها مسیر عملی برای پوشش **نمادهای delist‌شده** و رعایت قانون survivorship-bias (§65.3) است.
2. Smoke Universe عمداً شامل **حداقل ۳ نماد delist/غیرفعال‌شده** باشد، وگرنه مسیر survivorship اصلاً تست نمی‌شود.
3. سه تصمیم P0 که در Master Spec وجود ندارند و **قبل از Stage A** (نه قبل از Phase 0A) باید قفل شوند: Prediction Sampling Policy، Feature Embargo Rule، Alert Budget. در بخش ۴ ثبت شده‌اند.

---

## 1. فهم من از Task

### 1.1 آنچه باید ساخته شود

مطابق بخش‌های A تا G از Kickoff:

| کد | خروجی | ماهیت |
|---|---|---|
| A | اسکلت repo و reproducibility | پروژه‌ی Python تایپ‌دار، config، logging، tests، CLI، خروجی‌های نسخه‌دار |
| B | پشتیبانی Data Source Inventory | schema ماشین‌خوان Matrix + validator + نمونه‌ی فقط-verify‌شده |
| C | Exchange metadata collector | symbol/market metadata برای بازسازی آتی universe membership |
| D | OHLCV backfill PoC | فقط برای Smoke Universe کوچک، resumable/idempotent |
| E | فیلدهای Bitemporal/PIT | `event_time` / `available_at` / `ingested_at` + provenance |
| F | Data-quality report | gapها، duplicateها، نقض OHLC، ناهماهنگی cross-source |
| G | Deliverables | tree، کد، تست، دستورها، sample، schema docs، Matrix، گزارش، محدودیت‌ها |

### 1.2 آنچه صریحاً ساخته نمی‌شود

Frontend، multi-user/RBAC، live trading، Learning Engine، Specialist Agents، n8n production، خرید داده‌ی گسترده، مدل Stage A، و **هیچ dataset مبتنی بر «Top 200 امروز» به‌عنوان universe تاریخی**.

### 1.3 مرزهای تصمیم‌گیری که در این Task باز می‌مانند

DB نهایی، Feature Store، message bus، معماری مدل و cloud platform انتخاب **نمی‌شوند**. فقط trade-off ثبت می‌شود.

### 1.4 معیار موفقیت این فاز (بازنویسی عملیاتی)

Phase 0A موفق است اگر یک نفر روی یک ماشین تازه، با دستورهای مستند، بتواند dataset نمونه را از صفر بسازد و **دو بار اجرا، خروجی byte-identical** بدهد (به‌جز `ingested_at` و `run_id`)، و هر عدد در گزارش کیفیت به یک فایل خام با checksum قابل ردیابی باشد.

---

## 2. ساختار پیشنهادی Repository

### 2.1 تصمیم پایه: هم‌زیستی با بات فعلی

مخزن فعلی `signal-bot-v3` شامل `signal_bot.py` (v3.3، اسکورینگ heuristic روی Bybit) است. پیشنهاد:

- `signal_bot.py` **دست‌نخورده** بماند (نه refactor، نه انتقال).
- کار پژوهشی در پکیج مجزای `research/` با dependency و lifecycle مستقل.
- بات فعلی به‌عنوان **یکی از heuristic baselineهای الزامیِ §57.4** ثبت شود (رقیب Stage A)، نه کد dead.

اگر Majid ترجیح دهد repo کاملاً جدا باشد، ساختار زیر بدون تغییر منتقل می‌شود. `DECISION NEEDED`.

### 2.2 درخت پیشنهادی

```
signal-bot-v3/
├─ signal_bot.py                    # legacy v3.3 — frozen, untouched
├─ requirements.txt                 # legacy runtime deps
├─ docs/
│  ├─ phase0a_plan_v1.md            # این سند
│  ├─ decisions/                    # ADRها (کوتاه، شماره‌دار، append-only)
│  │  └─ 0001-archive-first-backfill.md
│  └─ schemas/
│     ├─ ohlcv_normalized_v1.md     # مستندات schema + معنای هر timestamp
│     └─ data_source_matrix_v1.md
└─ research/
   ├─ README.md                     # setup + دستورهای دقیق اجرا
   ├─ pyproject.toml                # deps پین‌شده + lock
   ├─ Makefile                      # entrypointهای بازتولیدپذیر
   ├─ .env.example                  # بدون هیچ secret
   ├─ config/
   │  ├─ smoke_universe_v1.yaml     # نسخه‌دار، با دلیل انتخاب هر نماد
   │  └─ sources.yaml               # پارامتر منابع + Δ availability هر منبع
   ├─ src/prepump/
   │  ├─ version.py                 # SCHEMA_VERSION, COLLECTOR_VERSION
   │  ├─ config.py                  # pydantic-settings (typed, fail-fast)
   │  ├─ logging.py                 # structured JSON logs + run_id در هر رکورد
   │  ├─ timeutils.py               # فقط UTC؛ ms-precision؛ ممنوعیت naive datetime
   │  ├─ manifest.py                # نوشتن/خواندن run manifest
   │  ├─ io/
   │  │  ├─ paths.py                # چیدمان raw/normalized/reports/manifests
   │  │  ├─ parquet.py              # نوشتن atomic + idempotent (tmp → rename)
   │  │  └─ checksums.py            # sha256 فایل و partition
   │  ├─ sources/
   │  │  ├─ base.py                 # Protocol مشترک: discover/fetch/normalize/Δ
   │  │  ├─ binance_vision.py       # آرشیو فایل‌محور (backfill اصلی)
   │  │  ├─ binance_rest.py         # klines REST (incremental + cross-check)
   │  │  ├─ bybit_rest.py           # kline v5 + instruments-info
   │  │  └─ ratelimit.py            # token bucket سراسری + backoff کران‌دار
   │  ├─ normalize/
   │  │  ├─ symbols.py              # symbol canonical id
   │  │  ├─ ohlcv.py                # نگاشت raw → normalized
   │  │  └─ schema.py               # pyarrow schema + validation
   │  ├─ quality/
   │  │  ├─ checks.py               # هر check یک تابع خالص و تست‌پذیر
   │  │  └─ report.py               # خروجی md + json
   │  ├─ matrix/
   │  │  ├─ model.py                # مدل تایپ‌دار Data Source Matrix
   │  │  ├─ validate.py             # قواعد evidence برای AVAILABLE
   │  │  └─ render.py               # yaml/json → جدول Markdown
   │  └─ cli/__main__.py            # typer: چند subcommand مستند
   ├─ tests/
   │  ├─ unit/                      # بدون شبکه، سریع
   │  ├─ integration/               # marker: @pytest.mark.integration (شبکه)
   │  └─ fixtures/                  # payloadهای ضبط‌شده‌ی کوچک
   └─ data/                         # کاملاً gitignored
      ├─ raw/                       # بایت‌های اصلی + sidecar metadata
      ├─ normalized/                # Parquet
      ├─ reports/
      └─ manifests/
```

### 2.3 چیدمان خروجی‌ها

**Raw** (تغییرناپذیر، هرگز بازنویسی نمی‌شود):
```
data/raw/{source}/{market_type}/{symbol}/{interval}/{YYYY-MM}/<original_file>
data/raw/.../<original_file>.meta.json     # url, etag, sha256, fetched_at, http headers
```

**Normalized** (Hive partitioning، واحد idempotency):
```
data/normalized/ohlcv_1m/exchange=<x>/market_type=<m>/symbol=<s>/date=<YYYY-MM-DD>/part-000.parquet
```

**Manifest هر run**:
```
data/manifests/{run_id}.json
```
شامل: `run_id`، زمان شروع/پایان، نسخه‌ی کد و schema، hash فایل config، فهرست partitionهای نوشته/رد‌شده، تعداد سطر، checksum هر partition، خطاها، و آمار rate-limit.

### 2.4 اصول طراحی که در کد الزام می‌شوند

- **هیچ gap-filling / imputation در Phase 0A.** gap ثبت می‌شود، پر نمی‌شود.
- **Raw هرگز mutate نمی‌شود**؛ normalized همیشه از raw قابل بازتولید است.
- **هیچ مسیر کدی برای order/trading وجود ندارد** — با یک تست ساختاری تضمین می‌شود.
- هر خروجی به `schema_version` + `collector_version` + `run_id` گره می‌خورد.

---

## 3. APIها و Datasetهایی که برای PoC رایگان استفاده می‌شوند

> قاعده‌ی سند: هیچ‌کدام از موارد زیر تا **قبل از verify واقعی** در Matrix `AVAILABLE` علامت نمی‌خورند. آنچه در ادامه می‌آید **نیت استفاده + فرضیه‌ی قابل‌آزمون** است، نه ادعای تأییدشده.

### 3.1 منبع اصلی backfill — آرشیو عمومی فایل‌محور Binance

`data.binance.vision` فایل‌های ماهانه/روزانه‌ی kline را برای spot و USDⓈ-M futures منتشر می‌کند، همراه فایل CHECKSUM.

چرا این به‌جای REST:

| معیار | آرشیو فایل | REST klines |
|---|---|---|
| سرعت backfill | یک فایل = یک ماه | ۱۰۰۰ کندل در هر request |
| Rate-limit | عملاً بی‌مسئله (CDN استاتیک) | weight-based، ban-prone |
| نمادهای delist‌شده | حفظ می‌شوند | معمولاً برنمی‌گردند |
| تعیّن (determinism) | فایل + checksum | وابسته به زمان درخواست |
| بازتولیدپذیری | بالا | متوسط |

**نکته‌ی حیاتی:** پوشش نمادهای delist‌شده در آرشیو، تنها مسیر عملی و رایگانی است که قانون §65.3 (منع Top-200 امروز) را قابل اجرا می‌کند.

مواردی که در M2 باید **verify** شوند و تا آن زمان `UNKNOWN` می‌مانند:
- تاریخ شروع پوشش برای هر بازار و هر نماد؛
- وجود/صحت CHECKSUM برای همه‌ی فایل‌ها؛
- ثبات layout ستون‌ها و وجود/نبود header row (drift مشاهده‌شده در برخی نسخه‌ها)؛
- روش listing نمادها (S3-style listing) برای کشف نمادهای غیرفعال؛
- شرایط استفاده/بازتوزیع.

### 3.2 Metadata صرافی

| صرافی | Endpoint | فیلدهای مورد انتظار | شکاف مورد انتظار |
|---|---|---|---|
| Binance spot | `GET /api/v3/exchangeInfo` | symbol، status، base/quote، filters | **تاریخ listing ندارد** → باید از «اولین کندل موجود» تقریب زده و صریحاً `APPROXIMATED` علامت بخورد |
| Binance USDⓈ-M | `GET /fapi/v1/exchangeInfo` | + `onboardDate`، `contractType`، `deliveryDate` | تاریخ delist معمولاً پس از حذف در دسترس نیست |
| Bybit | `GET /v5/market/instruments-info?category=spot\|linear` | symbol، status، `launchTime`، contract specs | عمق تاریخی و پایداری فیلدها باید تست شود |

Metadata به‌صورت **snapshot زمان‌دار** ذخیره می‌شود (`collected_at` + append-only)، چون همین snapshotهای متوالی در آینده تنها راه بازسازی تقریبی membership-as-of-time هستند. این را باید از **الان** شروع کنیم، حتی اگر امروز استفاده‌ای ندارد.

### 3.3 OHLCV زنده/incremental و cross-check

- Binance: `GET /api/v3/klines` و `GET /fapi/v1/klines` — برای پنجره‌ی اخیر و **مقایسه‌ی مستقل با آرشیو**.
- Bybit: `GET /v5/market/kline?category=...&interval=1` — عمق تاریخی ۱ دقیقه‌ای Bybit **نامعلوم و مشکوک به کوتاه بودن** است؛ اگر ناکافی بود، مسیر جایگزین آرشیو trade-level عمومی Bybit و ساخت کندل از trades است. این مسیر در Phase 0A فقط **بررسی و مستند** می‌شود، پیاده‌سازی کاملش خارج از scope است.

### 3.4 معناشناسی timestamp (تعریف دقیق)

برای kline با بازه‌ی ۱ دقیقه:

```
event_time        := open_time  (شروع کندل، UTC، ms)
event_time_close  := close_time (= open_time + 59_999ms)
available_at      := event_time_close + Δ_source
ingested_at       := زمان واقعی نوشتن رکورد توسط pipeline
```

`Δ_source` و روش آن صریحاً ثبت می‌شود:

| `available_at_method` | معنا | کاربرد |
|---|---|---|
| `DERIVED_FROM_CLOSE` | زودترین زمان قابل‌دفاع = بسته شدن کندل + Δ ثابت مستند | داده‌ی تاریخی آرشیو |
| `MEASURED` | اختلاف واقعی اندازه‌گیری‌شده بین close و لحظه‌ی دریافت | جمع‌آوری زنده |
| `APPROXIMATED` | تقریب مستند با عدم‌قطعیت | مواردی که منبع اجازه‌ی دقت بیشتر نمی‌دهد |

به‌همراه `available_at_uncertainty_ms`. مطابق Kickoff §E، **هرگز دقتی که منبع پشتیبانی نمی‌کند جعل نمی‌شود**.

### 3.5 آنچه در این فاز مصرف نمی‌شود

Derivatives (OI/Funding)، on-chain، social، news، order book. اینها فقط به‌عنوان ردیف‌های `UNKNOWN` با «برنامه‌ی بررسی» در Matrix ثبت می‌شوند. هیچ داده‌ای از آنها دانلود نمی‌شود.

---

## 4. فرض‌های حل‌نشده و Blockerها

### 4.1 Blockerهای Stage A که Phase 0A را متوقف نمی‌کنند ولی باید قبل از Stage A قفل شوند

این سه مورد در Master Spec v5 **وجود ندارند** و به‌نظر من مهم‌ترین شکاف‌های P0 هستند:

**(الف) Prediction Sampling Policy — `MISSING / P0`**
مشخص نیست پیش‌بینی روی چه واحدی صادر می‌شود (هر دقیقه × هر نماد؟ هر ۱۵ دقیقه؟ فقط پس از عبور از gate؟). بدون این، `Precision@K`، `False alerts/day` و AUPRC عدد بی‌معنا می‌دهند؛ با ۵۰۰ نماد × ۱۴۴۰ دقیقه، base rate چنان کوچک است که هر مدلی «عالی» به‌نظر می‌رسد. لازم است: cadence، قاعده‌ی dedup پنجره‌های هم‌پوشان، و صراحت اینکه ارزیابی **event-level** است نه row-level.

**(ب) Feature Embargo Rule — `MISSING / P0`**
قانون ۱۵ دقیقه فقط برای *صدور سیگنال* تعریف شده، نه برای *پنجره‌ی feature*. اگر featureها تا لحظه‌ی onset ادامه یابند، مدل عملاً «شروع‌شدن حرکت» را می‌بیند و lift تقلبی می‌شود. پیشنهاد: پنجره‌ی feature در `onset − 15min` قطع شود، و در walk-forward هم purge/embargo پیرامون هر event اعمال گردد. پیشنهاد می‌کنم به `ARCHITECTURAL RULE` ارتقا یابد.

**(ج) Alert Budget — `OPEN، ولی قابل تصمیم‌گیری همین حالا`**
§48.4 تعارض Recall-first با تعداد سیگنال را اذعان کرده ولی حل نکرده. بدون یک بودجه‌ی تقریبی هشدار (مثلاً ۱۰–۳۰ هشدار رسمی در روز)، مقدار K تعریف نمی‌شود و تصمیم Gate مبهم می‌ماند. پیشنهاد: KPI اصلی به `Recall @ fixed alert budget` تبدیل شود.

موارد تکمیلی که در نقد قبلی مطرح شد و همچنان باز است: مرجع دقیق قیمت در تعریف «+۵۰٪ در ۲۴ ساعت» (high یا close؟ mark یا last؟)، فیلتر حجم برای رد کردن wickهای غیرقابل‌معامله، dedup رویداد در سطح **توکن** به‌جای pair، و مدل fee/slippage برای paper trading.

### 4.2 فرض‌های فنی که Phase 0A باید verify کند

| # | فرض | ریسک اگر غلط باشد | برنامه‌ی تأیید |
|---|---|---|---|
| 1 | آرشیو Binance نمادهای delist‌شده را نگه می‌دارد | مسیر ضد-survivorship از بین می‌رود؛ باید به منابع پولی فکر کنیم | M2/M5 با یک نماد delist‌شده‌ی مشخص |
| 2 | layout ستون‌های آرشیو پایدار است | parse خاموش خراب می‌شود | parser سخت‌گیر + تست روی چند ماه از سال‌های مختلف |
| 3 | عمق ۱ دقیقه‌ای Bybit برای پژوهش کافی است | Bybit در Stage A حذف یا محدود می‌شود | M7 تست عمق |
| 4 | «دقیقه‌ی گمشده» عمدتاً یعنی «هیچ معامله‌ای نبوده» نه نقص جمع‌آوری | گزارش کیفیت پر از false alarm می‌شود | تفکیک `NO_TRADE` از `COLLECTION_GAP` با cross-source |
| 5 | استفاده‌ی پژوهشی داخلی از این داده مجاز است | ریسک حقوقی | خواندن ToS و ثبت در Matrix |
| 6 | دسترسی شبکه‌ای به این هاست‌ها از محیط اجرا ممکن است (بدون ۴۵۱) | backfill از آن محیط ناممکن | تست اتصال زودهنگام در M1 |

### 4.3 تصمیم‌هایی که به Majid نیاز دارند (قبل از M1)

1. `research/` داخل همین repo یا repo جداگانه؟ (پیشنهاد من: همین repo)
2. `taker_buy_volume` و `trade_count` — که رایگان و در همان payload کندل می‌آیند — جزو **Stage A** هستند یا **Stage C (microstructure)**؟ پیشنهاد من: همیشه در dataset ذخیره شوند، ولی با فلگ `stage_a_allowed=false` تا وقتی تصمیم رسمی گرفته شود.
3. آیا Bybit در Phase 0A الزامی است یا اگر عمق ۱ دقیقه‌ای کافی نبود، به Phase بعد موکول شود؟
4. آیا اجازه هست Master Spec به‌صورت نسخه‌دار در `docs/` همین repo نگه‌داری شود (به‌جای handoff دستی بین Seatها)؟

---

## 5. برنامه‌ی پیاده‌سازی — Milestoneهای کوچک

هر Milestone: مستقل، قابل review، قابل revert، و با acceptance مشخص. تخمین «اندازه» نسبی است، نه تعهد زمانی.

| M | عنوان | خروجی | Acceptance | اندازه |
|---|---|---|---|---|
| **M0** | تأیید همین Plan | امضای Majid + پاسخ به ۴ سؤال بخش ۴.۳ | تصمیم‌ها ثبت شده‌اند | — |
| **M1** | Scaffold و reproducibility | ساختار repo، config تایپ‌دار، logging ساخت‌یافته، CLI اسکلت، CI، `.env.example`، تست اتصال شبکه | `make setup && make test` روی محیط تازه سبز است | S |
| **M2** | Matrix schema + validator | مدل تایپ‌دار Matrix، validator قواعد evidence، renderer به Markdown، Matrix خالی با ردیف‌های `UNKNOWN` | تست: `AVAILABLE` بدون هر ۷ شاهد **رد** می‌شود | S |
| **M3** | Exchange metadata collector | snapshot زمان‌دار Binance spot/futures + Bybit spot/linear | snapshot معتبر تولید و schema-validate می‌شود؛ شکاف‌ها (نبود listing date در spot) ثبت شده | M |
| **M4** | تعریف Smoke Universe v1 | `smoke_universe_v1.yaml` با ۱۰–۲۰ نماد + دلیل هر انتخاب | شامل ≥۳ نماد delist/غیرفعال، ≥۲ market type، ≥۳ لایه‌ی نقدشوندگی؛ محدودیت‌ها صریحاً نوشته شده | S |
| **M5** | Backfill از آرشیو (هسته‌ی کار) | دانلود + تأیید checksum + نگه‌داری raw + normalize به Parquet؛ resumable، idempotent | اجرای دوباره → checksum یکسان؛ قطع در وسط → ادامه بدون duplicate | L |
| **M6** | REST incremental + cross-check | collector پنجره‌ی اخیر + گزارش اختلاف REST در برابر آرشیو | همپوشانی مقایسه و گزارش می‌شود؛ اختلاف‌ها کمّی‌اند | M |
| **M7** | Bybit collector | instruments-info + kline + گزارش عمق واقعی تاریخی | عمق واقعی اندازه‌گیری و در Matrix ثبت شد (حتی اگر نتیجه `LIMITED` باشد) | M |
| **M8** | Data-quality report | همه‌ی checkهای بخش F + تفکیک `NO_TRADE` از `COLLECTION_GAP` | گزارش md+json تولید می‌شود؛ هر عدد به partition و فایل خام قابل ردیابی است | M |
| **M9** | تحویل نهایی | README، مستندات schema، sample کوچک، Matrix نسبی، فهرست محدودیت‌ها، پیشنهاد Task بعدی | همه‌ی معیارهای Acceptance بخش ۸ سبز | S |

**نقاط توقف اجباری:** پس از M2 (اگر آرشیو نمادهای delist‌شده را نداشت) و پس از M7 (اگر Bybit عملاً غیرقابل‌استفاده بود) — هر دو مورد به تصمیم Majid برمی‌گردند، نه به ابتکار من.

### 5.1 Schema نرمال‌شده‌ی پیشنهادی (`ohlcv_1m`, `schema_version = 1`)

| ستون | نوع | توضیح |
|---|---|---|
| `exchange` | string | `binance` \| `bybit` |
| `market_type` | string | `spot` \| `linear_perpetual` \| ... |
| `symbol` | string | نماد بومی صرافی |
| `symbol_canonical` | string | شناسه‌ی یکنواخت پروژه |
| `base_asset`, `quote_asset` | string | |
| `interval` | string | `1m` |
| `event_time` | timestamp[ms, UTC] | شروع کندل — کلید زمانی |
| `event_time_close` | timestamp[ms, UTC] | پایان کندل |
| `available_at` | timestamp[ms, UTC] | زودترین زمان قابل‌دفاع استفاده |
| `available_at_method` | string(enum) | `DERIVED_FROM_CLOSE` \| `MEASURED` \| `APPROXIMATED` |
| `available_at_uncertainty_ms` | int32 | عدم‌قطعیت صریح |
| `ingested_at` | timestamp[ms, UTC] | زمان واقعی دریافت/نوشتن |
| `open`,`high`,`low`,`close` | float64 | مقدار خام رشته‌ای در لایه‌ی raw حفظ می‌شود |
| `base_volume` | float64 | |
| `quote_volume` | float64 (nullable) | |
| `trade_count` | int64 (nullable) | `stage_a_allowed` = تصمیم Majid |
| `taker_buy_base`, `taker_buy_quote` | float64 (nullable) | همان بالا |
| `source` | string | مثل `binance_vision_monthly_klines` |
| `source_revision` | string | نام فایل + sha256/etag یا hash پارامترهای endpoint |
| `collector_version`, `schema_version` | string | |
| `run_id` | string | شناسه‌ی اجرا |
| `dq_flags` | list<string> | پرچم‌های کیفیت در سطح سطر |

- **کلید یکتا:** `(exchange, market_type, symbol, interval, event_time)`
- **Partition:** `exchange / market_type / symbol / date`
- **از checksum تعیّن مستثنا:** `ingested_at`، `run_id` (به‌صورت طراحی‌شده)

---

## 6. حجم ذخیره‌سازی و ریسک‌های Rate-limit

### 6.1 برآورد حجم (مرتبه‌ی بزرگی، نه دقیق)

پایه: هر نماد در هر سال با کندل ۱ دقیقه‌ای ≈ **۵۲۵٬۶۰۰ سطر**. با Parquet + zstd، حدود **۲۵–۴۵ بایت بر سطر** برای این schema انتظار می‌رود (محافظه‌کارانه: ۴۰).

| سناریو | سطرها | Normalized | Raw | مجموع تقریبی |
|---|---:|---:|---:|---:|
| Smoke: ۲۰ نماد × ۳ سال | ~۳۲M | ~۱.۳ GB | ~۱.۵ GB | **~۳ GB** |
| میانی: ۱۰۰ نماد × ۳ سال | ~۱۵۸M | ~۶ GB | ~۷ GB | ~۱۳ GB |
| کامل (فقط برآورد، خارج از scope): ۶۰۰ نماد × ۵ سال | ~۱.۶B | ~۶۰–۸۰ GB | ~۸۰ GB | **~۱۵۰ GB** |

نتیجه‌گیری‌ای که باید **گزارش شود نه تصمیم‌گیری**: Parquet روی فایل‌سیستم برای Phase 0A و حتی Stage A کافی است؛ انتخاب DB نهایی همچنان `OPEN` می‌ماند (§ مرزهای تصمیم Kickoff).

**هشدار مقیاس:** اگر بعداً داده‌ی trade-level یا order-book لازم شود، حجم **۱ تا ۲ مرتبه‌ی بزرگی** بیشتر می‌شود (چند TB). این باید قبل از Stage C در برنامه‌ی هزینه دیده شود.

### 6.2 ریسک‌های rate-limit و کاهش آنها

| ریسک | اثر | کاهش |
|---|---|---|
| Ban IP در REST صرافی | توقف کل backfill، احتمال ban طولانی | استراتژی archive-first (حجم REST ناچیز می‌ماند)؛ token bucket سراسری؛ خواندن هدرهای مصرف weight؛ احترام به `Retry-After`؛ توقف کامل در ban به‌جای retry |
| فشار بیش‌ازحد روی CDN آرشیو | throttle | سقف concurrency (۴–۸)، backoff نمایی کران‌دار، cache محلی و skip فایل‌های موجود با checksum معتبر |
| محدودیت جغرافیایی (۴۵۱) از محیط اجرا | backfill از آن ماشین ناممکن | تست اتصال در M1؛ در صورت شکست، تصمیم زیرساخت به Majid برمی‌گردد |
| قطع وسط دانلود | فایل ناقص و داده‌ی خراب | نوشتن atomic (tmp → rename) + تأیید checksum قبل از normalize؛ فایل مشکوک هرگز وارد normalized نمی‌شود |
| تغییر خاموش داده در منبع (revision) | نقض تعیّن | ثبت `source_revision`؛ تشخیص و گزارش تغییر checksum به‌جای بازنویسی خاموش |

**سیاست backoff:** حداکثر N تلاش (پیش‌فرض ۵)، نمایی با jitter، سقف تأخیر مطلق، و **هرگز retry نامحدود**. هر backoff در manifest ثبت می‌شود تا فشار واقعی روی منبع قابل اندازه‌گیری باشد.

---

## 7. ملاحظات امنیتی و حقوقی

### 7.1 Secrets

- هیچ‌کدام از منابع Phase 0A **به API key نیاز ندارند** (همه public market data). این یک مزیت امنیتی است و باید حفظ شود.
- `.env.example` فقط کلیدهای اختیاری آینده را با مقدار خالی نشان می‌دهد.
- CI شامل secret scanning؛ pre-commit hook برای جلوگیری از commit تصادفی.
- **بررسی لازم:** تاریخچه‌ی git فعلی از نظر نشت `TELEGRAM_TOKEN` اسکن شود (کد فعلی درست از env می‌خواند، ولی تاریخچه باید تأیید شود).

### 7.2 مرز ایمنی معاملاتی

در پکیج پژوهشی **هیچ مسیر کدی به endpointهای order/withdraw وجود ندارد** و این با یک تست ساختاری تضمین می‌شود (نه فقط با قرارداد شفاهی). این همان الزام §45.2 در سطح Phase 0A است.

### 7.3 Licensing و بازتوزیع

- استفاده‌ی **داخلی و پژوهشی** از داده‌ی عمومی بازار معمولاً مجاز است، ولی **بازتوزیع** غالباً محدود است.
- بنابراین: **هیچ dataset واقعی در git commit نمی‌شود.** `data/` کاملاً gitignored است.
- «Sample dataset» تحویلی طبق بخش G، یا چند صد سطر بسیار کوچک به‌عنوان fixture تست است، یا نمونه‌ی **مصنوعی** به‌همراه اسکریپت بازتولید نمونه‌ی واقعی. تصمیم نهایی پس از خواندن ToS.
- ستون `licensing` و `redistribution` در Matrix تا زمان تأیید مستند، `UNKNOWN` می‌ماند — نه خوش‌بینانه پر می‌شود، نه خالی رها می‌شود.
- منابع آینده (social/scraping) ریسک حقوقی جدی‌تری دارند و باید قبل از هر جمع‌آوری بررسی شوند.

### 7.4 زنجیره‌ی تأمین وابستگی‌ها

وابستگی‌ها حداقلی و **پین‌شده با lock** خواهند بود. هیچ وابستگی‌ای که فقط برای «راحتی» است اضافه نمی‌شود؛ هر افزودن dependency باید دلیل نوشته‌شده داشته باشد.

---

## 8. فهرست Acceptance Testها

### 8.1 Unit (بدون شبکه، سریع، در هر CI run)

| # | تست | چه چیزی را ثابت می‌کند |
|---|---|---|
| 1 | نگاشت timestamp: `open_time/close_time → event_time/available_at` | معناشناسی زمان درست است |
| 2 | ممنوعیت datetime بدون timezone؛ همه‌چیز UTC با دقت ms | الزام Kickoff §E |
| 3 | `available_at_method` و `uncertainty` همیشه پر و سازگارند | منع جعل دقت |
| 4 | نرمال‌سازی نماد در سه حالت (spot/perp/Bybit) | یکتایی شناسه |
| 5 | Schema validation: رد null در کلید، رد نوع اشتباه، رد ستون ناشناخته | استحکام schema |
| 6 | Pagination روی منبع ساختگی: صفحه‌ی خالی، مضرب دقیق limit، صفحه‌ی ناقص آخر | صحت مرزها |
| 7 | Retry: ۴۲۹ سپس موفقیت؛ ban → توقف کامل؛ سقف تلاش رعایت می‌شود | backoff کران‌دار |
| 8 | **Idempotency**: دو بار اجرا روی همان ورودی → checksum یکسان، تعداد سطر ثابت | معیار Kickoff |
| 9 | **Resume**: استثنا در partition اِn → اجرای بعدی فقط باقی‌مانده را می‌سازد | معیار Kickoff |
| 10 | تشخیص duplicate روی سطرهای تزریق‌شده | معیار Kickoff |
| 11 | تشخیص missing interval روی gap مصنوعی | بخش F |
| 12 | تفکیک `NO_TRADE` از `COLLECTION_GAP` | جلوگیری از false alarm در small-capها |
| 13 | نقض OHLC (`low > open` و مشابه)، قیمت غیرمثبت، حجم منفی | بخش F |
| 14 | ترتیب زمانی: تشخیص سطرهای out-of-order | بخش F |
| 15 | Manifest شامل همه‌ی فیلدهای الزامی و hash فایل config | بازتولیدپذیری |
| 16 | Validator ماتریس: `AVAILABLE` بدون هر ۷ شاهد **رد** می‌شود | قاعده‌ی صریح Kickoff §B |
| 17 | Validator ماتریس: enum وضعیت، الزام reviewer و تاریخ بررسی | همان |
| 18 | فایل خراب/checksum ناموفق → **هیچ خروجی normalized تولید نمی‌شود** | جلوگیری از آلودگی داده |
| 19 | تست ساختاری: هیچ ماژولی endpoint معاملاتی import/صدا نمی‌زند | مرز ایمنی |
| 20 | هیچ مسیر کدی داده‌ی گمشده را impute نمی‌کند | قاعده‌ی no-gap-filling |

### 8.2 Integration (marker جدا، نیازمند شبکه، خارج از CI پیش‌فرض)

| # | تست |
|---|---|
| 21 | دانلود یک فایل ماهانه‌ی کوچک + تأیید checksum + normalize → تعداد سطر مورد انتظار |
| 22 | Cross-check: همان پنجره از REST و آرشیو → گزارش اختلاف تولید و کمّی می‌شود |
| 23 | Metadata هر دو صرافی جمع و schema-validate می‌شود |
| 24 | **مسیر نماد delist‌شده**: از آرشیو موفق؛ از REST خطا به‌شکل تمیز مدیریت می‌شود |
| 25 | parse صحیح هدرهای rate-limit در پاسخ واقعی |
| 26 | تست عمق تاریخی واقعی kline یک‌دقیقه‌ای Bybit (نتیجه هرچه باشد، ثبت می‌شود) |

### 8.3 End-to-End و بازتولیدپذیری

| # | تست |
|---|---|
| 27 | Clone تازه + دستورهای مستند README → backfill smoke کامل و گزارش تولید می‌شود |
| 28 | **تعیّن:** دو اجرا (زمان/ماشین متفاوت) → checksum یکسان به‌جز `ingested_at`/`run_id` |
| 29 | ردیابی: هر عدد گزارش کیفیت تا فایل خام با sha256 قابل دنبال کردن است |
| 30 | Secret scan تمیز؛ هیچ فایل داده‌ای tracked نیست |
| 31 | همه‌ی فرض‌های تاریخی پشتیبانی‌نشده در بخش «Limitations» فهرست شده‌اند (بررسی مستندی، نه خودکار) |

---

## 9. آنچه این Plan عمداً انجام نمی‌دهد

- Pump Label، Onset Detector، gate زنده، feature engineering، و هر مدل — هیچ‌کدام.
- انتخاب DB، Feature Store، message bus، cloud یا معماری مدل — فقط ثبت trade-off.
- ساخت universe نهایی — Smoke Universe صریحاً **provisional و غیرعلمی** برچسب می‌خورد.
- هر ادعای پوشش تاریخی که منبع پشتیبانی نمی‌کند — به‌جای ادعا، `gap` ثبت می‌شود.

---

## 10. توقف

مطابق دستور Kickoff:

> **Stop after the plan and wait for Majid's approval. Do not implement yet.**

پیاده‌سازی شروع نمی‌شود تا: (۱) تأیید این Plan، و (۲) پاسخ به چهار سؤال بخش ۴.۳.
