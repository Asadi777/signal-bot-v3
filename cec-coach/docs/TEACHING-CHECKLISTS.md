# چک‌لیستِ تدریسِ جلسه‌به‌جلسه — CEC 2024
**سیلابسِ استانداردِ معلم.** هر جلسه را همین‌طور جلوی خودت بگذار تا چیزی از قلم نیفتد.

نمادها:
✅ کارِ معلم · 🗣️ چیزی که باید بگویم · 🔑 **Rule/Table هایی که دانشجو باید highlight و حفظ کند** (عددِ داخلِ پرانتز = تعدادِ تکرار در بانکِ ۴۴۰سؤالی، یعنی اولویت) · ⚠️ تله‌ها · 📝 تکلیف

> الگوی ثابتِ هر جلسه: **(۱)** keyword-spotting → **(۲)** خواندنِ Table/Rule → **(۳)** حلِ نمونه‌ی محاسباتی → **(۴)** Code-Hunt با تایمر + mini-quiz.

---

## جلسه ۱ — Foundations: Section 0 (Definitions) + Section 2 (General & Working Space) + روشِ کار با کتاب
✅ کارِ معلم
- [ ] tab نزدن روی کتاب را یادآوری کن (کتابِ امتحان تمیز است) و روشِ پیداکردن از **Index** و **Table of Contents** را عملی نشان بده.
- [ ] متدِ ما را آموزش بده: keyword → کدام Rule/Table → کجای کتاب (صفحه) → چطور خوانده می‌شود.
- [ ] دانشجو را وادار کن تعریف‌های Section 0 را با صدای بلند بخواند (کلمه‌به‌کلمه مهم است).

🗣️ مطالبی که باید بگم
- تعریف‌های کلیدیِ Section 0: **Bonding** (مسیرِ کم‌امپدانسِ دائمی برای fault current)، **Grounding**، **Feeder** در مقابلِ **Branch circuit**، **Overcurrent** (محافظِ سیم) در مقابلِ **Overload** (محافظِ موتور)، **Ampacity**.
- کلاس‌های ولتاژ: Extra-Low = ≤30 V · Low = >30 تا 750 V · High = >750 V (مرز inclusive است).
- working space جلوی تجهیزات **1.0 m** با جای‌پای محکم (Rule 2-308 1)؛ headroom روی switchboard/MCC **2.2 m**.
- dwelling: voltage-to-ground حداکثر **150 V** (Rule 2-110).
- working space برای transformer فقط **> 50 kVA** لازم می‌شود (خودِ 50 هنوز نه).
- رمزِ enclosure: 3R=rainproof · 4=watertight/hose · 4X=watertight+corrosive.

🔑 highlight و حفظ
- **Rule 2-308** (12) · **Table 56** (12) · **Table 65** (10) · **Appendix B** (7) · **Rule 2-110** (5) · **Rule 2-328** (4) · **Rule 2-130** (3)
- تعریف‌های Section 0: Bonding · Feeder/Branch · Overcurrent/Overload.

⚠️ تله‌ها
- ورودیِ **Table 56**، line-to-ground voltage است؛ اگر قبلاً تقسیم بر √3 شده، دوباره تقسیم نکن.
- فاصله‌ی gas-meter در **Appendix B** است، نه در متنِ Rule 2-328 (آدم سرِ rule می‌ایستد و می‌بازد).
- **2-110** قانونِ 150 V مسکونی است؛ **2-130** مالِ flame-spread است — جابه‌جا نشود.

📝 تکلیف: ۲۰ تعریفِ Section 0 را بنویس + ۵ سؤالِ working space.

---

## جلسه ۲ — Occupational / Safety Skills (Block A)
✅ کارِ معلم
- [ ] استانداردهای ایمنی را معرفی کن: **CSA Z462** (arc-flash/PPE) و **lockout/tagout**.
- [ ] ترتیبِ lockout را روی تخته بنویس و تمرین کن.
- [ ] نحوه‌ی استفاده از **Appendix D** (داده‌های torque و conductor) را نشان بده.

🗣️ مطالبی که باید بگم
- گامِ اولِ lockout همیشه **IDENTIFY** همه‌ی منابعِ انرژی است؛ بعد جداسازی، قفل، و تاییدِ صفرِ انرژی.
- خطرِ ترانس: **primary = back-feed** · **secondary = arc flash**.
- torque: یراقِ 1/2 in ≈ **54 N·m** · پیچِ 1/4 in (HVAC) ≈ **8 N·m** (Table D7).
- available short-circuit current = FLCِ ثانویه ÷ %Z (مثال: 833 A ÷ 0.03 = 27.8 kA).
- سیگنالِ **4–20 mA**: خروجی = 4 + (٪باز × 16) ⇒ ‌85٪ = 17.6 mA.
- GFCI کلاس A در **5 mA** قطع می‌کند؛ دستکشِ HV هر ۶ ماهِ در حالِ استفاده re-test.

🔑 highlight و حفظ
- مرجعِ این Section مفهومی است؛ به دانشجو بگو در کتاب علامت بزند: **Rule 2-308** (working space) · **Table D7** (torque) · **Appendix D** · **Section 64** (PV) · **Section 46** (emergency systems).

⚠️ تله‌ها
- در lockout، گامِ اول **identify** است، نه نصبِ hasp یا پر کردنِ log.
- **primary=back-feed / secondary=arc flash** را جابه‌جا نکن.
- authority تابلوی **exit/smoke** برای نصب **NBC** است، اما سایزبندیِ برقی‌اش CEC.
- ترتیبِ VFD startup: اول چرخشِ آزادِ موتور → program کردنِ parameters → بعد bump/rotation test.

📝 تکلیف: یک procedure ‌lockout بنویس + ۵ سؤالِ محاسبه‌ی short-circuit/4-20 mA.

---

## جلسه ۳–۴ — Section 4: Conductors & Ampacity (پایه‌ی همه‌چیز · محاسباتی)
> این پایه‌ی Loads، Motors و Wiring است؛ محکم تدریس شود.

✅ کارِ معلم
- [ ] **table grid** را روی تخته بکش: T1=free-air Cu · T2=raceway Cu · T3=free-air Al · T4=raceway Al.
- [ ] یک مثالِ کاملِ derating (grouping × ambient) قدم‌به‌قدم حل کن.
- [ ] مثالِ «minimum size» (تقسیم بر factors → next standard size) را تمرین بده.

🗣️ مطالبی که باید بگم
- زنجیره‌ی **Rule 4-004**: ampacity پایه از Table 1–4 → اصلاحِ **Table 5C** (grouping) و **Table 5A** (ambient).
- ضرایبِ **Table 5C**: 4–6=0.80 · 7–24=0.70 · 25–42=0.60 — فقط وقتی **>3** current-carrying conductor.
- **Table 5A** فقط وقتی ambient **>30 °C**؛ دما را همیشه رو به بالا گرد کن.
- فقط **current-carrying** ها را بشمار؛ bonding و balanced neutral شمرده نمی‌شوند.
- **Rule 4-006** (termination): کابلِ 90°C روی ترمینالِ 75°C ⇒ به ستونِ 75°C محدود.
- **Rule 4-038**: در raceway با دماهای mixed، همه را از پایین‌ترین rating column بخوان.
- overhead service در NS: **Table 36B** (Cu) / **Table 36A** (Al)؛ min service = #10 Cu / #8 Al.

🔑 highlight و حفظ (به ترتیبِ اولویت)
- **Rule 4-004** (35) · **Table 5C** (27) · **Table 2** (24) · **Table 5A** (16) · **Table 36B** (11) · **Table 36A** (9) · **Table 12** (9) · **Appendix D** (8) · **Rule 4-038** (7) · **Table 3** (7) · **Rule 4-014** (6) · **Table 1** (5)

⚠️ تله‌ها
- خواندن از ستونِ 90°C وقتی terminal روی 75°C است (Rule 4-006).
- رفتن سراغِ Table 3 (free-air Al) برای کابلی که فقط **sheath**اش aluminum است — جدول را فلزِ هادی تعیین می‌کند.
- در free air، اگر spacing ذکر نشده اصلاً derate نکن.
- شمردنِ bonding یا balanced neutral در Table 5C.

📝 تکلیف: ۱۰ مسئله‌ی ampacity شاملِ grouping+ambient و ۳ مسئله‌ی «minimum size».

---

## جلسه ۵ — Section 8: Loads & Demand (محاسباتی)
✅ کارِ معلم
- [ ] محاسبه‌ی کاملِ single-dwelling service را از صفر تا آمپر حل کن.
- [ ] جدولِ base ها را بنویس تا قاطی نشود.

🗣️ مطالبی که باید بگم
- **VD limits (تغییرِ 2024)**: 3% branch · 3% feeder · **5% total**.
- range base: در service/feeder (**8-200**) = 6 kW · در dwelling branch (**8-300**) = 8 kW؛ هر دو 40% مازاد بر 12 kW.
- basic dwelling: 5000 W برای 90 m² اول + 1000 W هر 90 m² بعدی (incrementِ ناقص رو به بالا).
- min dwelling service: ≥80 m² ⇒ 24 000 W ⇒ **100 A** · <80 m² ⇒ 14 400 W ⇒ **60 A** (بزرگ‌تر برنده).
- continuous load (**8-104**): دستگاهِ non-100% فقط 80% of rating خود.
- heating (62-118): 100% برای 10 kW اول، 75% برای مابقی.

🔑 highlight و حفظ
- **Rule 8-200** (9) · **Rule 8-102** (8) · **Rule 8-104** (7) · **Rule 8-300** (7) · **Rule 8-210** (6) · **Rule 8-400** (6) · **Table 14** (5) · **Rule 8-106** (3)

⚠️ تله‌ها
- **connected load** را با **calculated load** قاطی نکن (وات‌های appliance دامند).
- range base: 6 kW (service) ≠ 8 kW (branch).
- 3%+3% ≠ 5%: اگر branch کل 3% را خورد، feeder فقط 2%.
- non-dwelling range هیچ reduction نمی‌گیرد (100%).
- در 3-phase wye، W→A یعنی ÷ (√3 × line voltage).

📝 تکلیف: ۲ محاسبه‌ی کاملِ service (یک خانه، یک apartment) + ۵ سؤالِ range/heating.

---

## جلسه ۶ — Section 6: Services & Metering
✅ کارِ معلم
- [ ] clearance های overhead را روی یک شکل بکش (attachment, highway, drip loop).
- [ ] روی Rule 6-300 (conductors داخلِ building) تأکیدِ ویژه کن.

🗣️ مطالبی که باید بگم
- attachmentِ overhead: min **3.5 m** (pedestrian) · max **9 m**؛ عبور از highway/vehicle: min **5.5 m** (Table 32).
- service mast: metal، min rigid steel، trade-size **63 (2-1/2)**؛ projection بدونِ guy ~1.5 m.
- min service conductor: **#10 Cu / #8 Al** (Rule 6-302) — یک code minimum.
- conductors داخلِ building باید در **≥50 mm concrete** embed شوند (Rule 6-300).
- service equipment: headroom **≥2 m** (Rule 6-206)؛ drip loop ≥750 mm؛ raceway min 21 (3/4).

🔑 highlight و حفظ
- **Rule 6-112** (20) · **Rule 6-206** (8) · **Rule 6-116** (5) · **Table 32** (4) · **Rule 6-102** (3) · **Rule 6-104** (3) · **Rule 6-300** (2)

⚠️ تله‌ها
- «max 9 m attachment» را با «min 5.5 m highway» قاطی نکن.
- #10 Cu/#8 Al = minimum، نه «خیلی کوچک»؛ سؤال minimum می‌خواهد.
- mast min (63) ≠ raceway min (21).
- سقفِ consumer service = **چهار** (نه unlimited).

📝 تکلیف: ۸ سؤالِ clearance + یک نقشه‌ی مسیرِ service.

---

## جلسه ۷ — Section 14: Overcurrent Protection
✅ کارِ معلم
- [ ] دو شرطِ GFP را به‌صورتِ منطقیِ AND روی تخته بنویس.
- [ ] مثالِ سایزبندیِ OCPD با Table 13 (next lower) حل کن.

🗣️ مطالبی که باید بگم
- **GFP (14-102)** فقط وقتی **هر دو** شرط: service **≥1000 A** **و** **>150 V to ground**. سیستمِ 208Y/120 هیچ‌وقت مشمول نیست.
- clearing time: ground-fault **≥3000 A** باید ظرفِ **≤1 s** قطع شود.
- **overcurrent** (Section 14) محافظِ **conductor** ≠ **overload** (Section 28) محافظِ **motor**.
- سایزبندی (**14-104**): rating ≤ conductor ampacity؛ اگر دقیق نخورد، **next lower** از Table 13.
- interrupting rating پیش‌فرض = **10 000 A** symmetrical (14-606)؛ plug fuse max 30 A.

🔑 highlight و حفظ
- **Rule 14-102** (14) · **Rule 14-104** (10) · **Table 13** (10) · **Rule 14-012** (6) · **Rule 14-100** (5) · **Table 2** (4)

⚠️ تله‌ها
- GFP وقتی فقط **یک** شرط برقرار است (2000 A/208 V یا 800 A/347 V) ⇒ جواب «خیر».
- overcurrent را با overload قاطی نکن.
- breaker را روی load سایز نکن (روی conductor ampacity)، و **next lower** را با next higher اشتباه نگیر.
- «>150 V» یعنی 347 V مشمول، ولی خودِ 150 V نه.

📝 تکلیف: ۶ سؤالِ GFP (yes/no با دلیل) + ۴ سایزبندیِ OCPD.

---

## جلسه ۸ — Section 10: Grounding & Bonding
✅ کارِ معلم
- [ ] تفاوتِ **Table 43** (electrode conductor) و **Table 16** (bonding) را با دو ستون مقابلِ هم بنویس.
- [ ] یک مثالِ انتخابِ grounding electrode conductor از service ampacity حل کن.

🗣️ مطالبی که باید بگم
- **Table 43** ← grounding electrode conductor، کلیدش **service-conductor ampacity** (مثال: 4/0 Cu @75°C = 230 A ⇒ size 2).
- **Table 16** ← bonding conductor، بر اساسِ ampacity یا OCPD rating.
- manufactured electrode (rod/plate): grounding conductor لازم نیست **>#6 Cu** باشد (10-114).
- field electrode: ≥6 m bare Cu، ≥#4، در 50 mm پایینیِ footing، ≥600 mm زیرِ grade؛ spacing ≥3 m.
- equipotential bonding (10-406): ≥#6 Cu یا #4 Al؛ bare-in-raceway فقط اگر run ≤15 m (10-806).

🔑 highlight و حفظ
- **Table 43** (17) · **Table 16** (14) · **Rule 10-102** (10) · **Rule 10-114** (9) · **Rule 10-406** (8) · **Table 2** (6) · **Rule 10-614** (6) · **Rule 10-616** (4) · **Rule 10-104** (4) · **Rule 10-806** (4)

⚠️ تله‌ها
- **Table 43 vs 16**: electrode فقط ampacity؛ bonding هم ampacity هم OCPD.
- Table 43 همیشه با service-conductor ampacity (نه breaker، نه load).
- سقفِ #6 Cu برای manufactured electrode را فراموش نکن.
- aluminum یک size بزرگ‌تر از copper (#4 Al ≈ #6 Cu).
- PV DC grounding مالِ **Section 64** است نه Section 10.

📝 تکلیف: ۵ انتخابِ electrode conductor + ۵ انتخابِ bonding jumper.

---

## جلسه ۹–۱۰ — Section 26: Equipment & Receptacles
✅ کارِ معلم
- [ ] قانونِ spacing را با یک نقشه‌ی دیوارِ آشپزخانه نشان بده.
- [ ] فهرستِ «کجا GFCI لازم است / کجا AFCI» را بساز.

🗣️ مطالبی که باید بگم
- spacing (**26-712**): هیچ نقطه‌ای **>1.8 m** از یک receptacle نباشد ⇒ receptacle ها **≤3.6 m** از هم.
- outdoor residential تا **2.5 m از grade** = Class A **GFCI** (26-714)؛ هر receptacle تا **1.5 m از sink** = GFCI (26-700).
- **combination AFCI** برای پریزِ 125 V خانگیِ ≤20 A (26-722).
- min **۲ پریزِ outdoor** در dwelling اجباری (26-724).
- panelboard handle max **1.7 m**؛ centre پریزِ range max 130 mm؛ capacitor conductor ≥135% (26-210).

🔑 highlight و حفظ
- **Rule 26-712** (8) · **Rule 26-700** (8) · **Rule 26-722** (8) · **Table 50** (7) · **Rule 26-244** (6) · **Rule 26-714** (6) · **Rule 26-210** (5) · **Rule 26-302** (5) · **Rule 26-012** (5) · **Rule 26-744** (3)

⚠️ تله‌ها
- **26-712** را «1.8 m apart» نخوان؛ 1.8 m فاصله‌ی نقطه تا پریز است، خودِ پریزها 3.6 m.
- GFCI روی dedicated پریزِ یک stationary appliance لازم نیست (exception).
- **135% capacitor (26-210)** را با **125% motor branch (28-106)** قاطی نکن.
- ستونِ Table 50 برای transformer وقتی primary >750 V فرق می‌کند.

📝 تکلیف: نقشه‌ی پریزهای یک خانه با علامتِ GFCI/AFCI + ۸ سؤالِ spacing/height.

---

## جلسه ۱۱–۱۴ — Section 12: Wiring Methods (بزرگ‌ترین موضوع · ۴ جلسه)
> ~۲۱٪ امتحان. تقسیمِ پیشنهادی: **۱۱** conduit fill · **۱۲** box fill + pull boxes · **۱۳** bends/support/cover · **۱۴** cable types + جمع‌بندی.

✅ کارِ معلم
- [ ] زنجیره‌ی fill را روی تخته: Table 10 (areas) → Table 8 (%) → Table 6/9 (conduit size).
- [ ] یک conduit fill و یک box fill کامل حل کن؛ تفاوتِ CEC با NEC را تأکید کن.
- [ ] straight pull (8×) و angle/U pull (6×+بقیه) را با دو شکل جدا کن.

🗣️ مطالبی که باید بگم
- **fill %** (Table 8): one=53% · two=31% · **three+=40%**. **bonding conductor در conduit fill شمرده می‌شود**.
- **box fill (CEC)**: هر device = یک conductorِ هم‌اندازه‌ی بزرگ‌ترین؛ bonding **volume جدا نمی‌گیرد**؛ wire connector جدا شمرده **نمی‌شود** (برخلافِ NEC).
- **pull box (12-3036)**: straight = 8× بزرگ‌ترین raceway · angle/U = 6× بزرگ‌ترین + مجموعِ بقیه‌ی همان دیوار.
- bends: max **360° (چهار 90°)** بینِ دو pull point (12-940)؛ bend radius از **Table 7** بر اساسِ trade size.
- cover (**Table 53**): ≤750 V بدونِ protection = 600 mm (non-vehicular) / 900 mm (vehicular)؛ protection ⇒ −150 mm.
- studs: 32 mm از لبه یا steel plate (12-516)؛ ≥150 mm free conductor در هر box (12-3014).
- voltage: **NMD90=300 V · AC90≤2000 V · TECK90≤5000 V**؛ parallel فقط از #1/0 (12-108)؛ هیچ splice در conduit (12-912).

🔑 highlight و حفظ (بالاترین اولویتِ کلِ کتاب)
- **Table 6** (18) · **Table 8** (18) · **Table 10** (17) · **Table 53** (16) · **Table 7** (14) · **Rule 12-910** (13) · **Rule 12-3036** (13) · **Table 22** (12) · **Rule 12-552** (12) · **Table 23** (10) · **Rule 12-3034** (9) · **Rule 12-012** (8)

⚠️ تله‌ها
- box fill: شمردنِ bonding به‌صورتِ volume جدا یا شمردنِ wire connector (در CEC نه).
- با **three+** باز هم 53%/31% زدن (تقریباً همیشه جواب **40%** است).
- pull box: زدنِ 8× برای entryِ angle/U.
- bend radius از Table 7 به trade size لوله بسته است نه conductor.
- cover: فراموشیِ کاهشِ **150 mm** با protection.
- **bonding در conduit fill شمرده می‌شود ولی در box fill نه** (دقیقاً برعکسِ هم).

📝 تکلیف: ۵ conduit fill + ۵ box fill + ۵ pull-box/bend + جدولِ voltage rating کابل‌ها.

---

## جلسه ۱۵–۱۸ — Section 28: Motors & Control (محاسباتی · دومین وزن · ۴ جلسه)
> ~۲۰٪ امتحان. تقسیم: **۱۵** FLC + branch conductor · **۱۶** branch OCPD (Table 29) · **۱۷** overload · **۱۸** feeder + disconnect + جمع‌بندی.

✅ کارِ معلم
- [ ] روی تخته بنویس: **conductor/branch از Table 44/45 FLC** ولی **overload از nameplate FLA**.
- [ ] یک مدارِ کاملِ موتور را end-to-end سایز کن (conductor → OCPD → overload → disconnect).

🗣️ مطالبی که باید بگم
- **FLC** از **Table 44** (3φ) / **Table 45** (1φ) — **نه از nameplate**.
- branch conductor = **125% FLC** (28-106)؛ feeder = 125%×بزرگ‌ترین + Σ بقیه (28-108).
- overload از **nameplate FLA**: ×1.25 (اگر SF≥1.15 یا rise≤40°C) وگرنه ×1.15 (28-306).
- branch OCPD (**Table 29**): time-delay fuse 175% (225% won't-start) · non-time-delay 300% · inverse-time CB 250% · instantaneous 1300%.
- OCPD را به **next standard size که not exceeding** گرد کن؛ فقط اگر موتور راه نیفتاد بالاتر.
- disconnect: هم within sight هم **≤9 m** (28-604)؛ single-phase فقط 1 overload device.

🔑 highlight و حفظ
- **Table 44** (28) · **Table 29** (15) · **Rule 28-108** (11) · **Table 27** (11) · **Rule 28-200** (10) · **Rule 28-306** (8) · **Rule 28-204** (5) · **Rule 28-300** (4) · **Rule 28-604** (4)

⚠️ تله‌ها
- nameplate FLA برای conductor/branch fuse (باید Table 44/45 باشد) — و برعکسش: Table 44 برای overload (باید nameplate باشد).
- overcurrent (محافظِ wire) را با overload (محافظِ motor) قاطی نکن.
- وقتی «won't start» نگفته، OCPD را به next-higher گرد نکن.
- duty-cycle motor: به‌جای 125% از درصدِ **Table 27**.
- DC motor FLC از **Table D2** می‌آید نه Table 44.
- در feeder فقط **largest** motor یک 125% می‌گیرد؛ بقیه plain FLC.

📝 تکلیف: ۳ مدارِ کاملِ موتور (single + 3-phase + feeder چندموتوره) + ۵ overload.

---

## جلسه ۱۹–۲۰ — Section 16: Class 1 & 2 / Signalling
✅ کارِ معلم
- [ ] جدولِ Class 1 vs Class 2 (ولتاژ/VA/wire) را بساز.
- [ ] دو separation در 16-212 را با مثال جدا کن.

🗣️ مطالبی که باید بگم
- **ELV = ≤30 V** (مرز inclusive).
- **Class 1 ELV power**: max 30 V و **1000 VA** (16-200).
- **Class 2**: power-limited؛ بالای 30 V قفل به **100 VA** ⇒ max fuse = 100 VA ÷ voltage؛ smallest conductor تا #30 AWG.
- separation (خارج از raceway): **50 mm** در ≤300 V · **600 mm** در ولتاژِ بالاتر (16-212).
- **fire alarm = Class 1** (life-safety)، نه Class 2؛ wireِ ELC را برای fire alarm به کار نبر.

🔑 highlight و حفظ
- **Rule 16-210** (19) · **Rule 16-200** (16) · **Rule 16-212** (11) · **Table 11** (4) · **Rule 32-100** (4) · **Rule 16-100** (3)

⚠️ تله‌ها
- fire alarmِ 24 V را به‌خاطرِ ولتاژِ پایین Class 2 نَنام — **Class 1** است.
- equipment wire (TEW) برای open wiringِ Class 2 ممنوع.
- 50 mm (≤300 V) و 600 mm (بالاتر) را جابه‌جا نکن.
- 100 VA همان max fuse را تعیین می‌کند، نه یک amp ثابت.

📝 تکلیف: ۶ سؤالِ Class 1/2 (شناسایی + fuse) + جمع‌بندیِ کلِ دوره.

---

## بعد از جلسه ۲۰ — مرور و آزمون
- بعد از هر بلوک یک mini-mock کوتاه بگیر.
- دو هفته‌ی آخر: **۲–۳ آزمونِ کاملِ ۱۰۰سؤالیِ تایم‌دار (۲۴۰ دقیقه)** + مرورِ Weak Areas.
- توزیعِ سؤال‌ها را دقیقاً مطابقِ جدولِ وزن در `TEACHING-SYLLABUS.md` بچین.
