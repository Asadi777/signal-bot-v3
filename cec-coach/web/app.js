/* CEC Coach — web app  (v1.5.0 — bilingual study, mobile tables, seeded generator)
   Offline quiz bank + AI features (tutor / generator / vision) via the Claude API.
   The API key lives only in this browser (localStorage) and is sent straight to
   Anthropic with the direct-browser-access header — no backend needed. */

(function () {
  "use strict";

  // ---------- data ----------
  var DATA = window.CEC_DATA || { questions: [], blueprint: {} };
  var ALL = DATA.questions || [];
  var BP = DATA.blueprint || {};

  var SECTION_NAMES = {
    0: "Definitions", 2: "General & Working Space", 4: "Conductors & Ampacity",
    6: "Services & Metering", 8: "Loads & Demand", 10: "Grounding & Bonding",
    12: "Wiring Methods", 14: "Overcurrent Protection", 16: "Class 1 & 2 Circuits",
    26: "Equipment & Receptacles", 28: "Motors & Control"
  };

  // ---------- tiny DOM helpers ----------
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function show(id) {
    var s = document.querySelectorAll(".screen");
    for (var i = 0; i < s.length; i++) s[i].classList.add("hidden");
    $(id).classList.remove("hidden");
    window.scrollTo(0, 0);
  }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); }
  function shuffle(a) { a = a.slice(); for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)); var t = a[i]; a[i] = a[j]; a[j] = t; } return a; }

  // ---------- markdown -> pretty HTML (tables, lists, hr, code, RTL-aware) ----------
  function mdInline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    return s;
  }
  function mdRow(line) {
    return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(function (c) { return c.trim(); });
  }
  // strip any LaTeX the model slips in, into readable plain text
  function deLatex(s) {
    if (!s) return s;
    s = s.replace(/\$\$([\s\S]*?)\$\$/g, function (_, x) { return x; });
    s = s.replace(/\\\[([\s\S]*?)\\\]/g, function (_, x) { return x; });
    s = s.replace(/\\\(([\s\S]*?)\\\)/g, function (_, x) { return x; });
    s = s.replace(/\$([^$\n]+)\$/g, function (_, x) { return x; });
    s = s.replace(/\{,\}/g, ",").replace(/\\,/g, " ").replace(/\\;/g, " ").replace(/\\!/g, "");
    s = s.replace(/\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)");
    s = s.replace(/\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)");
    s = s.replace(/\\sqrt\s*\{([^{}]*)\}/g, "√($1)");
    s = s.replace(/\\(?:text|mathrm|mathbf|textbf|operatorname)\s*\{([^{}]*)\}/g, "$1");
    var map = { "\\times": "×", "\\cdot": "·", "\\div": "÷", "\\approx": "≈", "\\leq": "≤", "\\le": "≤", "\\geq": "≥", "\\ge": "≥", "\\neq": "≠", "\\pm": "±", "\\Omega": "Ω", "\\omega": "ω", "\\pi": "π", "\\circ": "°", "\\rightarrow": "→", "\\Rightarrow": "⇒", "\\to": "→", "\\sqrt": "√" };
    Object.keys(map).forEach(function (k) { s = s.split(k).join(map[k]); });
    s = s.replace(/\\\\/g, "\n").replace(/\\left|\\right/g, "");
    s = s.replace(/\^\{([^{}]*)\}/g, "^$1").replace(/_\{([^{}]*)\}/g, "_$1");
    s = s.replace(/\\([%$#&_{}])/g, "$1");
    s = s.replace(/\\([a-zA-Z]+)\b/g, "$1");
    return s;
  }
  function mdToHtml(md) {
    if (!md) return "";
    md = deLatex(md);
    var lines = md.replace(/\r\n/g, "\n").split("\n"), out = [], i = 0;
    while (i < lines.length) {
      var line = lines[i];
      if (/^```/.test(line)) {
        var buf = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++; out.push("<pre><code>" + esc(buf.join("\n")) + "</code></pre>"); continue;
      }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { out.push("<hr>"); i++; continue; }
      var h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { var lv = h[1].length; out.push("<h" + lv + ">" + mdInline(h[2]) + "</h" + lv + ">"); i++; continue; }
      if (/\|/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && /-/.test(lines[i + 1])) {
        var head = mdRow(line); i += 2; var rows = [];
        while (i < lines.length && /\|/.test(lines[i]) && lines[i].trim() !== "") { rows.push(mdRow(lines[i])); i++; }
        var t = "<table><thead><tr>" + head.map(function (c) { return "<th>" + mdInline(c) + "</th>"; }).join("") + "</tr></thead><tbody>";
        rows.forEach(function (r) { t += "<tr>" + r.map(function (c) { return "<td>" + mdInline(c) + "</td>"; }).join("") + "</tr>"; });
        out.push(t + "</tbody></table>"); continue;
      }
      if (/^\s*[-*+]\s+/.test(line)) {
        var li = [];
        while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) { li.push("<li>" + mdInline(lines[i].replace(/^\s*[-*+]\s+/, "")) + "</li>"); i++; }
        out.push("<ul>" + li.join("") + "</ul>"); continue;
      }
      if (/^\s*\d+[.)]\s+/.test(line)) {
        var ol = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) { ol.push("<li>" + mdInline(lines[i].replace(/^\s*\d+[.)]\s+/, "")) + "</li>"); i++; }
        out.push("<ol>" + ol.join("") + "</ol>"); continue;
      }
      if (line.trim() === "") { i++; continue; }
      var para = [line]; i++;
      while (i < lines.length && lines[i].trim() !== "" &&
        !/^(#{1,6}\s|```|\s*[-*+]\s|\s*\d+[.)]\s)/.test(lines[i]) &&
        !/^\s*([-*_])(\s*\1){2,}\s*$/.test(lines[i]) && !/\|/.test(lines[i])) { para.push(lines[i]); i++; }
      out.push("<p>" + para.map(mdInline).join("<br>") + "</p>");
    }
    return out.join("\n");
  }

  // animated "thinking… Ns" placeholder until the first token streams in
  function thinking(node, baseClass) {
    var t0 = Date.now();
    function upd() { node.className = baseClass; node.innerHTML = "<p class='waiting'>⏳ Thinking… " + Math.round((Date.now() - t0) / 1000) + "s</p>"; }
    upd(); var iv = setInterval(upd, 1000);
    return { stop: function () { clearInterval(iv); } };
  }


  // ---------- progress (weak-area tracking) ----------
  var PROG = JSON.parse(localStorage.getItem("cec_progress") || "{}"); // id -> {seen, wrong}
  function saveProg() { localStorage.setItem("cec_progress", JSON.stringify(PROG)); }
  function record(id, correct) {
    var p = PROG[id] || { seen: 0, wrong: 0 };
    p.seen++; if (!correct) p.wrong++;
    PROG[id] = p; saveProg();
  }

  // ---------- settings ----------
  function getKey() { return localStorage.getItem("cec_apikey") || ""; }
  function getModel() { return localStorage.getItem("cec_model") || "claude-opus-4-8"; }

  // ---------- Claude API (direct from browser, streaming) ----------
  // messages: [{role, content}]; opts: {system, maxTokens, onText}
  function callClaude(messages, opts) {
    opts = opts || {};
    var key = getKey();
    if (!key) return Promise.reject(new Error("NO_KEY"));
    var body = {
      model: getModel(),
      max_tokens: opts.maxTokens || 3000,
      stream: true,
      thinking: { type: "adaptive" },
      messages: messages
    };
    if (opts.system) body.system = opts.system;

    return fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true"
      },
      body: JSON.stringify(body)
    }).then(function (res) {
      if (!res.ok) {
        return res.text().then(function (t) {
          var msg = t;
          try { msg = JSON.parse(t).error.message; } catch (e) {}
          throw new Error(msg || ("HTTP " + res.status));
        });
      }
      var reader = res.body.getReader();
      var dec = new TextDecoder();
      var buf = "", full = "";
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) return full;
          buf += dec.decode(r.value, { stream: true });
          var lines = buf.split("\n");
          buf = lines.pop();
          for (var i = 0; i < lines.length; i++) {
            var line = lines[i];
            if (line.indexOf("data:") !== 0) continue;
            var data = line.slice(5).trim();
            if (!data || data === "[DONE]") continue;
            var ev;
            try { ev = JSON.parse(data); } catch (e) { continue; }
            if (ev.type === "error") { throw new Error((ev.error && ev.error.message) || "API stream error"); }
            if (ev.type === "content_block_delta" && ev.delta && ev.delta.type === "text_delta") {
              full += ev.delta.text;
              if (opts.onText) opts.onText(full);
            }
          }
          return pump();
        });
      }
      return pump();
    });
  }

  var TUTOR_SYSTEM =
    "You are CEC Coach, a tutor for the BC Construction Electrician (Red Seal) exam on the " +
    "Canadian Electrical Code 2024 (CSA C22.1:24). The exam is open-book with a CLEAN un-tabbed code book, " +
    "100 multiple-choice questions, 70% to pass, 5 hours. The student is a native Persian (Farsi) speaker.\n\n" +
    "TEACHING STYLE (always follow):\n" +
    "1. Answer in clear ENGLISH first (exam language), then give a short Persian (فارسی) explanation of the key idea.\n" +
    "2. Always name the KEYWORD(s) in the question that point to the answer, and the exact Rule/Table to jump to.\n" +
    "3. Give the fastest step-by-step path to the answer (open-book speed matters).\n" +
    "4. Be precise with CEC 2024 references; if unsure of an exact table value, say so rather than inventing it.\n\n" +
    "KEYWORD→TABLE MAP (CEC 2024): ampacity→Tables 1-4 (free air: T1 Cu/T3 Al; raceway/cable: T2 Cu/T4 Al, start 90°C col); " +
    ">3 conductors→Table 5C derate; ambient>30°C→Table 5A; terminal temp marked→Rule 4-006; continuous load→Rule 8-104 (125%); " +
    "single dwelling demand→Rule 8-200; basic load W/m²→Table 14; voltage drop→Rule 8-102 + Table D3 (3% branch/3% feeder/5% total; " +
    "2024 = one formula VD=(K×I×L×F)/1000, F=2 default, F=1.73 for 3-phase); grounding electrode conductor→Table 43 (by service ampacity); " +
    "bonding conductor→Table 16; box fill→Rule 12-3036 + Table 23; conduit fill→Rule 12-910 + Tables 6/8/9 (1→53%,2→31%,3+→40%); " +
    "bends→Rule 12-936 (max 360°); cover/burial→Table 53; cable from stud→32mm (Rule 12-516); overcurrent→Rule 14-104 + Table 13; " +
    "ground-fault protection→Rule 14-102 (≥1000A AND >150V to ground); receptacle spacing→Rule 26-712 (≤1.8m); " +
    "motor FLC→Table 44 (3ph)/45 (1ph); motor branch conductor→Rule 28-106 (125% FLC); overload→Rule 28-306 (nameplate FLA); " +
    "motor fuse/breaker→Table 29; working space→Rule 2-308 + Table 56; voltage to ground in dwelling→Rule 2-110 (150V); " +
    "enclosure type→Table 65. Unknown defined term→Section 0.\n\n" +
    "TEACH like a patient teacher explaining to a brand-new beginner who is nervous about the exam. Be encouraging and concrete. ALWAYS give the FULL explanation in BOTH English AND Persian (فارسی) — not just a short Persian summary, but a real Persian explanation of every step, because the student is a native Persian speaker.\n\n" +
    "CRITICAL BILINGUAL RULE: the student studies from the ENGLISH code book and must recognize the exact words. So in the Persian text, KEEP these in ENGLISH (Latin), never translate them: material names & types (copper, aluminum, conductor, NMD90, TECK90, RW90…), Table names AND their parts/columns (e.g. Table 2, the 90°C column, free air, raceway), Rule/Section names & numbers (Rule 8-200, Section 0), defined terms / equipment / electrical nouns (ampacity, conductor, raceway, conduit, breaker, fuse, overcurrent, overload, branch circuit, feeder, service, grounding, bonding, bonding jumper, receptacle, GFCI, capacitor, transformer, enclosure, nameplate, full-load current/FLC, voltage drop, working space, disconnect, derate, ambient…), plus all numbers and units. Only the connective/explanatory prose (the 'why', verbs, conjunctions) is Persian. First use of a term may add a 1-2 word Persian gloss in parentheses. Example: 'برای ampacity یک copper conductor داخلِ raceway، برو Table 2 و the 90°C column.'\n\n" +
    "FORMATTING (always): clean Markdown — use ## headings, **bold** for the final answer and for each keyword, numbered steps, and Markdown TABLES for any lookup/comparison/calculation. " +
    "NEVER use LaTeX or MathJax — do NOT use $ … $, $$ … $$, \\frac, \\text, \\times, or any backslash commands. Write ALL math in plain readable text with ÷ × = ( ) and units, e.g. `I = 100000 ÷ 600 = 166.67 A`; for a multi-line calculation use a fenced code block so it lines up. " +
    "Separate major parts with a horizontal rule (---). When a figure helps (circuit, conduit cross-section, panel/box layout, one-line), draw a clear ASCII/Unicode diagram in a ``` fenced code block. Structure each answer as: \n## ✅ Answer\n## 🔑 Keyword(s) you should spot — quote the exact words in the question and say what they signal.\n## 📖 Which Table/Rule & why — name it and explain WHY that is the right place.\n## 🧭 Step-by-step to the answer (numbered, the fastest open-book path)\n## 🇮🇷 توضیح کامل فارسی — همه‌چیز را روان و کامل به فارسی توضیح بده (کلمهٔ کلیدی، کدام جدول/قانون و چرا، و قدم‌به‌قدم چطور به جواب می‌رسیم).";

  // ---------- student profile (personalization memory) ----------
  var PROFILE = JSON.parse(localStorage.getItem("cec_profile") || '{"sections":{},"topics":{},"exams":0,"answered":0,"correct":0}');
  function saveProfile() { localStorage.setItem("cec_profile", JSON.stringify(PROFILE)); }
  function profileRecord(q, ok) {
    var s = (q.section != null ? q.section : "occ");
    var ps = PROFILE.sections[s] || (PROFILE.sections[s] = { seen: 0, wrong: 0 });
    ps.seen++; if (!ok) ps.wrong++;
    if (q.topic) { var pt = PROFILE.topics[q.topic] || (PROFILE.topics[q.topic] = { seen: 0, wrong: 0 }); pt.seen++; if (!ok) pt.wrong++; }
    PROFILE.answered = (PROFILE.answered || 0) + 1; if (ok) PROFILE.correct = (PROFILE.correct || 0) + 1;
    saveProfile();
  }
  function weakSummary() {
    var secs = Object.keys(PROFILE.sections).map(function (s) {
      var p = PROFILE.sections[s];
      return { name: (s === "occ" ? "Occupational/Safety" : (SECTION_NAMES[s] || ("Section " + s))), wrong: p.wrong, seen: p.seen };
    }).filter(function (x) { return x.wrong > 0; }).sort(function (a, b) { return b.wrong - a.wrong; });
    var tops = Object.keys(PROFILE.topics).map(function (t) {
      var p = PROFILE.topics[t]; return { name: t, wrong: p.wrong, seen: p.seen };
    }).filter(function (x) { return x.wrong > 0; }).sort(function (a, b) { return b.wrong - a.wrong; }).slice(0, 8);
    return { secs: secs.slice(0, 5), tops: tops };
  }
  function tutorSystem() {
    var w = weakSummary();
    if (!w.secs.length && !w.tops.length) return TUTOR_SYSTEM;
    var prof = "\n\nSTUDENT PROFILE (use it to personalize — focus on these weak spots, drill them, and check the student really learned): weakest sections: " +
      w.secs.map(function (x) { return x.name + " (" + x.wrong + "/" + x.seen + " wrong)"; }).join(", ") +
      (w.tops.length ? ". Weak topics: " + w.tops.map(function (x) { return x.name; }).join(", ") : "") + ".";
    return TUTOR_SYSTEM + prof;
  }

  // ---------- quiz engine ----------
  var Q = { list: [], i: 0, mode: "", answers: [], totalTimer: null, endAt: 0, budget: 0, startedAt: 0, qStartAt: 0 };

  function pickWeighted(n) {
    // group by block, sample proportional to blueprint block_weights
    var w = BP.block_weights || { A: 11, B: 29, C: 30, D: 20, E: 10 };
    var byBlock = {};
    ALL.forEach(function (q) { var b = q.block || "B"; (byBlock[b] = byBlock[b] || []).push(q); });
    var out = [];
    Object.keys(w).forEach(function (b) {
      var want = Math.round(n * w[b] / 100);
      var pool = shuffle(byBlock[b] || []);
      out = out.concat(pool.slice(0, want));
    });
    // top up / trim to n
    if (out.length < n) {
      var rest = shuffle(ALL.filter(function (q) { return out.indexOf(q) < 0; }));
      out = out.concat(rest.slice(0, n - out.length));
    }
    return shuffle(out).slice(0, n);
  }

  function fmt(ms) { if (ms < 0) ms = 0; var m = Math.floor(ms / 60000), s = Math.floor((ms % 60000) / 1000); return m + ":" + (s < 10 ? "0" : "") + s; }

  function startQuiz(list, mode, timed) {
    list = (list || []).filter(function (q) { return (q.options || []).length >= 2 && q.answer_key; });
    if (!list.length) { alert("No answerable questions in that selection yet."); return; }
    if (Q.totalTimer) clearInterval(Q.totalTimer);
    Q = { list: list, i: 0, mode: mode, answers: list.map(function () { return null; }), flags: {},
      totalTimer: null, endAt: 0, budget: 0, startedAt: Date.now(), qStartAt: Date.now() };
    if (timed || mode === "mock") {
      var mins = list.length >= 100 ? (BP.time_minutes_standard || 240) : Math.max(1, Math.round(list.length * 1.2));
      Q.budget = mins * 60000; Q.endAt = Q.startedAt + Q.budget;
    }
    show("quiz");
    Q.totalTimer = setInterval(tickTimers, 500); tickTimers();
    renderQ();
  }
  function tickTimers() {
    var a = Q.answers[Q.i];
    var qms = a ? (a.timeMs || 0) : (Date.now() - Q.qStartAt);
    $("qtimer").textContent = "⏱ " + fmt(qms);
    var tt = $("ttimer");
    if (Q.budget) {
      var rem = Q.endAt - Date.now();
      tt.textContent = "🕒 " + fmt(rem) + " left";
      tt.classList.toggle("warn", rem < 120000);
      if (rem <= 0) { finish(); }
    } else {
      tt.textContent = "🕒 " + fmt(Date.now() - Q.startedAt);
    }
  }
  function liveScore() {
    var ans = Q.answers.filter(Boolean), ok = ans.filter(function (a) { return a.ok; }).length;
    $("qscore").textContent = "✓ " + ok + "/" + ans.length + " right";
  }

  function renderQ() {
    var q = Q.list[Q.i], a = Q.answers[Q.i];
    Q.qStartAt = Date.now();
    $("bar").style.width = (((Q.i + (a ? 1 : 0)) / Q.list.length) * 100) + "%";
    $("qcount").textContent = "Q " + (Q.i + 1) + " / " + Q.list.length;
    liveScore();
    var tags = $("qtags"); tags.innerHTML = "";
    tags.appendChild(el("span", "tag", esc(SECTION_NAMES[q.section] || ("Section " + (q.section || "?")))));
    if (q.block) tags.appendChild(el("span", "tag block", "Block " + esc(q.block)));
    if (q.topic) tags.appendChild(el("span", "tag", esc(q.topic)));
    $("qtext").textContent = q.question;
    var correctKey = (q.answer_key || "").toUpperCase();
    var opts = $("options"); opts.innerHTML = "";
    (q.options || []).forEach(function (optText, idx) {
      var keyLetter = ((optText.match(/^([A-Da-d])\)/) || [])[1] || String.fromCharCode(65 + idx)).toUpperCase();
      var label = optText.replace(/^([A-Da-d])\)\s*/, "");
      var b = el("button", "opt");
      b.appendChild(el("span", "k", esc(keyLetter)));
      b.appendChild(el("span", null, esc(label)));
      b.dataset.key = keyLetter;
      if (a) {
        b.disabled = true;
        if (keyLetter === correctKey) b.classList.add("correct");
        else if (keyLetter === a.picked) b.classList.add("wrong");
        else b.classList.add("dim");
      } else {
        b.onclick = function () { choose(keyLetter); };
      }
      opts.appendChild(b);
    });
    if (a) { showBanner(a.ok, correctKey); showExplain(q); $("teach").classList.remove("hidden"); }
    else { $("banner").classList.add("hidden"); $("explain").classList.add("hidden"); $("explain").innerHTML = ""; $("teach").classList.add("hidden"); }
    $("prev").disabled = (Q.i === 0);
    $("next").disabled = (Q.i >= Q.list.length - 1);
    updateFlagBtn();
  }
  function updateFlagBtn() { var b = $("flag"); if (!b) return; var on = !!Q.flags[Q.i]; b.classList.toggle("on", on); b.textContent = on ? "🚩 Flagged" : "🚩 Flag"; }
  function toggleFlag() { if (Q.flags[Q.i]) delete Q.flags[Q.i]; else Q.flags[Q.i] = true; updateFlagBtn(); }

  function showBanner(ok, correctKey) {
    var b = $("banner");
    b.className = "banner " + (ok ? "good" : "bad");
    b.innerHTML = ok ? "✅ <b>Correct!</b> — درست بود! 🎉"
      : "❌ <b>Incorrect</b> — the right answer is <b>" + esc(correctKey) + "</b>. اشکالی نداره، با هم یاد می‌گیریم 👇";
    b.classList.remove("hidden");
  }

  function choose(key) {
    if (Q.answers[Q.i]) return;
    var q = Q.list[Q.i];
    var correctKey = (q.answer_key || "").toUpperCase();
    var ok = key === correctKey;
    Q.answers[Q.i] = { picked: key, ok: ok, timeMs: Date.now() - Q.qStartAt };
    record(q.id, ok); profileRecord(q, ok);
    var btns = $("options").children;
    for (var i = 0; i < btns.length; i++) {
      var btn = btns[i]; btn.disabled = true; btn.onclick = null;
      if (btn.dataset.key === correctKey) btn.classList.add("correct");
      else if (btn.dataset.key === key) btn.classList.add("wrong");
      else btn.classList.add("dim");
    }
    showBanner(ok, correctKey); showExplain(q); liveScore();
    $("teach").classList.remove("hidden");
    $("bar").style.width = (((Q.i + 1) / Q.list.length) * 100) + "%";
  }

  function showExplain(q) {
    var e = $("explain");
    var refs = (q.references || []).join(" · ");
    var html = "<h4>✅ Answer</h4><p>" + esc(q.answer || ("Option " + (q.answer_key || ""))) + "</p>";
    if (q.solution_steps && q.solution_steps.length) {
      html += "<h4>🧭 Fastest path (English)</h4><ol>";
      q.solution_steps.forEach(function (s) { html += "<li>" + esc(s) + "</li>"; });
      html += "</ol>";
    }
    if (refs) html += "<div class='refs'>📖 " + esc(refs) + "</div>";
    html += "<div id='teachout'></div>";
    e.innerHTML = html;
    e.classList.remove("hidden");
  }

  var teachConvo = [];
  function teachThis() {
    var q = Q.list[Q.i], a = Q.answers[Q.i];
    if (!getKey()) { gotoSettings("Add your API key to get the bilingual teacher explanation."); return; }
    var host = $("teachout"); if (!host) return;
    teachConvo = [{ role: "user", content:
      "Teach me this exam question like a patient teacher to a beginner, fully in BOTH English AND Persian. " +
      "Name the keyword(s), which Table/Rule and why, then the fastest step-by-step path. Use plain readable math (NO LaTeX). " +
      "At the very END, ask me ONE short multiple-choice check question to confirm I understood — give 4 options labelled A) B) C) D) — and tell me to reply with the letter. Then wait for my answer; when I answer, tell me if I'm right and why.\n\n" +
      "Question: " + q.question + "\n" + (q.options || []).join("\n") +
      "\nCorrect answer: " + (q.answer_key || "") + " — " + (q.answer || "") +
      (a ? ("\nI answered: " + a.picked + " (" + (a.ok ? "correct" : "incorrect") + ").") : "") }];
    host.innerHTML =
      "<div id='teachthread' class='teachbox'></div>" +
      "<div class='composer teachcomposer'>" +
      "<textarea id='teachinput' rows='1' placeholder='جواب چک‌سؤال را بنویس (مثلاً B) یا هر سؤال دیگری بپرس…'></textarea>" +
      "<button id='teachsend' class='primary'>Send</button></div>";
    $("teachsend").onclick = teachReply;
    $("teachinput").addEventListener("keydown", function (e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); teachReply(); } });
    runTeach();
  }
  function renderTeachThread() {
    var thr = $("teachthread"); if (!thr) return;
    var html = "";
    teachConvo.forEach(function (m, i) {
      if (i === 0) return; // hide the long initial instruction
      if (m.role === "user") html += "<div class='tmsg me'>" + esc(m.content) + "</div>";
      else html += "<div class='tmsg ai rich'>" + mdToHtml(m.content) + "</div>";
    });
    thr.innerHTML = html;
  }
  function runTeach() {
    var thr = $("teachthread"); if (!thr) return;
    renderTeachThread();
    var bubble = el("div", "tmsg ai rich");
    var t0 = Date.now();
    bubble.innerHTML = "<p class='waiting'>⏳ Thinking…</p>";
    thr.appendChild(bubble); thr.scrollTop = thr.scrollHeight;
    var iv = setInterval(function () { if (bubble) bubble.innerHTML = "<p class='waiting'>⏳ Thinking… " + Math.round((Date.now() - t0) / 1000) + "s</p>"; }, 1000);
    var started = false;
    callClaude(teachConvo, { system: tutorSystem(), maxTokens: 3500, onText: function (t) {
      if (!started) { started = true; clearInterval(iv); }
      bubble.innerHTML = mdToHtml(t); thr.scrollTop = thr.scrollHeight;
    } }).then(function (full) {
      clearInterval(iv); teachConvo.push({ role: "assistant", content: full });
      bubble.innerHTML = mdToHtml(full);
      var inp = $("teachinput"); if (inp) inp.focus();
    }).catch(function (err) { clearInterval(iv); bubble.className = "status err"; bubble.textContent = "⚠️ " + apiErr(err); });
  }
  function teachReply() {
    var inp = $("teachinput"); if (!inp) return;
    var txt = inp.value.trim(); if (!txt) return;
    inp.value = "";
    teachConvo.push({ role: "user", content: txt });
    runTeach();
  }

  function gotoQ(i) { if (i < 0 || i >= Q.list.length) return; Q.i = i; renderQ(); window.scrollTo(0, 0); }
  function nextQ() { if (Q.i < Q.list.length - 1) gotoQ(Q.i + 1); }
  function prevQ() { if (Q.i > 0) gotoQ(Q.i - 1); }

  function finish() {
    if (Q.totalTimer) { clearInterval(Q.totalTimer); Q.totalTimer = null; }
    var answered = Q.answers.filter(Boolean);
    var correct = answered.filter(function (a) { return a.ok; }).length;
    PROFILE.exams = (PROFILE.exams || 0) + 1; saveProfile();
    show("results");
    var pct = Q.list.length ? Math.round(correct / Q.list.length * 100) : 0;
    var pass = pct >= (BP.pass_mark_percent || 70);
    var totalMs = answered.reduce(function (s, a) { return s + (a.timeMs || 0); }, 0);
    $("score").innerHTML = pct + "%<small>" + correct + " / " + Q.list.length + " correct · " +
      "answered " + answered.length + " · avg " + (answered.length ? Math.round(totalMs / answered.length / 1000) : 0) + "s/Q · pass mark " + (BP.pass_mark_percent || 70) + "%</small>";
    var d = $("rdetail");
    var h = "<p class='" + (pass ? "pass" : "fail") + "'>" + (pass ? "✅ PASS — keep this pace!" : "❌ Below 70% — let's drill the misses below.") + "</p>";
    // weak sections this session
    var bySec = {};
    Q.list.forEach(function (q, i) { var a = Q.answers[i]; if (a && !a.ok) { var s = SECTION_NAMES[q.section] || ("Sec " + q.section); bySec[s] = (bySec[s] || 0) + 1; } });
    var secList = Object.keys(bySec).sort(function (a, b) { return bySec[b] - bySec[a]; });
    if (secList.length) h += "<p><b>Weakest areas this session:</b> " + secList.map(function (s) { return esc(s) + " (" + bySec[s] + ")"; }).join(" · ") + "</p>";
    // missed questions with full explanation
    var missed = [];
    Q.list.forEach(function (q, i) { var a = Q.answers[i]; if (a && !a.ok) missed.push({ q: q, a: a }); });
    if (missed.length) {
      h += "<h3 style='color:#fff'>📚 Review your misses (" + missed.length + ")</h3>";
      missed.forEach(function (m) {
        var q = m.q;
        h += "<div class='miss'>";
        h += "<div class='missq'>" + esc(q.question) + "</div>";
        h += "<div class='missline'>You: <b class='bad'>" + esc(m.a.picked) + "</b> · Correct: <b class='ok'>" + esc((q.answer_key || "")) + " — " + esc(q.answer || "") + "</b></div>";
        if (q.solution_steps && q.solution_steps.length) {
          h += "<ol>"; q.solution_steps.forEach(function (s) { h += "<li>" + esc(s) + "</li>"; }); h += "</ol>";
        }
        if ((q.references || []).length) h += "<div class='refs'>📖 " + esc(q.references.join(" · ")) + "</div>";
        h += "</div>";
      });
    } else if (answered.length) {
      h += "<p class='pass'>No misses — excellent! 🌟</p>";
    }
    var flagged = [];
    Q.list.forEach(function (q, i) { if (Q.flags[i]) flagged.push(q); });
    if (flagged.length) {
      h += "<h3 style='color:#fff'>🚩 You flagged (" + flagged.length + ")</h3>";
      flagged.forEach(function (q) {
        h += "<div class='miss'><div class='missq'>" + esc(q.question) + "</div>" +
          "<div class='missline'>Correct: <b class='ok'>" + esc((q.answer_key || "") + " — " + (q.answer || "")) + "</b></div>" +
          ((q.references || []).length ? "<div class='refs'>📖 " + esc(q.references.join(" · ")) + "</div>" : "") + "</div>";
      });
    }
    d.innerHTML = h;
    // buttons
    var ra = $("resultActions"); ra.innerHTML = "";
    if (missed.length) {
      var drill = el("button", "primary", "🎯 Drill these " + missed.length + " again");
      drill.onclick = function () { startQuiz(shuffle(missed.map(function (m) { return m.q; })), "weak", false); };
      ra.appendChild(drill);
    }
    var home = el("button", "ghost", "Home"); home.setAttribute("data-home", "1"); ra.appendChild(home);
  }

  // ---------- section picker ----------
  function buildPicker() {
    var counts = {};
    ALL.forEach(function (q) { var s = (q.section != null ? q.section : "occ"); counts[s] = (counts[s] || 0) + 1; });
    var list = $("sectionList"); list.innerHTML = "";
    Object.keys(counts).sort(function (a, b) { return (a === "occ" ? 999 : +a) - (b === "occ" ? 999 : +b); }).forEach(function (s) {
      var name = s === "occ" ? "Occupational / Safety Skills" : (SECTION_NAMES[s] || ("Section " + s));
      var item = el("button", "secitem");
      item.appendChild(el("b", null, esc(name)));
      item.appendChild(el("span", "cnt", counts[s] + " questions"));
      item.onclick = function () {
        var pool = ALL.filter(function (q) { return (q.section != null ? q.section : "occ") == s; });
        startQuiz(shuffle(pool), "section", false);
      };
      list.appendChild(item);
    });
  }

  // ---------- AI tutor ----------
  var chatHistory = [];
  function addMsg(role, text) {
    var m = el("div", "msg " + (role === "user" ? "me" : "ai"));
    m.textContent = text;
    $("chat").appendChild(m);
    $("chat").scrollTop = $("chat").scrollHeight;
    return m;
  }
  function sendChat(prefill) {
    var input = $("chatinput");
    var text = prefill || input.value.trim();
    if (!text) return;
    if (!getKey()) { gotoSettings("Add your API key first to use the AI tutor."); return; }
    input.value = "";
    addMsg("user", text);
    chatHistory.push({ role: "user", content: text });
    var bubble = addMsg("assistant", "");
    var wait = thinking(bubble, "msg ai");
    var started = false;
    function paint(t) { if (!started) { started = true; wait.stop(); } bubble.className = "msg ai rich"; bubble.innerHTML = mdToHtml(t); $("chat").scrollTop = $("chat").scrollHeight; }
    callClaude(chatHistory, {
      system: tutorSystem(), maxTokens: 4096, onText: paint
    }).then(function (full) {
      wait.stop(); bubble.className = "msg ai rich"; bubble.innerHTML = mdToHtml(full);
      chatHistory.push({ role: "assistant", content: full });
      $("chat").scrollTop = $("chat").scrollHeight;
    }).catch(function (err) { wait.stop(); bubble.className = "msg ai"; bubble.textContent = "⚠️ " + apiErr(err); });
  }

  // ---------- generate questions ----------
  function seedExamples(secNum) {
    var pool = ALL.filter(function (q) { return (q.options || []).length >= 4 && q.answer_key && !q.needs_review; });
    if (secNum != null) { var sp = pool.filter(function (q) { return (q.section != null ? q.section : "occ") == secNum; }); if (sp.length) pool = sp; }
    return shuffle(pool).slice(0, 3).map(function (q) {
      return { section: q.section, topic: q.topic, question: q.question, options: q.options, answer: q.answer, answer_key: q.answer_key, references: q.references, solution_steps: q.solution_steps };
    });
  }
  function doGenerate(topic, count, secNum, statusEl, btn) {
    if (!getKey()) { gotoSettings("Add your API key first to generate questions."); return; }
    statusEl.className = "status"; statusEl.textContent = "Generating " + count + " questions…";
    if (btn) btn.disabled = true;
    var seeds = seedExamples(secNum);
    var prompt =
      "You are writing practice questions for the BC Construction Electrician (Red Seal) exam on CEC 2024.\n" +
      "STUDY these REAL exam questions the student gave me — learn their exact STYLE, difficulty, phrasing, the way options are written, and the reference/solution format:\n" +
      JSON.stringify(seeds, null, 1) + "\n\n" +
      "Now WRITE " + count + " BRAND-NEW questions (do NOT copy the examples) on: " + topic + ".\n" +
      "Match that style and difficulty exactly. Return ONLY a JSON array, no prose. Each item:\n" +
      '{"id":"GEN-<n>","section":<number>,"block":"A|B|C|D|E","topic":"short","question":"...",' +
      '"options":["A) ...","B) ...","C) ...","D) ..."],"answer":"correct answer text","answer_key":"A|B|C|D",' +
      '"references":["Rule/Table"],"solution_steps":["name the keyword + which Table/Rule to jump to","then the steps to the answer"]}\n' +
      "Vary the correct letter. Be accurate with CEC 2024 references. Use plain text (no LaTeX).";
    callClaude([{ role: "user", content: prompt }], { maxTokens: 4096 })
      .then(function (full) {
        var arr = parseJSONArray(full);
        if (!arr || !arr.length) throw new Error("Could not parse generated questions.");
        arr.forEach(function (q, i) { q.id = q.id || ("GEN-" + Date.now() + "-" + i); q._generated = true; if (secNum != null && q.section == null) q.section = secNum; });
        statusEl.className = "status ok"; statusEl.textContent = "Made " + arr.length + " questions. Starting…";
        if (btn) btn.disabled = false;
        setTimeout(function () { startQuiz(arr, "generated", false); }, 400);
      })
      .catch(function (err) { statusEl.className = "status err"; statusEl.textContent = "⚠️ " + apiErr(err); if (btn) btn.disabled = false; });
  }
  function generate() {
    doGenerate($("genTopic").value.trim() || "mixed CEC 2024 topics", $("genCount").value, null, $("genStatus"), $("genGo"));
  }
  function lessonGen() {
    var s = currentLesson; if (!s) return;
    if (!$("lessonGenStatus")) { var d = el("div", "status"); d.id = "lessonGenStatus"; $("lessonTeachOut").appendChild(d); }
    doGenerate("CEC 2024 " + secLabel(s).replace(/—.*$/, "").trim() + " (" + (s.name || "") + ")", 5, s.section, $("lessonGenStatus"), $("lessonGen"));
  }
  function parseJSONArray(s) {
    var a = s.indexOf("["), b = s.lastIndexOf("]");
    if (a < 0 || b < 0) return null;
    try { return JSON.parse(s.slice(a, b + 1)); } catch (e) { return null; }
  }

  // ---------- vision: pdf / image / camera ----------
  var attached = null; // {type:'image'|'document', media, data, name}
  function readFile(file) {
    return new Promise(function (resolve, reject) {
      var r = new FileReader();
      r.onload = function () { resolve(r.result); };
      r.onerror = reject;
      r.readAsDataURL(file);
    });
  }
  function handleFile(file) {
    if (!file) return;
    readFile(file).then(function (dataUrl) {
      var comma = dataUrl.indexOf(",");
      var meta = dataUrl.slice(5, comma); // e.g. image/jpeg;base64
      var media = meta.split(";")[0];
      var b64 = dataUrl.slice(comma + 1);
      var isPdf = media === "application/pdf";
      attached = { type: isPdf ? "document" : "image", media: media, data: b64, name: file.name };
      var p = $("attachPreview"); p.innerHTML = "";
      if (isPdf) p.appendChild(el("span", "pill", "📄 " + esc(file.name)));
      else { var img = new Image(); img.src = dataUrl; p.appendChild(img); }
    });
  }
  function askVision() {
    var qtext = $("visionQ").value.trim() || "Explain this and how it relates to the CEC 2024 exam.";
    if (!getKey()) { gotoSettings("Add your API key first to use vision."); return; }
    if (!attached) { alert("Attach an image or PDF first."); return; }
    var out = $("visionOut"); out.classList.remove("hidden");
    var wait = thinking(out, "explain rich");
    var started = false;
    function paint(t) { if (!started) { started = true; wait.stop(); } out.className = "explain rich"; out.innerHTML = mdToHtml(t); }
    var block = attached.type === "document"
      ? { type: "document", source: { type: "base64", media_type: "application/pdf", data: attached.data } }
      : { type: "image", source: { type: "base64", media_type: attached.media, data: attached.data } };
    var content = [block, { type: "text", text: qtext }];
    callClaude([{ role: "user", content: content }], {
      system: tutorSystem(), maxTokens: 4096, onText: paint
    }).then(function (full) { wait.stop(); out.className = "explain rich"; out.innerHTML = mdToHtml(full); })
      .catch(function (err) { wait.stop(); out.className = "status err"; out.textContent = "⚠️ " + apiErr(err); });
  }

  function apiErr(err) {
    if (err && err.message === "NO_KEY") return "No API key set. Open ⚙️ Settings.";
    return (err && err.message) ? err.message : "Request failed.";
  }

  // ---------- keyword → table map ----------
  var KEYWORD_MAP = [
    { sec: "Section 4 — Conductors / Ampacity", rows: [
      ["\"ampacity\" / \"max current a conductor can carry\"", "Tables 1–4", "base ampacity"],
      ["\"in free air\" (Cu / Al)", "Table 1 (Cu) / Table 3 (Al)", ""],
      ["\"in a raceway / conduit / cable\" (Cu / Al)", "Table 2 (Cu) / Table 4 (Al)", "start in the 90 °C column"],
      ["\"more than 3 conductors\" / \"X conductors\"", "Table 5C", "grouping derate (4–6 = 0.80, 7–24 = 0.70)"],
      ["\"ambient … °C\" (above 30 °C)", "Table 5A", "temperature correction"],
      ["\"terminated at / terminal marked X °C\"", "Rule 4-006", "use the X °C column"],
      ["\"continuous load\"", "Rule 8-104", "size at 125 %"],
      ["\"neutral-supported / NS / overhead service\"", "Table 36B (Cu) / 36A (Al)", ""],
      ["\"flexible cord / equipment wire / portable\"", "Table 12", ""],
      ["\"underground, direct-buried, spaced\"", "Rule 4-004 1)d) + Diagrams D8/D10/D11", ""]
    ]},
    { sec: "Section 8 — Loads & Demand", rows: [
      ["\"single dwelling … demand/service\"", "Rule 8-200", ""],
      ["\"apartment / multiple dwelling\"", "Rule 8-202", ""],
      ["\"basic load W/m² by occupancy\"", "Table 14", ""],
      ["\"show window\"", "650 W/m", ""],
      ["\"electric range\"", "Rule 8-200 (service = 6 kW) / 8-300 (branch = 8 kW)", ""],
      ["\"voltage drop\"", "Rule 8-102 + Table D3", "3 % / 3 % / 5 %"],
      ["\"how many receptacles on a circuit\"", "max 12 on a 15 A circuit", ""],
      ["\"parking / EV charger load\"", "Rule 8-400", ""]
    ]},
    { sec: "Section 10 — Grounding & Bonding", rows: [
      ["\"grounding electrode conductor / size of grounding conductor\"", "Table 43", "keyed to service-conductor ampacity"],
      ["\"bonding conductor / bonding jumper\"", "Table 16", "ampacity OR overcurrent device"],
      ["\"field-assembled grounding electrode\"", "Rule 10-102 / 10-104", ""],
      ["\"equipotential bonding\"", "Rule 10-406", "#6 Cu / #4 Al"]
    ]},
    { sec: "Section 12 — Wiring Methods", rows: [
      ["\"box fill / conductors in a box\"", "Rule 12-3036 + Table 23", ""],
      ["\"conduit fill / minimum conduit size\"", "Rule 12-910 + Tables 6/8/9", "1→53 %, 2→31 %, 3+→40 %"],
      ["\"bend radius\"", "Table 7", ""],
      ["\"number of bends\"", "Rule 12-936", "max 360°"],
      ["\"support / secured / spacing\"", "Rule 12-510/560; Table 21", "300 mm then ≤1.5 m"],
      ["\"minimum cover / buried depth\"", "Table 53", ""],
      ["\"NMD90 / NMSC distance from stud\"", "32 mm (Rule 12-516)", ""],
      ["\"cable type selection / dry-damp-wet\"", "Table 19 + Table D1", ""]
    ]},
    { sec: "Section 14 — Protection", rows: [
      ["\"overcurrent / breaker or fuse size for a conductor\"", "Rule 14-104 + Table 13", ""],
      ["\"is ground-fault protection required?\"", "Rule 14-102", "only if ≥1000 A AND >150 V to ground"]
    ]},
    { sec: "Section 26 — Equipment", rows: [
      ["\"receptacle spacing in a dwelling\"", "Rule 26-712", ""],
      ["\"GFCI required?\"", "Rule 26-700 area", ""],
      ["\"transformer overcurrent / primary fuse\"", "Rule 26-250 / 26-254", ""],
      ["\"capacitor conductor size\"", "Rule 26-210", "≥135 %"],
      ["\"panelboard handle height\"", "1.7 m max", ""]
    ]},
    { sec: "Section 28 — Motors", rows: [
      ["\"motor full-load current / FLC\"", "Table 44 (3-phase) / 45 (1-phase)", "not the nameplate"],
      ["\"motor branch conductor size\"", "Rule 28-106", "125 % of FLC"],
      ["\"overload size\"", "Rule 28-306", "nameplate FLA × 1.25 (or 1.15)"],
      ["\"max fuse/breaker for a motor branch\"", "Table 29", "175 / 225 / 300 / 1300 %"],
      ["\"disconnect distance\"", "within 9 m / in sight", ""]
    ]},
    { sec: "Section 2 / 0 — General & Definitions", rows: [
      ["\"working space / clearance in front of\"", "Rule 2-308 + Table 56", "Table 56 uses line-to-ground voltage"],
      ["\"voltage to ground in a dwelling\"", "Rule 2-110", "150 V"],
      ["\"enclosure type (3R, 4X…)\"", "Table 65", ""],
      ["\"deviation / special permission\"", "Rule 2-030", ""]
    ]}
  ];

  function hl(text, q) {
    var s = esc(text);
    if (!q) return s;
    var re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
    return s.replace(re, "<mark>$1</mark>");
  }
  // ===== Study program + comprehensive keyword map (data-driven from study.js) =====
  var STUDY = (window.CEC_STUDY && window.CEC_STUDY.sections) || [];
  function getLang() { return localStorage.getItem("cec_lang") || "en"; }
  function setLang(l) { localStorage.setItem("cec_lang", l); }
  function pick(o, base) { if (getLang() === "fa") { var v = o[base + "_fa"]; if (v !== undefined && v !== "") return v; } return o[base] || ""; }
  function pickArr(o, base) { if (getLang() === "fa") { var v = o[base + "_fa"]; if (Array.isArray(v) && v.length) return v; } return o[base] || []; }
  function langBar(hostId, rerender) {
    var host = $(hostId); if (!host) return;
    var l = getLang();
    host.innerHTML = "<button class='" + (l === "en" ? "on" : "") + "' data-setlang='en'>EN</button>" +
      "<button class='" + (l === "fa" ? "on" : "") + "' data-setlang='fa'>فارسی</button>";
    var bs = host.querySelectorAll("[data-setlang]");
    for (var i = 0; i < bs.length; i++) (function (b) { b.onclick = function () { setLang(b.getAttribute("data-setlang")); rerender(); }; })(bs[i]);
  }
  function stars(n) { n = Math.max(0, Math.min(5, n || 0)); return "<span class='stars' title='" + n + "/5'>" + "★★★★★".slice(0, n) + "<span class='stardim'>" + "★★★★★".slice(n) + "</span></span>"; }
  function secLabel(s) {
    var head = s.section === "occ" ? "Occupational / Safety" : (s.section === 0 ? "Section 0" : "Section " + s.section);
    var nm = (s.name || "").replace(/^Section\s+\d+\s*[—-]\s*/, "");
    return head + (nm ? " — " + nm : "");
  }

  function keymapGroups() {
    return STUDY.slice().sort(function (a, b) { return (b.importance || 0) - (a.importance || 0); })
      .map(function (s) {
        var rows = (s.keywords || []).slice().sort(function (a, b) { return (b.stars || 0) - (a.stars || 0); });
        return { sec: secLabel(s), rows: rows };
      }).filter(function (g) { return g.rows.length; });
  }
  function renderKeymap(q) {
    q = (q || "").trim().toLowerCase();
    var host = $("kmList"); host.innerHTML = ""; var any = false;
    keymapGroups().forEach(function (group) {
      var rows = group.rows.filter(function (r) {
        if (!q) return true;
        return ((r.kw || "") + " " + (r.jump || "") + " " + (r.why || "") + " " + group.sec).toLowerCase().indexOf(q) >= 0;
      });
      if (!rows.length) return;
      any = true;
      var box = el("div", "kmsec");
      box.appendChild(el("h3", null, esc(group.sec) + " <span class='pill'>" + rows.length + "</span>"));
      rows.forEach(function (r) {
        var row = el("div", "kmrow");
        row.innerHTML =
          "<div class='kmkw'>" + hl(r.kw || "", q) + "</div>" +
          "<div class='kmjump'><span class='kmlbl'>Jump to</span><span class='arrow'>→</span> <b>" + hl(r.jump || "", q) + "</b> " + pageChip(r.jump || "") + "</div>" +
          "<div class='kmwhy'>" + (r.why ? "<span class='kmlbl'>Why</span>" + hl(r.why, q) + " " : "") + stars(r.stars) + "</div>";
        box.appendChild(row);
      });
      host.appendChild(box);
    });
    $("kmEmpty").classList.toggle("hidden", any);
  }

  function buildStudyList() {
    langBar("studyLang", buildStudyList);
    var fa = getLang() === "fa";
    var host = $("studyList"); host.innerHTML = ""; host.dir = fa ? "rtl" : "ltr";
    STUDY.slice().sort(function (a, b) { return (b.importance || 0) - (a.importance || 0); }).forEach(function (s) {
      var card = el("button", "studycard");
      card.innerHTML =
        "<div class='sctop'><b>" + esc(secLabel(s)) + "</b>" + stars(s.importance) + "</div>" +
        "<div class='scsum'>" + esc(pick(s, "summary")) + "</div>" +
        "<div class='scmeta'>Block " + esc(s.block || "?") + " · " + (s.qcount || 0) + (fa ? " سؤال" : " questions") + (secPage(s.section) ? " · 📖 p." + secPage(s.section) : "") + "</div>";
      card.onclick = function () { renderLesson(s); };
      host.appendChild(card);
    });
  }

  var currentLesson = null;
  function ltable(items, refLbl, col2, col3, base2, base3) {
    var h = "<div class='ltable'>";
    items.forEach(function (it) {
      h += "<div class='ltr'>" +
        "<div class='ltc ref'><span class='ltlbl'>" + refLbl + "</span><b>" + esc(it.ref) + "</b> " + stars(it.stars) + " " + pageChip(it.ref) + "</div>" +
        "<div class='ltc'><span class='ltlbl'>" + col2 + "</span>" + esc(pick(it, base2)) + "</div>" +
        "<div class='ltc'><span class='ltlbl'>" + col3 + "</span>" + esc(pick(it, base3)) + "</div>" +
        "</div>";
    });
    return h + "</div>";
  }
  function renderLesson(s) {
    currentLesson = s;
    langBar("lessonLang", function () { renderLesson(currentLesson); });
    var fa = getLang() === "fa";
    var T = fa ? { tables: "📊 جدول‌های کلیدی — مهم‌ترین اول", rules: "📏 قوانین کلیدی", method: "🧭 روشِ سریع", must: "⭐ مقادیرِ حفظی", traps: "⚠️ تله‌های رایج", memo: "ترفندِ حافظه", ex: "📝 مثال‌های حل‌شده", tbl: "جدول", rule: "قانون", what: "چیست", how: "چطور/چرا", note: "نکته", inbank: " سؤال در بانک" }
      : { tables: "📊 Key tables — most important first", rules: "📏 Key rules", method: "🧭 The fast method", must: "⭐ Must-know values", traps: "⚠️ Common traps", memo: "Memory aid", ex: "📝 Worked examples", tbl: "Table", rule: "Rule", what: "What", how: "How / why", note: "Note", inbank: " questions in the bank" };
    var h = "<h2 style='margin-top:0'>" + esc(secLabel(s)) + " " + stars(s.importance) + "</h2>";
    h += "<p class='lsum'>" + esc(pick(s, "summary")) + "</p>";
    var pg = secPage(s.section);
    h += "<div class='lmeta'>Block " + esc(s.block || "?") + " · " + (s.qcount || 0) + esc(T.inbank) +
      (pg ? " · 📖 " + (fa ? "این Section از صفحهٔ " : "starts on p. ") + "<b>" + pg + "</b>" + (fa ? " کتاب شروع می‌شود" : "") : "") + "</div>";
    h += "<div class='lmeta'>📚 Tables → p." + BOOK_ANCHORS.tables + " · Appendix B → p." + BOOK_ANCHORS.appendixB + " · Index → p." + BOOK_ANCHORS.index + "</div>";
    if ((s.keyTables || []).length) { h += "<h3>" + T.tables + "</h3>" + ltable(s.keyTables, T.tbl, T.what, T.how, "what", "use"); }
    if ((s.keyRules || []).length) { h += "<h3>" + T.rules + "</h3>" + ltable(s.keyRules, T.rule, T.what, T.note, "what", "note"); }
    if ((s.method || []).length) { h += "<h3>" + T.method + "</h3><ol>"; pickArr(s, "method").forEach(function (m) { h += "<li>" + esc(m) + "</li>"; }); h += "</ol>"; }
    if ((s.mustKnow || []).length) { h += "<h3>" + T.must + "</h3><ul>"; pickArr(s, "mustKnow").forEach(function (m) { h += "<li>" + esc(m) + "</li>"; }); h += "</ul>"; }
    if ((s.traps || []).length) { h += "<h3>" + T.traps + "</h3><ul class='traps'>"; pickArr(s, "traps").forEach(function (m) { h += "<li>" + esc(m) + "</li>"; }); h += "</ul>"; }
    if (pick(s, "mnemonic")) h += "<div class='mnemo'>🧠 <b>" + T.memo + ":</b> " + esc(pick(s, "mnemonic")) + "</div>";
    var exQ = (s.examples || []).map(function (id) { return ALL.filter(function (q) { return q.id === id; })[0]; }).filter(Boolean).slice(0, 4);
    if (exQ.length) {
      h += "<h3>" + T.ex + "</h3>";
      exQ.forEach(function (q) {
        h += "<div class='exq'><div class='exqq'>" + esc(q.question) + "</div>";
        h += "<div class='exqa'>✅ " + esc((q.answer_key || "") + " — " + (q.answer || "")) + "</div>";
        if ((q.solution_steps || []).length) { h += "<ol>"; q.solution_steps.slice(0, 5).forEach(function (st) { h += "<li>" + esc(st) + "</li>"; }); h += "</ol>"; }
        if ((q.references || []).length) h += "<div class='refs'>📖 " + esc(q.references.join(" · ")) + "</div>";
        h += "</div>";
      });
    }
    var body = $("lessonBody"); body.innerHTML = h; body.dir = fa ? "rtl" : "ltr";
    $("lessonTeachOut").innerHTML = "";
    show("lesson");
  }
  function lessonDrill() {
    if (!currentLesson) return;
    var sec = currentLesson.section;
    startQuiz(shuffle(ALL.filter(function (q) { return (q.section != null ? q.section : "occ") == sec; })), "section", false);
  }
  function lessonTeach() {
    var s = currentLesson; if (!s) return;
    if (!getKey()) { gotoSettings("Add your API key to have the AI teach this section."); return; }
    var out = $("lessonTeachOut");
    var wait = thinking(out, "rich teachbox"); var started = false;
    function paint(t) { if (!started) { started = true; wait.stop(); } out.className = "rich teachbox"; out.innerHTML = mdToHtml(t); }
    var msg = "Give me a fast, high-yield lesson on CEC 2024 " + secLabel(s) + " for the BC Construction Electrician exam, in BOTH English and full Persian. " +
      "Cover the most-tested tables/rules, the fastest keyword→table method, the common traps, and one short worked example. Keep it tight and exam-ready.";
    callClaude([{ role: "user", content: msg }], { system: tutorSystem(), maxTokens: 4096, onText: paint })
      .then(function (full) { wait.stop(); out.className = "rich teachbox"; out.innerHTML = mdToHtml(full); })
      .catch(function (err) { wait.stop(); out.className = "status err"; out.textContent = "⚠️ " + apiErr(err); });
  }

  // ===== page numbers from the real CEC 2024 book (section starts are exact) =====
  var SECTION_PAGES = { 0: 53, 2: 68, 4: 76, 6: 83, 8: 90, 10: 98, 12: 108, 14: 157, 16: 166, 26: 211, 28: 232 };
  var BOOK_ANCHORS = { tables: 393, diagrams: 515, appendixB: 556, index: 930 };
  function secPage(sec) { return SECTION_PAGES[sec]; }
  var PAGES = JSON.parse(localStorage.getItem("cec_pagenums") || "{}"); // exact ref -> page (user-filled)
  function savePages() { localStorage.setItem("cec_pagenums", JSON.stringify(PAGES)); }
  function pageChip(ref) {
    var v = PAGES[ref];
    return "<button class='pagechip" + (v ? " set" : "") + "' data-pageref='" + esc(ref) + "' title='page in your CEC 2024 book'>" + (v ? ("p." + esc(v)) : "+p") + "</button>";
  }
  document.addEventListener("click", function (ev) {
    var pc = ev.target.closest("[data-pageref]"); if (!pc) return;
    ev.stopPropagation();
    var ref = pc.getAttribute("data-pageref");
    var v = prompt("Page number in your CEC 2024 book for:\n" + ref, PAGES[ref] || "");
    if (v === null) return; v = v.trim();
    if (v) PAGES[ref] = v; else delete PAGES[ref];
    savePages();
    if (!$("keymap").classList.contains("hidden")) renderKeymap($("kmSearch").value);
    else if (!$("lesson").classList.contains("hidden") && currentLesson) renderLesson(currentLesson);
    else if (!$("codehunt").classList.contains("hidden")) chShowAns();
  });

  // ===== Study Plan: 6 weeks x 3h/day, live timer with pause/resume =====
  var PLAN_DAY = 180, PLAN_DAYS_WK = 6, PLAN_WKS = 6, PLAN_TOTAL = PLAN_DAY * PLAN_DAYS_WK * PLAN_WKS;
  var PLAN_CUR = [
    { title: "Week 1 — Foundations + Section 4 (ampacity)", secs: ["0", "2", "4"] },
    { title: "Week 2 — Section 8 (loads & demand)", secs: ["8"] },
    { title: "Week 3 — Section 10 (grounding & bonding)", secs: ["10"] },
    { title: "Week 4 — Section 12 (wiring methods, part 1)", secs: ["12"] },
    { title: "Week 5 — Section 12 part 2 + 14 + 6 + 26 + 16", secs: ["12", "14", "6", "26", "16"] },
    { title: "Week 6 — Section 28 (motors) + full mocks", secs: ["28"] }
  ];
  var PLAN_BLK = [
    { min: 25, label: "Code-hunt / keyword drills", fa: "تمرینِ Code-Hunt و کلیدواژه‌ها", go: "codehunt" },
    { min: 70, label: "Study the day's lesson(s)", fa: "خواندنِ درسِ امروز", go: "study" },
    { min: 55, label: "Timed practice quiz", fa: "کوییزِ تمرینیِ زمان‌دار", go: "practice" },
    { min: 30, label: "Review your misses", fa: "مرورِ غلط‌هایت", go: "weak" }
  ];
  function blockLabel(b) { return (getLang() === "fa" && b.fa) ? b.fa : b.label; }
  function planT() {
    return getLang() === "fa"
      ? { title: "📅 برنامهٔ درسی — ۶ هفته × ۳ ساعت/روز", today: "امروز", week: "این هفته", total: "کل", start: "▶ شروع مطالعه", pause: "⏸ توقف", reset: "ریستِ امروز", now: "همین الان", day: "روزِ", startTask: "این کار را شروع کن →", doneA: "🎉 برنامه تمام شد — ", doneB: " مطالعه ثبت شد. حالا Mockهای کاملِ زمان‌دار بزن.", secs: "بخش‌ها:" }
      : { title: "📅 Study Plan — 6 weeks · 3h/day", today: "Today", week: "This week", total: "Total", start: "▶ Start studying", pause: "⏸ Pause", reset: "Reset day", now: "Right now", day: "Day", startTask: "Start this task →", doneA: "🎉 Plan complete — ", doneB: " logged. Keep running full timed mocks.", secs: "Sections:" };
  }
  var TIME = JSON.parse(localStorage.getItem("cec_time") || "null") || { total: 0, days: {}, start: null, running: false };
  function saveTime() { localStorage.setItem("cec_time", JSON.stringify(TIME)); }
  function dkey(d) { d = d || new Date(); return d.getFullYear() + "-" + ("0" + (d.getMonth() + 1)).slice(-2) + "-" + ("0" + d.getDate()).slice(-2); }
  var _tick = null;
  function timeCredit() { if (!TIME.running || _tick == null) return; var now = Date.now(); var add = (now - _tick) / 60000; _tick = now; if (add <= 0) return; TIME.total += add; var k = dkey(); TIME.days[k] = (TIME.days[k] || 0) + add; saveTime(); }
  function timeStart() { if (TIME.running) return; if (!TIME.start) TIME.start = Date.now(); TIME.running = true; _tick = Date.now(); saveTime(); planUI(); }
  function timePause() { if (!TIME.running) return; timeCredit(); TIME.running = false; _tick = null; saveTime(); planUI(); }
  function timeHeartbeat() { if (TIME.running) { timeCredit(); planUI(); } }
  function fmtH(m) { return (m / 60).toFixed(1) + "h"; }
  function weekMin() { var s = 0; for (var i = 0; i <= PLAN_DAYS_WK; i++) { var d = new Date(); d.setDate(d.getDate() - i); s += TIME.days[dkey(d)] || 0; } return s; }
  function planSegs() { var segs = [], cum = 0; PLAN_CUR.forEach(function (wk, wi) { for (var d = 0; d < PLAN_DAYS_WK; d++) PLAN_BLK.forEach(function (b) { segs.push({ wi: wi, day: d + 1, b: b, start: cum }); cum += b.min; }); }); return segs; }
  function curSeg() { var s = planSegs(); for (var i = 0; i < s.length; i++) if (TIME.total < s[i].start + s[i].b.min) return s[i]; return null; }
  function planGoTask(go, wk) {
    if (go === "practice") { var pool = ALL.filter(function (q) { return wk.secs.indexOf(String(q.section != null ? q.section : "occ")) >= 0; }); startQuiz(shuffle(pool.length ? pool : ALL).slice(0, 25), "section", false); }
    else if (go === "study") { buildStudyList(); show("study"); }
    else if (go === "weak") { var w = ALL.filter(function (q) { return PROG[q.id] && PROG[q.id].wrong > 0; }); if (w.length) startQuiz(shuffle(w), "weak", false); else { buildStudyList(); show("study"); } }
    else if (go === "codehunt") { startCodeHunt(); show("codehunt"); }
  }
  function planUI() {
    if (!$("trkToday")) { var hud0 = $("hud"); if (hud0) hud0.innerHTML = TIME.running ? "<span class='studying'>⏺ studying</span>" : ""; return; }
    var T = planT();
    var today = TIME.days[dkey()] || 0, wk = weekMin(), total = TIME.total;
    $("trkToday").textContent = fmtH(today) + " / 3h";
    $("trkWeek").textContent = fmtH(wk) + " / 18h";
    $("trkTotal").textContent = fmtH(total) + " / 108h";
    $("barToday").style.width = Math.min(100, today / PLAN_DAY * 100) + "%";
    $("barWeek").style.width = Math.min(100, wk / (PLAN_DAY * PLAN_DAYS_WK) * 100) + "%";
    $("barTotal").style.width = Math.min(100, total / PLAN_TOTAL * 100) + "%";
    if ($("planTitle")) $("planTitle").textContent = T.title;
    var lbls = $("plan") ? $("plan").querySelectorAll(".tracklbl") : [];
    if (lbls.length === 3) { lbls[0].textContent = T.today; lbls[1].textContent = T.week; lbls[2].textContent = T.total; }
    var rb = $("timerReset"); if (rb) rb.textContent = T.reset;
    var bt = $("timerToggle"); if (bt) bt.textContent = TIME.running ? T.pause : T.start;
    var big = $("timerBig"); if (big) { var s = Math.floor(today * 60); big.textContent = Math.floor(s / 3600) + ":" + ("0" + Math.floor(s % 3600 / 60)).slice(-2) + ":" + ("0" + Math.floor(s % 60)).slice(-2); big.classList.toggle("live", TIME.running); }
    var hud = $("hud"); if (hud) hud.innerHTML = TIME.running ? "<span class='studying'>⏺ studying</span>" : "";
  }
  function renderPlan() {
    langBar("planLang", renderPlan);
    planUI();
    var T = planT();
    var seg = curSeg(), nt = $("nowTask");
    if (seg) {
      var wk = PLAN_CUR[seg.wi];
      nt.innerHTML = "<div class='ntlbl'>" + esc(T.now) + " · " + esc(wk.title) + " · " + esc(T.day) + " " + seg.day + "</div>" +
        "<div class='nttask'>" + esc(blockLabel(seg.b)) + " · " + seg.b.min + " min</div>" +
        "<button class='primary' id='nowGo'>" + esc(T.startTask) + "</button>";
      $("nowGo").onclick = function () { if (!TIME.running) timeStart(); planGoTask(seg.b.go, wk); };
    } else { nt.innerHTML = "<div class='nttask'>" + esc(T.doneA) + fmtH(TIME.total) + esc(T.doneB) + "</div>"; }
    var segs = planSegs(), pw = {};
    segs.forEach(function (s) { var p = pw[s.wi] = pw[s.wi] || { total: 0, start: 1e9, end: 0 }; p.total += s.b.min; p.start = Math.min(p.start, s.start); p.end = Math.max(p.end, s.start + s.b.min); });
    var h = "";
    PLAN_CUR.forEach(function (wk, wi) {
      var p = pw[wi], done = Math.max(0, Math.min(p.total, TIME.total - p.start)), pct = Math.round(done / p.total * 100);
      var cur = TIME.total >= p.start && TIME.total < p.end;
      h += "<div class='planweek" + (cur ? " cur" : "") + "'><div class='pwtop'><b>" + esc(wk.title) + "</b><span>" + pct + "%</span></div>" +
        "<div class='trackbar'><div class='trackfill' style='width:" + pct + "%'></div></div>" +
        "<div class='pwsecs'>" + esc(T.secs) + " " + wk.secs.map(function (s) { var pg = secPage(s); return (s === "occ" ? "Occ" : "S" + s) + (pg ? (" · p." + pg) : ""); }).join("  ") + "</div></div>";
    });
    $("planList").innerHTML = h;
  }

  // ===== Code-Hunt drills (bilingual) =====
  var CH = { pool: [], i: 0, timer: null, end: 0, found: 0, total: 0, revealed: false };
  function chT() {
    return getLang() === "fa"
      ? { title: "🧭 تمرینِ Code-Hunt", intro: "یک سناریو + تایمرِ ۹۰ ثانیه. مسابقه بده تا Rule/Table مربوطه را در کتاب پیدا کنی، بعد «نمایش محل» را بزن. سرعتِ کار با کتابِ تمیز را می‌سازد.", find: "این کجای کتاب است؟", score: "به‌موقع پیدا شد:", show: "نمایش محل", found: "✅ به‌موقع پیدا کردم", miss: "❌ نشد" }
      : { title: "🧭 Code-Hunt Drills", intro: "A scenario appears + a 90-second timer. Race to find the governing Rule/Table in your code book, then reveal to check. Builds clean-book speed.", find: "Find where this lives:", score: "Found in time:", show: "Reveal location", found: "✅ Found in time", miss: "❌ Missed" };
  }
  function startCodeHunt() {
    var rows = [];
    STUDY.forEach(function (s) { (s.keywords || []).forEach(function (k) { rows.push({ sec: secLabel(s), kw: k.kw, jump: k.jump, why: k.why, why_fa: k.why_fa }); }); });
    CH = { pool: shuffle(rows), i: 0, timer: null, end: 0, found: 0, total: 0, revealed: false };
    langBar("chLang", chLocalize);
    chNext();
  }
  function chRenderPrompt() {
    var T = chT(), r = CH.pool[CH.i % CH.pool.length]; if (!r) return;
    $("chPrompt").innerHTML = esc(T.find) + "<br><b>" + esc(r.kw) + "</b><div class='chsec'>" + esc(r.sec) + "</div>";
  }
  function chRenderReveal() {
    var r = CH.pool[CH.i % CH.pool.length];
    var why = (getLang() === "fa" && r.why_fa) ? r.why_fa : r.why;
    $("chReveal").classList.remove("hidden");
    $("chReveal").innerHTML = "<b>→ " + esc(r.jump) + "</b> " + pageChip(r.jump) + (why ? ("<div class='chwhy'>" + esc(why) + "</div>") : "");
  }
  function chLocalize() {
    var T = chT();
    if ($("chTitle")) $("chTitle").textContent = T.title;
    if ($("chIntro")) $("chIntro").textContent = T.intro;
    $("chScore").textContent = T.score + " " + CH.found + " / " + CH.total;
    $("chShow").textContent = T.show; $("chFound").textContent = T.found; $("chMiss").textContent = T.miss;
    chRenderPrompt();
    if (CH.revealed) chRenderReveal();
  }
  function chNext() {
    if (CH.timer) clearInterval(CH.timer);
    CH.revealed = false;
    $("chReveal").classList.add("hidden"); $("chReveal").innerHTML = "";
    $("chShow").classList.remove("hidden"); $("chFound").classList.add("hidden"); $("chMiss").classList.add("hidden");
    chLocalize();
    CH.end = Date.now() + 90000; chTick(); CH.timer = setInterval(chTick, 250);
  }
  function chTick() { var ms = CH.end - Date.now(); if (ms < 0) ms = 0; var t = $("chTimer"); t.textContent = fmt(ms); t.classList.toggle("warn", ms < 20000); if (ms <= 0) { clearInterval(CH.timer); chShowAns(); } }
  function chShowAns() {
    if (CH.timer) clearInterval(CH.timer);
    CH.revealed = true; chRenderReveal();
    $("chShow").classList.add("hidden"); $("chFound").classList.remove("hidden"); $("chMiss").classList.remove("hidden");
  }
  function chMark(ok) { CH.total++; if (ok) CH.found++; CH.i++; chNext(); }

  // ---------- nav ----------
  function gotoSettings(note) {
    show("settings");
    $("apiKey").value = getKey();
    $("model").value = getModel();
    if (note) { var st = $("keyStatus"); st.className = "status err"; st.textContent = note; }
  }
  function homeStats() {
    var seen = Object.keys(PROG).length;
    var weak = Object.keys(PROG).filter(function (id) { return PROG[id].wrong > 0; }).length;
    $("stats").innerHTML = seen ? ("📊 " + seen + " questions practised · " + weak + " in your weak list") : "";
    $("libline").textContent = ALL.length + " questions loaded · CEC 2024";
  }

  // ---------- events ----------
  document.addEventListener("click", function (ev) {
    var t = ev.target.closest("[data-home]");
    if (t) { if (Q.totalTimer) { clearInterval(Q.totalTimer); Q.totalTimer = null; } show("home"); homeStats(); return; }
    var ds = ev.target.closest("[data-study]");
    if (ds) { buildStudyList(); show("study"); return; }
    var mode = ev.target.closest("[data-mode]");
    if (mode) {
      var m = mode.getAttribute("data-mode");
      if (m === "mock") startQuiz(pickWeighted(100), "mock", true);
      else if (m === "section") { buildPicker(); show("picker"); }
      else if (m === "weak") {
        var weak = ALL.filter(function (q) { return PROG[q.id] && PROG[q.id].wrong > 0; });
        if (!weak.length) { alert("No weak questions yet — practise some first!"); return; }
        startQuiz(shuffle(weak), "weak", false);
      } else if (m === "random") startQuiz(shuffle(ALL).slice(0, 20), "random", false);
      return;
    }
    var go = ev.target.closest("[data-go]");
    if (go) {
      var g = go.getAttribute("data-go");
      if (g === "tutor") show("tutor");
      else if (g === "generate") show("generate");
      else if (g === "vision") show("vision");
      else if (g === "study") { buildStudyList(); show("study"); }
      else if (g === "keymap") { show("keymap"); renderKeymap($("kmSearch").value); }
      else if (g === "plan") { show("plan"); renderPlan(); }
      else if (g === "codehunt") { startCodeHunt(); show("codehunt"); }
      return;
    }
  });

  $("gear").onclick = function () { gotoSettings(); };
  $("next").onclick = nextQ;
  $("prev").onclick = prevQ;
  $("finish").onclick = function () {
    var un = Q.answers.filter(function (a) { return !a; }).length;
    if (un && !confirm(un + " question(s) still unanswered. Finish anyway?")) return;
    finish();
  };
  $("teach").onclick = teachThis;
  // keyboard: A–D to answer, ← / → to move
  document.addEventListener("keydown", function (e) {
    if ($("quiz").classList.contains("hidden")) return;
    var tag = (document.activeElement && document.activeElement.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    var k = e.key.toUpperCase();
    if (k === "ARROWRIGHT") { nextQ(); e.preventDefault(); }
    else if (k === "ARROWLEFT") { prevQ(); e.preventDefault(); }
    else if ("ABCD".indexOf(k) >= 0 && !Q.answers[Q.i]) { choose(k); e.preventDefault(); }
  });
  $("chatsend").onclick = function () { sendChat(); };
  $("chatinput").addEventListener("keydown", function (e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } });
  $("kmSearch").addEventListener("input", function (e) { renderKeymap(e.target.value); });
  $("lessonDrill").onclick = lessonDrill;
  $("lessonGen").onclick = lessonGen;
  $("lessonTeach").onclick = lessonTeach;
  $("flag").onclick = toggleFlag;
  $("timerToggle").onclick = function () { if (TIME.running) timePause(); else timeStart(); };
  $("timerReset").onclick = function () { if (!confirm("Reset today's studied time?")) return; if (TIME.running) timePause(); var k = dkey(); TIME.total = Math.max(0, TIME.total - (TIME.days[k] || 0)); delete TIME.days[k]; saveTime(); planUI(); };
  $("chShow").onclick = chShowAns;
  $("chFound").onclick = function () { chMark(true); };
  $("chMiss").onclick = function () { chMark(false); };
  setInterval(timeHeartbeat, 5000);
  document.addEventListener("visibilitychange", function () { if (document.hidden) timeCredit(); });
  if (TIME.running) { _tick = Date.now(); }
  planUI();
  $("genGo").onclick = generate;
  $("visionGo").onclick = askVision;
  $("fileInput").onchange = function (e) { handleFile(e.target.files[0]); };
  $("camInput").onchange = function (e) { handleFile(e.target.files[0]); };
  $("saveKey").onclick = function () {
    localStorage.setItem("cec_apikey", $("apiKey").value.trim());
    localStorage.setItem("cec_model", $("model").value);
    var st = $("keyStatus"); st.className = "status ok"; st.textContent = "Saved on this device. ✅";
  };

  // ---------- boot ----------
  homeStats();
})();
