/* CEC Coach — web app  (v1.2.0 — markdown answers + error surfacing)
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
  function mdToHtml(md) {
    if (!md) return "";
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
    "FORMATTING (always): reply in clean Markdown for a study app — use ## headings, **bold** for the final answer and for each keyword, numbered steps, and Markdown TABLES for any lookup/comparison/calculation. Separate major parts with a horizontal rule (---) and keep generous spacing. When a figure helps (circuit, conduit cross-section, panel/box layout, one-line), draw a clear ASCII/Unicode diagram inside a ``` fenced code block. Structure most answers as: ## Answer · ## Keyword(s) to spot · ## Table/Rule to jump to · ## Fastest steps (numbered) · ## خلاصهٔ فارسی (a short right-to-left Persian summary).";

  // ---------- quiz engine ----------
  var Q = { list: [], i: 0, correct: 0, mode: "", answered: false, timer: null, endAt: 0, answers: [] };

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

  function startQuiz(list, mode, timed) {
    // safety: never show a card you can't answer
    list = (list || []).filter(function (q) { return (q.options || []).length >= 2 && q.answer_key; });
    if (!list.length) { alert("No answerable questions in that selection yet."); return; }
    Q = { list: list, i: 0, correct: 0, mode: mode, answered: false, timer: null, endAt: 0, answers: [] };
    show("quiz");
    if (timed) {
      var mins = list.length >= 100 ? (BP.time_minutes_standard || 240) : Math.max(1, Math.round(list.length * 1.5));
      Q.endAt = Date.now() + mins * 60000;
      $("timer").classList.remove("hidden");
      tick();
      Q.timer = setInterval(tick, 1000);
    } else {
      $("timer").classList.add("hidden");
    }
    renderQ();
  }
  function tick() {
    var ms = Q.endAt - Date.now();
    if (ms <= 0) { ms = 0; clearInterval(Q.timer); finish(); return; }
    var m = Math.floor(ms / 60000), s = Math.floor((ms % 60000) / 1000);
    var t = $("timer"); t.textContent = "⏱ " + m + ":" + (s < 10 ? "0" : "") + s;
    t.classList.toggle("warn", ms < 60000);
  }

  function renderQ() {
    Q.answered = false;
    var q = Q.list[Q.i];
    $("bar").style.width = ((Q.i) / Q.list.length * 100) + "%";
    $("qcount").textContent = (Q.i + 1) + " / " + Q.list.length;
    var tags = $("qtags"); tags.innerHTML = "";
    var secName = SECTION_NAMES[q.section] || ("Section " + (q.section || "?"));
    tags.appendChild(el("span", "tag", esc(secName)));
    if (q.block) tags.appendChild(el("span", "tag block", "Block " + esc(q.block)));
    if (q.topic) tags.appendChild(el("span", "tag", esc(q.topic)));
    $("qtext").textContent = q.question;
    var opts = $("options"); opts.innerHTML = "";
    (q.options || []).forEach(function (optText, idx) {
      var keyLetter = (optText.match(/^([A-Da-d])\)/) || [])[1] || String.fromCharCode(65 + idx);
      var label = optText.replace(/^([A-Da-d])\)\s*/, "");
      var b = el("button", "opt");
      b.appendChild(el("span", "k", esc(keyLetter.toUpperCase())));
      b.appendChild(el("span", null, esc(label)));
      b.dataset.key = keyLetter.toUpperCase();
      b.onclick = function () { choose(b.dataset.key); };
      opts.appendChild(b);
    });
    $("reveal").classList.add("hidden");
    $("explain").classList.add("hidden");
    $("explain").innerHTML = "";
    $("next").classList.add("hidden");
    $("askai").classList.add("hidden");
  }

  function choose(key) {
    if (Q.answered) return;
    Q.answered = true;
    var q = Q.list[Q.i];
    var correctKey = (q.answer_key || "").toUpperCase();
    var ok = key === correctKey;
    if (ok) Q.correct++;
    record(q.id, ok);
    Q.answers.push({ q: q, picked: key, ok: ok });
    var btns = $("options").children;
    for (var i = 0; i < btns.length; i++) {
      var b = btns[i]; b.disabled = true;
      if (b.dataset.key === correctKey) b.classList.add("correct");
      else if (b.dataset.key === key) b.classList.add("wrong");
      else b.classList.add("dim");
    }
    showExplain(q);
    $("next").classList.remove("hidden");
    $("askai").classList.remove("hidden");
    $("next").textContent = (Q.i + 1 >= Q.list.length) ? "See results →" : "Next →";
  }

  function showExplain(q) {
    var e = $("explain");
    var refs = (q.references || []).join(" · ");
    var html = "";
    html += "<h4>Answer</h4><p>" + esc(q.answer || ("Option " + (q.answer_key || ""))) + "</p>";
    if (q.solution_steps && q.solution_steps.length) {
      html += "<h4>Fastest path</h4><ol>";
      q.solution_steps.forEach(function (s) { html += "<li>" + esc(s) + "</li>"; });
      html += "</ol>";
    }
    if (refs) html += "<div class='refs'>📖 " + esc(refs) + "</div>";
    e.innerHTML = html;
    e.classList.remove("hidden");
  }

  function nextQ() {
    Q.i++;
    if (Q.i >= Q.list.length) { finish(); return; }
    renderQ();
  }

  function finish() {
    if (Q.timer) clearInterval(Q.timer);
    show("results");
    var pct = Math.round(Q.correct / Q.list.length * 100);
    var pass = pct >= (BP.pass_mark_percent || 70);
    $("score").innerHTML = pct + "%<small>" + Q.correct + " / " + Q.list.length +
      " correct · pass mark " + (BP.pass_mark_percent || 70) + "%</small>";
    var d = $("rdetail");
    var missed = Q.answers.filter(function (a) { return !a.ok; });
    var h = "<p class='" + (pass ? "pass" : "fail") + "'>" + (pass ? "✅ Pass — keep this pace." : "❌ Below 70% — drill the misses.") + "</p>";
    if (missed.length) {
      h += "<p>Review these:</p><ul>";
      missed.slice(0, 20).forEach(function (a) {
        var sec = SECTION_NAMES[a.q.section] || ("Sec " + a.q.section);
        h += "<li>" + esc(sec) + " — " + esc((a.q.references || []).join(", ") || a.q.topic || a.q.id) + "</li>";
      });
      h += "</ul>";
    }
    d.innerHTML = h;
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
      system: TUTOR_SYSTEM, maxTokens: 4096, onText: paint
    }).then(function (full) {
      wait.stop(); bubble.className = "msg ai rich"; bubble.innerHTML = mdToHtml(full);
      chatHistory.push({ role: "assistant", content: full });
      $("chat").scrollTop = $("chat").scrollHeight;
    }).catch(function (err) { wait.stop(); bubble.className = "msg ai"; bubble.textContent = "⚠️ " + apiErr(err); });
  }

  // ---------- generate questions ----------
  function generate() {
    var topic = $("genTopic").value.trim() || "mixed CEC 2024 topics";
    var count = $("genCount").value;
    if (!getKey()) { gotoSettings("Add your API key first to generate questions."); return; }
    var st = $("genStatus"); st.className = "status"; st.textContent = "Generating " + count + " questions…";
    $("genGo").disabled = true;
    var prompt =
      "Write " + count + " NEW multiple-choice questions for the BC Construction Electrician exam (CEC 2024) on: " + topic + ".\n" +
      "Return ONLY a JSON array, no prose. Each item:\n" +
      '{"id":"GEN-<n>","section":<number>,"block":"A|B|C|D|E","topic":"short","question":"...",' +
      '"options":["A) ...","B) ...","C) ...","D) ..."],"answer":"the correct answer text","answer_key":"A|B|C|D",' +
      '"references":["Rule/Table"],"solution_steps":["name the keyword and which Table/Rule to jump to","then the steps to the answer"]}\n' +
      "Make them realistic and open-book style. Be accurate with CEC 2024 references.";
    callClaude([{ role: "user", content: prompt }], { maxTokens: 4000 })
      .then(function (full) {
        var arr = parseJSONArray(full);
        if (!arr || !arr.length) throw new Error("Could not parse generated questions.");
        arr.forEach(function (q, i) { q.id = q.id || ("GEN-" + Date.now() + "-" + i); q._generated = true; });
        st.className = "status ok"; st.textContent = "Made " + arr.length + " questions. Starting…";
        $("genGo").disabled = false;
        setTimeout(function () { startQuiz(arr, "generated", false); }, 500);
      })
      .catch(function (err) { st.className = "status err"; st.textContent = "⚠️ " + apiErr(err); $("genGo").disabled = false; });
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
      system: TUTOR_SYSTEM, maxTokens: 4096, onText: paint
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
  function renderKeymap(q) {
    q = (q || "").trim().toLowerCase();
    var host = $("kmList"); host.innerHTML = "";
    var any = false;
    KEYWORD_MAP.forEach(function (group) {
      var rows = group.rows.filter(function (r) {
        if (!q) return true;
        return (r[0] + " " + r[1] + " " + r[2] + " " + group.sec).toLowerCase().indexOf(q) >= 0;
      });
      if (!rows.length) return;
      any = true;
      var box = el("div", "kmsec");
      var count = group.rows.length;
      box.appendChild(el("h3", null, esc(group.sec) + " <span class='pill'>" + rows.length + "/" + count + "</span>"));
      rows.forEach(function (r) {
        var row = el("div", "kmrow");
        row.innerHTML =
          "<div class='kmkw'>" + hl(r[0], q) + "</div>" +
          "<div class='kmjump'><span class='kmlbl'>Jump to</span><span class='arrow'>→</span> <b>" + hl(r[1], q) + "</b></div>" +
          "<div class='kmwhy'>" + (r[2] ? "<span class='kmlbl'>Why</span>" + hl(r[2], q) : "") + "</div>";
        box.appendChild(row);
      });
      host.appendChild(box);
    });
    $("kmEmpty").classList.toggle("hidden", any);
  }

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
    if (t) { if (Q.timer) clearInterval(Q.timer); show("home"); homeStats(); return; }
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
      else if (g === "keymap") { show("keymap"); renderKeymap($("kmSearch").value); }
      return;
    }
  });

  $("gear").onclick = function () { gotoSettings(); };
  $("next").onclick = nextQ;
  $("askai").onclick = function () {
    var q = Q.list[Q.i];
    show("tutor");
    sendChat("Explain this exam question step by step (keyword → table, fastest path), with a short Persian summary:\n\n" +
      q.question + "\n" + (q.options || []).join("\n") + "\nCorrect answer: " + (q.answer_key || ""));
  };
  $("chatsend").onclick = function () { sendChat(); };
  $("chatinput").addEventListener("keydown", function (e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } });
  $("kmSearch").addEventListener("input", function (e) { renderKeymap(e.target.value); });
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
