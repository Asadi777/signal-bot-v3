// CEC Coach — AI proxy (Cloudflare Worker)
// Holds YOUR Anthropic API key as a secret so you can share the app with a few
// people WITHOUT giving them the key. The browser app calls this Worker instead
// of api.anthropic.com; the key never leaves the server.
//
// Protections (all optional, set as env vars / secrets — see proxy/README.md):
//   ANTHROPIC_API_KEY  (secret, required)  your sk-ant-… key
//   ACCESS_CODE        (secret, optional)  a password users must enter in Settings
//   ALLOWED_ORIGINS    (var, optional)     comma-separated site URLs allowed to call it
//   FORCE_MODEL        (var, optional)     pin every request to one model (cost control)
//   MAX_TOKENS_CAP     (var, optional)     clamp max_tokens (cost control)

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const cors = {
      "Access-Control-Allow-Origin": origin || "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "content-type, x-access-code",
      "Access-Control-Max-Age": "86400",
      "Vary": "Origin"
    };
    const json = (obj, status) =>
      new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json", ...cors } });

    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    if (request.method !== "POST") return json({ error: { message: "Use POST" } }, 405);

    // Optional: only allow calls from your own site(s)
    const allow = (env.ALLOWED_ORIGINS || "").split(",").map(s => s.trim()).filter(Boolean);
    if (allow.length && origin && !allow.includes(origin))
      return json({ error: { message: "Origin not allowed." } }, 403);

    // Optional: shared access code (you give it privately to your students)
    if (env.ACCESS_CODE) {
      const code = request.headers.get("x-access-code") || "";
      if (code !== env.ACCESS_CODE)
        return json({ error: { message: "Wrong or missing access code. در Settings رمزِ دسترسی را درست وارد کن." } }, 401);
    }

    if (!env.ANTHROPIC_API_KEY)
      return json({ error: { message: "Server is missing ANTHROPIC_API_KEY." } }, 500);

    let body;
    try { body = await request.json(); } catch { return json({ error: { message: "Bad JSON" } }, 400); }

    // Optional cost controls
    if (env.FORCE_MODEL) body.model = env.FORCE_MODEL;
    if (env.MAX_TOKENS_CAP) body.max_tokens = Math.min(body.max_tokens || 3000, +env.MAX_TOKENS_CAP);

    const upstream = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify(body)
    });

    // Stream the SSE response straight back to the browser (with CORS)
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "text/event-stream",
        ...cors
      }
    });
  }
};
