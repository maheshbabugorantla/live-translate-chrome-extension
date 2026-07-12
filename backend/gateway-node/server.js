/*
 * FDE · Assignment 1 · Node Gateway  (the "software backend")
 * ==========================================================
 * This is the ONLY server the browser widget talks to. Its jobs:
 *   - serve the widget file at /widget.js
 *   - accept translation requests from the widget (CORS, validation)
 *   - forward them to the Python AI service
 *   - expose /health and /stats
 *   - log every request
 *
 * It is ~90% done. Find the two `TODO (YOU)` blocks and implement them.
 * Everything else works out of the box.
 *
 * Run:  npm install && npm start      (needs Node 18+ for global fetch)
 */
const express = require("express");
const cors = require("cors");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
require("dotenv").config();

const PORT = process.env.PORT || 8787;
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://localhost:8000";
const WIDGET_PATH = path.join(__dirname, "..", "..", "widget", "translation-widget.js");
const LOG_FILE = path.join(__dirname, "gateway.log");

const app = express();
const startedAt = Date.now();

// --- request logging: one grep-friendly line per finished request, mirrored
// to gateway.log so a request's trace ID can be correlated with the AI
// service's own log file for the same request. ----------------------------
function logLine(line) {
  console.log(line);
  fs.appendFile(LOG_FILE, line + "\n", () => {});
}

app.use(cors()); // dev: allow every origin so the widget works on any page
app.use(express.json({ limit: "1mb" }));

app.use((req, res, next) => {
  const t0 = Date.now();
  // Trust an incoming X-Request-Id (from the widget/eval); otherwise mint one
  // so every request is traceable across both services.
  req.requestId = req.headers["x-request-id"] || crypto.randomUUID();
  res.on("finish", () => {
    const ms = Date.now() - t0;
    logLine(
      `[req] requestId=${req.requestId} method=${req.method} url=${req.originalUrl} status=${res.statusCode} duration=${ms}ms`
    );
  });
  next();
});

// --- serve the widget to the console loader ------------------------------
app.get("/widget.js", (req, res) => {
  res.type("application/javascript");
  res.sendFile(WIDGET_PATH);
});

// --- helper: forward a request to the Python AI service ------------------
/*
 * TODO (YOU) #2 — implement the proxy call.
 * POST `body` as JSON to `${AI_SERVICE_URL}${path}` and return the parsed
 * JSON response. Throw on a non-2xx so callers can turn it into a 502.
 * (Node 18+ has global fetch — no import needed.)
 */
async function callAiService(path, body, requestId) {
  const res = await fetch(AI_SERVICE_URL + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Request-Id": requestId },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("AI service " + res.status);
  return res.json();
}

// --- routes the widget calls ---------------------------------------------
app.post("/translate", async (req, res) => {
  const { text, target } = req.body || {};
  if (typeof text !== "string") return res.status(400).json({ error: "`text` (string) is required" });
  try {
    const data = await callAiService("/translate", { text, target: target || "es-MX" }, req.requestId);
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: "AI service error: " + err.message });
  }
});

app.post("/translate/batch", async (req, res) => {
  const { texts, target } = req.body || {};
  if (!Array.isArray(texts)) return res.status(400).json({ error: "`texts` (array) is required" });
  try {
    const data = await callAiService("/translate/batch", { texts, target: target || "es-MX" }, req.requestId);
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: "AI service error: " + err.message });
  }
});

app.get("/health", async (req, res) => {
  const uptimeSec = Math.round((Date.now() - startedAt) / 1000);
  let ai = "unreachable";
  try {
    const r = await fetch(AI_SERVICE_URL + "/health");
    ai = r.ok ? await r.json() : "error";
  } catch (_) {}
  res.json({ status: "ok", gatewayUptimeSec: uptimeSec, aiService: ai });
});

app.get("/stats", async (req, res) => {
  try {
    const r = await fetch(AI_SERVICE_URL + "/stats");
    res.json(await r.json());
  } catch (err) {
    res.status(502).json({ error: "AI service error: " + err.message });
  }
});

app.listen(PORT, () => {
  console.log(`FDE gateway on http://localhost:${PORT}  →  AI service ${AI_SERVICE_URL}`);
  console.log(`Widget served at http://localhost:${PORT}/widget.js`);
});
