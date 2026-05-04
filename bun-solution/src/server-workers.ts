/**
 * Server with Worker Threads - Approach 2
 * Offloads brute-force search to worker threads to avoid blocking the event loop.
 */
import { readFileSync } from "fs";
import { vectorize, type NormalizationConstants, type MccRiskMap, type TransactionPayload } from "./vectorize";
import { quantizeVector } from "./search";
import { WorkerPool } from "./worker-pool";

const DATA_DIR = process.env.DATA_DIR || "./data";
const PORT = parseInt(process.env.PORT || "9999", 10);
const NUM_WORKERS = parseInt(process.env.NUM_WORKERS || "2", 10);

// Load config in main thread (lightweight)
console.time("load-config");
const normalization: NormalizationConstants = JSON.parse(readFileSync(`${DATA_DIR}/normalization.json`, "utf-8"));
const mccRisk: MccRiskMap = JSON.parse(readFileSync(`${DATA_DIR}/mcc_risk.json`, "utf-8"));
console.timeEnd("load-config");

// Create worker pool
const pool = new WorkerPool(new URL("./search-worker.ts", import.meta.url).pathname, NUM_WORKERS);

const RESPONSES: Record<number, string> = {
  0: '{"approved":true,"fraud_score":0}',
  1: '{"approved":true,"fraud_score":0.2}',
  2: '{"approved":true,"fraud_score":0.4}',
  3: '{"approved":false,"fraud_score":0.6}',
  4: '{"approved":false,"fraud_score":0.8}',
  5: '{"approved":false,"fraud_score":1}',
};
const JSON_HEADERS = { "Content-Type": "application/json" };

// Wait for workers to be ready before starting server
await pool.waitReady();
console.log(`All ${NUM_WORKERS} workers ready`);

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/ready") {
      return new Response("OK", { status: 200 });
    }

    if (url.pathname === "/fraud-score") {
      if (req.method !== "POST") {
        return new Response("Method Not Allowed", { status: 405 });
      }

      try {
        const payload: TransactionPayload = await req.json();
        const floatVector = vectorize(payload, normalization, mccRisk);
        const queryVector = quantizeVector(floatVector);
        const fraudCount = await pool.findFraudCount(queryVector);
        return new Response(RESPONSES[fraudCount], { status: 200, headers: JSON_HEADERS });
      } catch {
        return new Response(RESPONSES[0], { status: 200, headers: JSON_HEADERS });
      }
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`Worker pool server running on port ${PORT}`);
