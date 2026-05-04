/**
 * Server using Product Quantization - Approach 4
 */
import { readFileSync } from "fs";
import { deserializePQIndex, findFraudCountPQ, type PQIndex } from "./pq-search";
import { vectorize, type NormalizationConstants, type MccRiskMap, type TransactionPayload } from "./vectorize";
import { quantizeVector } from "./search";

const DATA_DIR = process.env.DATA_DIR || "./data";
const PORT = parseInt(process.env.PORT || "9999", 10);

console.time("load-data");
let index: PQIndex;
let normalization: NormalizationConstants;
let mccRisk: MccRiskMap;

try {
  const indexBuffer = readFileSync(`${DATA_DIR}/pq-index.bin`);
  index = deserializePQIndex(Buffer.from(indexBuffer));
  normalization = JSON.parse(readFileSync(`${DATA_DIR}/normalization.json`, "utf-8"));
  mccRisk = JSON.parse(readFileSync(`${DATA_DIR}/mcc_risk.json`, "utf-8"));
  console.timeEnd("load-data");
  console.log(`PQ index loaded: ${index.n} vectors, codes=${(index.codes.byteLength / 1024 / 1024).toFixed(1)} MB`);
} catch (e) {
  console.error("Failed to load PQ data.", e);
  process.exit(1);
}

const RESPONSES: Record<number, string> = {
  0: '{"approved":true,"fraud_score":0}',
  1: '{"approved":true,"fraud_score":0.2}',
  2: '{"approved":true,"fraud_score":0.4}',
  3: '{"approved":false,"fraud_score":0.6}',
  4: '{"approved":false,"fraud_score":0.8}',
  5: '{"approved":false,"fraud_score":1}',
};
const JSON_HEADERS = { "Content-Type": "application/json" };

const server = Bun.serve({
  port: PORT,
  fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/ready") {
      return new Response("OK", { status: 200 });
    }

    if (url.pathname === "/fraud-score") {
      if (req.method !== "POST") {
        return new Response("Method Not Allowed", { status: 405 });
      }

      return req.json().then((payload: TransactionPayload) => {
        const floatVector = vectorize(payload, normalization, mccRisk);
        const queryVector = quantizeVector(floatVector);
        const fraudCount = findFraudCountPQ(index, queryVector);
        return new Response(RESPONSES[fraudCount], { status: 200, headers: JSON_HEADERS });
      }).catch(() => {
        return new Response(RESPONSES[0], { status: 200, headers: JSON_HEADERS });
      });
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`PQ server running on port ${PORT}`);
