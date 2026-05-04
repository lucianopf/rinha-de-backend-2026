/**
 * Server using LSH search - Approach 5
 */
import { readFileSync } from "fs";
import { deserializeLSHIndex, findFraudCountLSH, type LSHIndex } from "./lsh-search";
import { vectorize, type NormalizationConstants, type MccRiskMap, type TransactionPayload } from "./vectorize";
import { quantizeVector } from "./search";

const DATA_DIR = process.env.DATA_DIR || "./data";
const PORT = parseInt(process.env.PORT || "9999", 10);

console.time("load-data");
let index: LSHIndex;
let normalization: NormalizationConstants;
let mccRisk: MccRiskMap;

try {
  const indexBuffer = readFileSync(`${DATA_DIR}/lsh-index.bin`);
  index = deserializeLSHIndex(Buffer.from(indexBuffer));
  normalization = JSON.parse(readFileSync(`${DATA_DIR}/normalization.json`, "utf-8"));
  mccRisk = JSON.parse(readFileSync(`${DATA_DIR}/mcc_risk.json`, "utf-8"));
  console.timeEnd("load-data");
  console.log(`LSH index loaded: ${index.n} vectors, L=${index.L}, K=${index.K}`);
} catch (e) {
  console.error("Failed to load LSH data.", e);
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
        const fraudCount = findFraudCountLSH(index, queryVector);
        return new Response(RESPONSES[fraudCount], { status: 200, headers: JSON_HEADERS });
      }).catch(() => {
        return new Response(RESPONSES[0], { status: 200, headers: JSON_HEADERS });
      });
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`LSH server running on port ${PORT}`);
