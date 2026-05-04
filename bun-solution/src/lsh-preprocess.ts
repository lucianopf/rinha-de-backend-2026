/**
 * LSH Pre-processing script.
 */
import { gunzipSync } from "zlib";
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { buildLSHIndex, serializeLSHIndex } from "./lsh-search";

const RESOURCES_DIR = process.env.RESOURCES_DIR || "../resources";
const OUTPUT_DIR = process.env.OUTPUT_DIR || "./data";

console.time("lsh-preprocess-total");

console.time("load-references");
const gzipped = readFileSync(`${RESOURCES_DIR}/references.json.gz`);
const decompressed = gunzipSync(gzipped);
const references: Array<{ vector: number[]; label: string }> = JSON.parse(
  decompressed.toString()
);
console.timeEnd("load-references");
console.log(`Loaded ${references.length} reference vectors`);

console.time("quantize");
const DIMENSIONS = 14;
const n = references.length;
const vectors = new Uint8Array(n * DIMENSIONS);
const labels = new Uint8Array(n);

for (let i = 0; i < n; i++) {
  const vec = references[i].vector;
  const offset = i * DIMENSIONS;
  for (let d = 0; d < DIMENSIONS; d++) {
    const mapped = (vec[d] + 1) * 127.5;
    vectors[offset + d] = mapped < 0 ? 0 : mapped > 255 ? 255 : mapped | 0;
  }
  labels[i] = references[i].label === "fraud" ? 1 : 0;
}
console.timeEnd("quantize");

// Build LSH index with L=4, K=10
console.time("build-lsh");
console.log(`Building LSH index with L=4, K=10...`);
const index = buildLSHIndex(vectors, labels, n, 4, 10);
console.timeEnd("build-lsh");

console.time("serialize");
mkdirSync(OUTPUT_DIR, { recursive: true });
const serialized = serializeLSHIndex(index);
writeFileSync(`${OUTPUT_DIR}/lsh-index.bin`, serialized);
console.timeEnd("serialize");
console.log(`LSH index serialized: ${(serialized.length / 1024 / 1024).toFixed(1)} MB`);

writeFileSync(`${OUTPUT_DIR}/normalization.json`, readFileSync(`${RESOURCES_DIR}/normalization.json`, "utf-8"));
writeFileSync(`${OUTPUT_DIR}/mcc_risk.json`, readFileSync(`${RESOURCES_DIR}/mcc_risk.json`, "utf-8"));

console.timeEnd("lsh-preprocess-total");
console.log("LSH pre-processing complete!");
