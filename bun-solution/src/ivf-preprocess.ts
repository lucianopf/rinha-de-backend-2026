/**
 * IVF Pre-processing script.
 * Builds IVF (Inverted File Index) from references.json.gz.
 */
import { gunzipSync } from "zlib";
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { buildIVFIndex, serializeIVFIndex } from "./ivf-search";

const RESOURCES_DIR = process.env.RESOURCES_DIR || "../resources";
const OUTPUT_DIR = process.env.OUTPUT_DIR || "./data";

console.time("ivf-preprocess-total");

// Load references
console.time("load-references");
const gzipped = readFileSync(`${RESOURCES_DIR}/references.json.gz`);
const decompressed = gunzipSync(gzipped);
const references: Array<{ vector: number[]; label: string }> = JSON.parse(
  decompressed.toString()
);
console.timeEnd("load-references");
console.log(`Loaded ${references.length} reference vectors`);

// Quantize vectors to Uint8
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

// Build IVF index
console.time("build-ivf");
const index = buildIVFIndex(vectors, labels, n, 1024, 20);
console.timeEnd("build-ivf");

// Serialize
console.time("serialize");
mkdirSync(OUTPUT_DIR, { recursive: true });
const serialized = serializeIVFIndex(index);
writeFileSync(`${OUTPUT_DIR}/ivf-index.bin`, serialized);
console.timeEnd("serialize");
console.log(`IVF index serialized: ${(serialized.length / 1024 / 1024).toFixed(1)} MB`);

// Copy normalization and mcc_risk
const normalization = readFileSync(`${RESOURCES_DIR}/normalization.json`, "utf-8");
writeFileSync(`${OUTPUT_DIR}/normalization.json`, normalization);
const mccRisk = readFileSync(`${RESOURCES_DIR}/mcc_risk.json`, "utf-8");
writeFileSync(`${OUTPUT_DIR}/mcc_risk.json`, mccRisk);

console.timeEnd("ivf-preprocess-total");
console.log("IVF pre-processing complete!");
