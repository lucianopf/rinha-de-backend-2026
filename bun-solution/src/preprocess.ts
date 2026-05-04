/**
 * Pre-processing script: loads references.json.gz, builds quantized search index,
 * and serializes it to a binary file for fast loading at startup.
 * 
 * Memory optimization: quantizes Float32 vectors (168MB) to Uint8 (42MB)
 * to fit within the 165MB container memory limit.
 */
import { gunzipSync } from "zlib";
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { buildIndex, serializeIndex } from "./search";

const RESOURCES_DIR = process.env.RESOURCES_DIR || "../resources";
const OUTPUT_DIR = process.env.OUTPUT_DIR || "./data";

console.time("preprocess-total");

// Load references
console.time("load-references");
const gzipped = readFileSync(`${RESOURCES_DIR}/references.json.gz`);
const decompressed = gunzipSync(gzipped);
const references: Array<{ vector: number[]; label: string }> = JSON.parse(
  decompressed.toString()
);
console.timeEnd("load-references");
console.log(`Loaded ${references.length} reference vectors`);

// Convert to typed arrays
console.time("convert-and-quantize");
const vectors: Float32Array[] = new Array(references.length);
const labels: boolean[] = new Array(references.length);

for (let i = 0; i < references.length; i++) {
  vectors[i] = new Float32Array(references[i].vector);
  labels[i] = references[i].label === "fraud";
}
console.timeEnd("convert-and-quantize");

// Build quantized search index
console.time("build-index");
const index = buildIndex(vectors, labels);
console.timeEnd("build-index");

// Serialize
console.time("serialize");
mkdirSync(OUTPUT_DIR, { recursive: true });
const serialized = serializeIndex(index);
writeFileSync(`${OUTPUT_DIR}/index.bin`, serialized);
console.timeEnd("serialize");
console.log(`Search index serialized: ${(serialized.length / 1024 / 1024).toFixed(1)} MB`);

// Also copy normalization and mcc_risk for runtime
const normalization = readFileSync(`${RESOURCES_DIR}/normalization.json`, "utf-8");
writeFileSync(`${OUTPUT_DIR}/normalization.json`, normalization);

const mccRisk = readFileSync(`${RESOURCES_DIR}/mcc_risk.json`, "utf-8");
writeFileSync(`${OUTPUT_DIR}/mcc_risk.json`, mccRisk);

console.timeEnd("preprocess-total");
console.log("Pre-processing complete!");
