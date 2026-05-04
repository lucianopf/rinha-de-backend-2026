/**
 * Benchmark script - tests accuracy and latency of a running server.
 * Usage: DATA_DIR=./data bun run src/benchmark.ts [concurrency] [totalRequests]
 */
import { readFileSync } from "fs";

const TEST_DATA_PATH = process.env.TEST_DATA_PATH || "../test/test-data.json";
const URL = process.env.BENCH_URL || "http://localhost:9999/fraud-score";
const CONCURRENCY = parseInt(process.argv[2] || "20", 10);
const TOTAL = parseInt(process.argv[3] || "2000", 10);

// Load test data
const data = JSON.parse(readFileSync(TEST_DATA_PATH, "utf-8"));
const entries = data.entries.slice(0, TOTAL);

console.log(`Benchmark: ${TOTAL} requests, concurrency ${CONCURRENCY}`);
console.log(`Target: ${URL}\n`);

let tp = 0, tn = 0, fp = 0, fn = 0, errors = 0;
const latencies: number[] = [];

async function sendRequest(entry: any) {
  const start = performance.now();
  try {
    const res = await fetch(URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry.request),
    });
    latencies.push(performance.now() - start);

    if (res.status === 200) {
      const body = (await res.json()) as { approved: boolean };
      if (entry.expected_approved === body.approved) {
        if (body.approved) tn++;
        else tp++;
      } else {
        if (body.approved) fn++;
        else fp++;
      }
    } else {
      await res.text();
      errors++;
    }
  } catch {
    latencies.push(performance.now() - start);
    errors++;
  }
}

const t0 = performance.now();
for (let i = 0; i < entries.length; i += CONCURRENCY) {
  await Promise.all(entries.slice(i, i + CONCURRENCY).map(sendRequest));
}
const totalTime = (performance.now() - t0) / 1000;

// Compute metrics
latencies.sort((a, b) => a - b);
const N = tp + tn + fp + fn + errors;
const p50 = latencies[Math.floor(N * 0.50)];
const p90 = latencies[Math.floor(N * 0.90)];
const p99 = latencies[Math.floor(N * 0.99)];
const avg = latencies.reduce((a, b) => a + b, 0) / N;
const failures = fp + fn + errors;
const failureRate = failures / N;

// Score calculation (matches AVALIACAO.md formula)
const K_SCORE = 1000, T_MAX = 1000, P99_MIN = 1, P99_MAX = 2000;
const BETA = 300;

const p99Score = p99 > P99_MAX ? -3000 : K_SCORE * Math.log10(T_MAX / Math.max(p99, P99_MIN));
const E = fp + 3 * fn + 5 * errors;
const epsilon = N > 0 ? E / N : 0;
let detScore: number;
if (failureRate > 0.15) detScore = -3000;
else if (E === 0) detScore = 3000;
else detScore = K_SCORE * Math.log10(1 / Math.max(epsilon, 0.001)) - BETA * Math.log10(1 + E);

const finalScore = p99Score + detScore;

console.log(`═══════════════════════════════════════════════`);
console.log(`  ${N} reqs in ${totalTime.toFixed(1)}s (${(N / totalTime).toFixed(0)} req/s)`);
console.log(`  Latency: avg=${avg.toFixed(1)}ms p50=${p50.toFixed(1)}ms p90=${p90.toFixed(1)}ms p99=${p99.toFixed(1)}ms`);
console.log(`  Detection: TP=${tp} TN=${tn} FP=${fp} FN=${fn} Err=${errors}`);
console.log(`  Failure rate: ${(failureRate * 100).toFixed(2)}%`);
console.log(`  Score: p99=${p99Score.toFixed(0)} det=${detScore.toFixed(0)} FINAL=${finalScore.toFixed(0)}`);
console.log(`═══════════════════════════════════════════════`);
