/**
 * Search Worker - runs KNN search in a worker thread.
 * Approach 2: Worker Threads Pool
 * 
 * Each worker loads the index on startup and handles search requests
 * via message passing, keeping the main thread free for I/O.
 */
import { readFileSync } from "fs";
import { deserializeIndex, findFraudCount, type SearchIndex } from "./search";

const DATA_DIR = process.env.DATA_DIR || "./data";

// Load index in worker
const indexBuffer = readFileSync(`${DATA_DIR}/index.bin`);
const index: SearchIndex = deserializeIndex(Buffer.from(indexBuffer));

// Signal ready
declare const self: Worker;
self.postMessage({ type: "ready" });

// Handle search requests
self.onmessage = (event: MessageEvent) => {
  const { id, query } = event.data;
  const queryVec = new Uint8Array(query);
  const fraudCount = findFraudCount(index, queryVec);
  self.postMessage({ id, fraudCount });
};
