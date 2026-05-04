/**
 * IVF Search Worker - runs IVF search in a worker thread.
 * Approach 3: IVF + Worker Threads Combined
 */
import { readFileSync } from "fs";
import { deserializeIVFIndex, findFraudCountIVF, type IVFIndex } from "./ivf-search";

const DATA_DIR = process.env.DATA_DIR || "./data";

// Load IVF index in worker
const indexBuffer = readFileSync(`${DATA_DIR}/ivf-index.bin`);
const index: IVFIndex = deserializeIVFIndex(Buffer.from(indexBuffer));

declare const self: Worker;
self.postMessage({ type: "ready" });

// Handle search requests
self.onmessage = (event: MessageEvent) => {
  const { id, query } = event.data;
  const queryVec = new Uint8Array(query);
  const fraudCount = findFraudCountIVF(index, queryVec);
  self.postMessage({ id, fraudCount });
};
