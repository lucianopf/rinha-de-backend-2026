/**
 * Worker Pool - manages a pool of search workers.
 * Approach 2: Distributes search tasks across worker threads.
 */

interface PendingRequest {
  resolve: (fraudCount: number) => void;
  reject: (err: Error) => void;
}

export class WorkerPool {
  private workers: Worker[] = [];
  private pending: Map<number, PendingRequest> = new Map();
  private nextId = 0;
  private roundRobin = 0;
  private readyCount = 0;
  private readyPromiseResolve: (() => void) | null = null;
  private numWorkers: number;

  constructor(workerPath: string, numWorkers: number = 2) {
    this.numWorkers = numWorkers;
    for (let i = 0; i < numWorkers; i++) {
      const worker = new Worker(workerPath);
      worker.onmessage = (event: MessageEvent) => {
        const data = event.data;
        if (data.type === "ready") {
          this.readyCount++;
          if (this.readyCount === this.numWorkers && this.readyPromiseResolve) {
            this.readyPromiseResolve();
          }
          return;
        }
        const req = this.pending.get(data.id);
        if (req) {
          this.pending.delete(data.id);
          req.resolve(data.fraudCount);
        }
      };
      worker.onerror = (err: ErrorEvent) => {
        console.error("Worker error:", err);
      };
      this.workers.push(worker);
    }
  }

  waitReady(): Promise<void> {
    if (this.readyCount === this.numWorkers) return Promise.resolve();
    return new Promise((resolve) => {
      this.readyPromiseResolve = resolve;
    });
  }

  findFraudCount(query: Uint8Array): Promise<number> {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });

      const worker = this.workers[this.roundRobin % this.numWorkers];
      this.roundRobin++;

      const transferBuffer = query.buffer.slice(0);
      worker.postMessage({ id, query: transferBuffer }, [transferBuffer]);
    });
  }

  terminate() {
    for (const w of this.workers) {
      w.terminate();
    }
  }
}
