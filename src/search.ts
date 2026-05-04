/**
 * Optimized KNN search using quantized vectors for memory efficiency.
 * 
 * Strategy: Quantize 14-dimensional Float32 vectors to Uint8 (0-255 range).
 * This reduces memory from ~168MB to ~42MB for 3M vectors while maintaining
 * sufficient precision for nearest neighbor search.
 * 
 * For values in [-1, 1] range, we map:
 *   -1.0 → 0
 *    0.0 → 128
 *    1.0 → 255
 * 
 * The sentinel value -1 (for missing last_transaction) maps to 0.
 */

const DIMENSIONS = 14;
const K = 5; // Number of nearest neighbors

export interface SearchIndex {
  vectors: Uint8Array;    // Quantized vectors (n * 14 bytes)
  labels: Uint8Array;     // 1 = fraud, 0 = legit
  n: number;
}

/** Quantize a float value [-1, 1] → [0, 255] */
function quantize(value: number): number {
  // Map [-1, 1] to [0, 255]
  const mapped = (value + 1) * 127.5;
  if (mapped < 0) return 0;
  if (mapped > 255) return 255;
  return mapped | 0; // Fast floor
}

/** Quantize a full vector */
export function quantizeVector(vec: Float32Array): Uint8Array {
  const result = new Uint8Array(DIMENSIONS);
  for (let i = 0; i < DIMENSIONS; i++) {
    result[i] = quantize(vec[i]);
  }
  return result;
}

/**
 * Build the search index from reference vectors.
 * Quantizes all vectors to Uint8 for memory efficiency.
 */
export function buildIndex(
  vectors: Float32Array[],
  labels: boolean[]
): SearchIndex {
  const n = vectors.length;
  const quantized = new Uint8Array(n * DIMENSIONS);
  const labelArray = new Uint8Array(n);

  for (let i = 0; i < n; i++) {
    const offset = i * DIMENSIONS;
    for (let d = 0; d < DIMENSIONS; d++) {
      quantized[offset + d] = quantize(vectors[i][d]);
    }
    labelArray[i] = labels[i] ? 1 : 0;
  }

  return { vectors: quantized, labels: labelArray, n };
}

/**
 * Find K nearest neighbors using brute-force with quantized vectors.
 * Optimized with early termination and minimal allocations.
 * 
 * Returns the fraud count among the K nearest neighbors.
 */
export function findFraudCount(index: SearchIndex, query: Uint8Array): number {
  const { vectors, labels, n } = index;

  // Maintain top-K using a simple sorted array (K=5, so overhead is minimal)
  const topDists = new Int32Array(K).fill(0x7FFFFFFF); // Max int32
  const topLabels = new Uint8Array(K);

  for (let i = 0; i < n; i++) {
    const offset = i * DIMENSIONS;

    // Compute squared L2 distance in quantized space (no sqrt needed for comparison)
    let dist = 0;
    for (let d = 0; d < DIMENSIONS; d++) {
      const diff = query[d] - vectors[offset + d];
      dist += diff * diff;
    }

    // Check if this distance is better than the worst in top-K
    if (dist < topDists[K - 1]) {
      // Insert in sorted position
      let pos = K - 1;
      while (pos > 0 && dist < topDists[pos - 1]) {
        topDists[pos] = topDists[pos - 1];
        topLabels[pos] = topLabels[pos - 1];
        pos--;
      }
      topDists[pos] = dist;
      topLabels[pos] = labels[i];
    }
  }

  // Count fraud labels
  let fraudCount = 0;
  for (let i = 0; i < K; i++) {
    fraudCount += topLabels[i];
  }
  return fraudCount;
}

/**
 * Serialize the search index to a binary buffer.
 */
export function serializeIndex(index: SearchIndex): Buffer {
  const headerSize = 8; // n (4 bytes) + reserved (4 bytes)
  const vectorsSize = index.n * DIMENSIONS;
  const labelsSize = index.n;
  const totalSize = headerSize + vectorsSize + labelsSize;

  const buffer = Buffer.alloc(totalSize);
  let offset = 0;

  buffer.writeInt32LE(index.n, offset); offset += 4;
  buffer.writeInt32LE(0, offset); offset += 4; // reserved

  Buffer.from(index.vectors.buffer).copy(buffer, offset, 0, vectorsSize);
  offset += vectorsSize;

  Buffer.from(index.labels.buffer).copy(buffer, offset, 0, labelsSize);

  return buffer;
}

/**
 * Deserialize the search index from a binary buffer.
 */
export function deserializeIndex(buffer: Buffer): SearchIndex {
  let offset = 0;

  const n = buffer.readInt32LE(offset); offset += 4;
  offset += 4; // reserved

  const vectorsSize = n * DIMENSIONS;
  const vectors = new Uint8Array(vectorsSize);
  buffer.copy(Buffer.from(vectors.buffer), 0, offset, offset + vectorsSize);
  offset += vectorsSize;

  const labels = new Uint8Array(n);
  buffer.copy(Buffer.from(labels.buffer), 0, offset, offset + n);

  return { vectors, labels, n };
}
