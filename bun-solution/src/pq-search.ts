/**
 * Product Quantization (PQ) Search - Approach 4
 * 
 * Splits 14 dimensions into M=7 subspaces of 2 dims each.
 * Trains 256 centroids per subspace. Encodes each vector as 7 bytes.
 * At query time, precompute distance tables (7×256) then scan codes.
 * Each vector distance = 7 table lookups + 6 additions (vs 14 mults+subs+adds).
 */

const DIMENSIONS = 14;
const M = 7;         // Number of subspaces
const DSUB = 2;      // Dimensions per subspace
const KS = 256;      // Centroids per subspace (8-bit codes)
const K = 5;         // KNN neighbors

export interface PQIndex {
  codes: Uint8Array;           // n * M bytes (PQ codes)
  labels: Uint8Array;          // n bytes
  codebooks: Float32Array[];   // M codebooks, each KS * DSUB floats
  n: number;
}

/**
 * Train PQ codebooks using K-means on subspace vectors.
 */
function trainCodebook(
  vectors: Uint8Array,
  n: number,
  subspace: number
): Float32Array {
  const codebook = new Float32Array(KS * DSUB);
  const dimStart = subspace * DSUB;

  // Sample for training
  const sampleSize = Math.min(50000, n);
  const samples = new Float32Array(sampleSize * DSUB);
  for (let i = 0; i < sampleSize; i++) {
    const srcIdx = Math.floor(Math.random() * n);
    const srcOffset = srcIdx * DIMENSIONS + dimStart;
    for (let d = 0; d < DSUB; d++) {
      samples[i * DSUB + d] = vectors[srcOffset + d];
    }
  }

  // Random init centroids
  for (let c = 0; c < KS; c++) {
    const idx = Math.floor(Math.random() * sampleSize);
    for (let d = 0; d < DSUB; d++) {
      codebook[c * DSUB + d] = samples[idx * DSUB + d];
    }
  }

  // K-means iterations
  const assignments = new Uint8Array(sampleSize);
  const sums = new Float64Array(KS * DSUB);
  const counts = new Int32Array(KS);

  for (let iter = 0; iter < 20; iter++) {
    // Assign
    for (let i = 0; i < sampleSize; i++) {
      let bestDist = Infinity;
      let bestC = 0;
      for (let c = 0; c < KS; c++) {
        let dist = 0;
        for (let d = 0; d < DSUB; d++) {
          const diff = samples[i * DSUB + d] - codebook[c * DSUB + d];
          dist += diff * diff;
        }
        if (dist < bestDist) { bestDist = dist; bestC = c; }
      }
      assignments[i] = bestC;
    }

    // Update
    sums.fill(0);
    counts.fill(0);
    for (let i = 0; i < sampleSize; i++) {
      const c = assignments[i];
      counts[c]++;
      for (let d = 0; d < DSUB; d++) {
        sums[c * DSUB + d] += samples[i * DSUB + d];
      }
    }
    for (let c = 0; c < KS; c++) {
      if (counts[c] > 0) {
        for (let d = 0; d < DSUB; d++) {
          codebook[c * DSUB + d] = sums[c * DSUB + d] / counts[c];
        }
      }
    }
  }

  return codebook;
}

/**
 * Build PQ index: train codebooks, encode all vectors.
 */
export function buildPQIndex(vectors: Uint8Array, labels: Uint8Array, n: number): PQIndex {
  const codebooks: Float32Array[] = [];
  const codes = new Uint8Array(n * M);

  // Train codebooks
  for (let m = 0; m < M; m++) {
    console.log(`  Training codebook ${m + 1}/${M}...`);
    const cb = trainCodebook(vectors, n, m);
    codebooks.push(cb);
  }

  // Encode all vectors
  console.log(`  Encoding ${n} vectors...`);
  for (let i = 0; i < n; i++) {
    for (let m = 0; m < M; m++) {
      const dimStart = m * DSUB;
      const vOffset = i * DIMENSIONS + dimStart;
      const cb = codebooks[m];

      let bestDist = Infinity;
      let bestC = 0;
      for (let c = 0; c < KS; c++) {
        let dist = 0;
        for (let d = 0; d < DSUB; d++) {
          const diff = vectors[vOffset + d] - cb[c * DSUB + d];
          dist += diff * diff;
        }
        if (dist < bestDist) { bestDist = dist; bestC = c; }
      }
      codes[i * M + m] = bestC;
    }
  }

  return { codes, labels: new Uint8Array(labels), codebooks, n };
}

/**
 * PQ search using Asymmetric Distance Computation (ADC).
 * Query is NOT quantized - we compute exact subvector distances to centroids.
 */
export function findFraudCountPQ(index: PQIndex, query: Uint8Array): number {
  const { codes, labels, codebooks, n } = index;

  // Precompute distance tables: for each subspace, distance from query subvector to each centroid
  const distTable = new Float32Array(M * KS);
  for (let m = 0; m < M; m++) {
    const dimStart = m * DSUB;
    const cb = codebooks[m];
    const tableOffset = m * KS;

    for (let c = 0; c < KS; c++) {
      let dist = 0;
      for (let d = 0; d < DSUB; d++) {
        const diff = query[dimStart + d] - cb[c * DSUB + d];
        dist += diff * diff;
      }
      distTable[tableOffset + c] = dist;
    }
  }

  // Scan all codes, using table lookups for distance
  const topDists = new Float32Array(K).fill(Infinity);
  const topLabels = new Uint8Array(K);

  for (let i = 0; i < n; i++) {
    const codeOffset = i * M;
    let dist = 0;

    // 7 table lookups + 6 additions
    for (let m = 0; m < M; m++) {
      dist += distTable[m * KS + codes[codeOffset + m]];
    }

    if (dist < topDists[K - 1]) {
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

  let fraudCount = 0;
  for (let i = 0; i < K; i++) {
    fraudCount += topLabels[i];
  }
  return fraudCount;
}

/**
 * Serialize PQ index.
 */
export function serializePQIndex(index: PQIndex): Buffer {
  const headerSize = 16;
  const codesSize = index.n * M;
  const labelsSize = index.n;
  const codebooksSize = M * KS * DSUB * 4; // Float32
  const totalSize = headerSize + codesSize + labelsSize + codebooksSize;

  const buffer = Buffer.alloc(totalSize);
  let off = 0;

  buffer.writeInt32LE(index.n, off); off += 4;
  buffer.writeInt32LE(M, off); off += 4;
  buffer.writeInt32LE(KS, off); off += 4;
  buffer.writeInt32LE(DSUB, off); off += 4;

  Buffer.from(index.codes.buffer).copy(buffer, off, 0, codesSize); off += codesSize;
  Buffer.from(index.labels.buffer).copy(buffer, off, 0, labelsSize); off += labelsSize;

  for (let m = 0; m < M; m++) {
    const cbSize = KS * DSUB * 4;
    Buffer.from(index.codebooks[m].buffer).copy(buffer, off, 0, cbSize);
    off += cbSize;
  }

  return buffer;
}

/**
 * Deserialize PQ index.
 */
export function deserializePQIndex(buffer: Buffer): PQIndex {
  let off = 0;

  const n = buffer.readInt32LE(off); off += 4;
  const m = buffer.readInt32LE(off); off += 4;
  const ks = buffer.readInt32LE(off); off += 4;
  const dsub = buffer.readInt32LE(off); off += 4;

  const codesSize = n * m;
  const codes = new Uint8Array(codesSize);
  buffer.copy(Buffer.from(codes.buffer), 0, off, off + codesSize); off += codesSize;

  const labels = new Uint8Array(n);
  buffer.copy(Buffer.from(labels.buffer), 0, off, off + n); off += n;

  const codebooks: Float32Array[] = [];
  for (let i = 0; i < m; i++) {
    const cbSize = ks * dsub * 4;
    const cbBuf = new ArrayBuffer(cbSize);
    buffer.copy(Buffer.from(cbBuf), 0, off, off + cbSize);
    codebooks.push(new Float32Array(cbBuf));
    off += cbSize;
  }

  return { codes, labels, codebooks, n };
}
