/**
 * LSH (Locality-Sensitive Hashing) Search - Approach 5
 * 
 * Uses random hyperplane projections to hash vectors into buckets.
 * At query time, look up candidate buckets and do exact KNN on candidates only.
 * 
 * Config: L=4 tables, K=10 bits per hash (1024 buckets per table).
 * Multi-probe: check adjacent buckets (flip 1 bit at a time) for better recall.
 */

const DIMENSIONS = 14;
const KNN = 5;

export interface LSHIndex {
  vectors: Uint8Array;          // n * 14 bytes
  labels: Uint8Array;           // n bytes
  // L hash tables, each with up to 2^K buckets
  // bucketIndices[table][bucket] = array of vector indices in that bucket
  bucketIndices: Int32Array[];  // Flattened bucket data per table
  bucketOffsets: Int32Array[];  // Offsets into bucketIndices per table
  // Hash functions: L * K random projections (each 14-dim)
  projections: Float32Array;    // L * K * 14 floats
  thresholds: Float32Array;     // L * K thresholds (medians)
  L: number;
  K: number;
  n: number;
}

/**
 * Compute hash for a vector in a given table.
 */
function computeHash(
  vec: Uint8Array, 
  projections: Float32Array, 
  thresholds: Float32Array,
  table: number, 
  K: number
): number {
  let hash = 0;
  const tableOffset = table * K * DIMENSIONS;
  const threshOffset = table * K;

  for (let k = 0; k < K; k++) {
    const projOffset = tableOffset + k * DIMENSIONS;
    let dot = 0;
    for (let d = 0; d < DIMENSIONS; d++) {
      dot += vec[d] * projections[projOffset + d];
    }
    if (dot >= thresholds[threshOffset + k]) {
      hash |= (1 << k);
    }
  }
  return hash;
}

/**
 * Build LSH index.
 */
export function buildLSHIndex(
  vectors: Uint8Array,
  labels: Uint8Array,
  n: number,
  L: number = 4,
  K: number = 10
): LSHIndex {
  const nBuckets = 1 << K; // 2^K buckets per table

  // Generate random projections (Gaussian)
  const projections = new Float32Array(L * K * DIMENSIONS);
  for (let i = 0; i < projections.length; i++) {
    // Box-Muller transform for Gaussian
    const u1 = Math.random();
    const u2 = Math.random();
    projections[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // Compute thresholds (median of projections for each hash function)
  console.log(`  Computing thresholds...`);
  const thresholds = new Float32Array(L * K);
  const sampleSize = Math.min(50000, n);

  for (let t = 0; t < L; t++) {
    for (let k = 0; k < K; k++) {
      const projOffset = (t * K + k) * DIMENSIONS;
      const dots = new Float32Array(sampleSize);
      for (let i = 0; i < sampleSize; i++) {
        const idx = Math.floor(Math.random() * n);
        const vOffset = idx * DIMENSIONS;
        let dot = 0;
        for (let d = 0; d < DIMENSIONS; d++) {
          dot += vectors[vOffset + d] * projections[projOffset + d];
        }
        dots[i] = dot;
      }
      dots.sort();
      thresholds[t * K + k] = dots[sampleSize >> 1]; // median
    }
  }

  // Hash all vectors into buckets
  console.log(`  Hashing ${n} vectors into ${L} tables...`);
  const bucketLists: number[][] = [];
  for (let t = 0; t < L; t++) {
    const tableBuckets: number[][] = new Array(nBuckets);
    for (let b = 0; b < nBuckets; b++) tableBuckets[b] = [];

    for (let i = 0; i < n; i++) {
      const vOffset = i * DIMENSIONS;
      const vec = vectors.subarray(vOffset, vOffset + DIMENSIONS);
      const hash = computeHash(vec as unknown as Uint8Array, projections, thresholds, t, K);
      tableBuckets[hash].push(i);
    }

    // Flatten to typed arrays
    let totalInTable = 0;
    for (let b = 0; b < nBuckets; b++) totalInTable += tableBuckets[b].length;

    const offsets = new Int32Array(nBuckets + 1);
    const indices = new Int32Array(totalInTable);
    let pos = 0;
    for (let b = 0; b < nBuckets; b++) {
      offsets[b] = pos;
      for (const idx of tableBuckets[b]) {
        indices[pos++] = idx;
      }
    }
    offsets[nBuckets] = pos;

    bucketLists.push(...[]);
    // Store in result format
    if (!bucketLists.length) bucketLists.length = 0;
  }

  // Re-do with proper storage
  const bucketIndicesArr: Int32Array[] = [];
  const bucketOffsetsArr: Int32Array[] = [];

  for (let t = 0; t < L; t++) {
    const tableBuckets: number[][] = new Array(nBuckets);
    for (let b = 0; b < nBuckets; b++) tableBuckets[b] = [];

    for (let i = 0; i < n; i++) {
      const vOffset = i * DIMENSIONS;
      const vec = vectors.subarray(vOffset, vOffset + DIMENSIONS);
      const hash = computeHash(vec as unknown as Uint8Array, projections, thresholds, t, K);
      tableBuckets[hash].push(i);
    }

    let totalInTable = 0;
    for (let b = 0; b < nBuckets; b++) totalInTable += tableBuckets[b].length;

    const offsets = new Int32Array(nBuckets + 1);
    const indices = new Int32Array(totalInTable);
    let pos = 0;
    for (let b = 0; b < nBuckets; b++) {
      offsets[b] = pos;
      for (const idx of tableBuckets[b]) {
        indices[pos++] = idx;
      }
    }
    offsets[nBuckets] = pos;

    bucketOffsetsArr.push(offsets);
    bucketIndicesArr.push(indices);

    const nonEmpty = offsets.filter((_, i) => i < nBuckets && offsets[i + 1] - offsets[i] > 0).length;
    console.log(`  Table ${t + 1}: ${nonEmpty} non-empty buckets, ${totalInTable} entries`);
  }

  return {
    vectors: new Uint8Array(vectors),
    labels: new Uint8Array(labels),
    bucketIndices: bucketIndicesArr,
    bucketOffsets: bucketOffsetsArr,
    projections,
    thresholds,
    L, K, n
  };
}

/**
 * LSH search with multi-probe.
 */
export function findFraudCountLSH(index: LSHIndex, query: Uint8Array): number {
  const { vectors, labels, bucketIndices, bucketOffsets, projections, thresholds, L, K, n } = index;

  // Collect candidates from all tables + multi-probe
  const candidateSet = new Set<number>();

  for (let t = 0; t < L; t++) {
    const hash = computeHash(query, projections, thresholds, t, K);
    
    // Main bucket
    const offsets = bucketOffsets[t];
    const indices = bucketIndices[t];
    const start = offsets[hash];
    const end = offsets[hash + 1];
    for (let i = start; i < end; i++) {
      candidateSet.add(indices[i]);
    }

    // Multi-probe: flip each bit
    for (let bit = 0; bit < K; bit++) {
      const probeHash = hash ^ (1 << bit);
      const pStart = offsets[probeHash];
      const pEnd = offsets[probeHash + 1];
      for (let i = pStart; i < pEnd; i++) {
        candidateSet.add(indices[i]);
      }
    }
  }

  // If too few candidates, do brute force (safety net)
  if (candidateSet.size < KNN) {
    // Fallback to brute force on small portion
    const topDists = new Int32Array(KNN).fill(0x7FFFFFFF);
    const topLabels = new Uint8Array(KNN);
    for (let i = 0; i < n; i++) {
      const offset = i * DIMENSIONS;
      let dist = 0;
      for (let d = 0; d < DIMENSIONS; d++) {
        const diff = query[d] - vectors[offset + d];
        dist += diff * diff;
      }
      if (dist < topDists[KNN - 1]) {
        let pos = KNN - 1;
        while (pos > 0 && dist < topDists[pos - 1]) {
          topDists[pos] = topDists[pos - 1];
          topLabels[pos] = topLabels[pos - 1];
          pos--;
        }
        topDists[pos] = dist;
        topLabels[pos] = labels[i];
      }
    }
    let fc = 0;
    for (let i = 0; i < KNN; i++) fc += topLabels[i];
    return fc;
  }

  // Exact KNN on candidates
  const topDists = new Int32Array(KNN).fill(0x7FFFFFFF);
  const topLabels = new Uint8Array(KNN);

  for (const idx of candidateSet) {
    const offset = idx * DIMENSIONS;
    let dist = 0;
    for (let d = 0; d < DIMENSIONS; d++) {
      const diff = query[d] - vectors[offset + d];
      dist += diff * diff;
    }
    if (dist < topDists[KNN - 1]) {
      let pos = KNN - 1;
      while (pos > 0 && dist < topDists[pos - 1]) {
        topDists[pos] = topDists[pos - 1];
        topLabels[pos] = topLabels[pos - 1];
        pos--;
      }
      topDists[pos] = dist;
      topLabels[pos] = labels[idx];
    }
  }

  let fraudCount = 0;
  for (let i = 0; i < KNN; i++) fraudCount += topLabels[i];
  return fraudCount;
}

/**
 * Serialize LSH index.
 */
export function serializeLSHIndex(index: LSHIndex): Buffer {
  const headerSize = 16;
  const vectorsSize = index.n * DIMENSIONS;
  const labelsSize = index.n;
  const projectionsSize = index.L * index.K * DIMENSIONS * 4;
  const thresholdsSize = index.L * index.K * 4;

  // Calculate bucket data sizes
  let bucketDataSize = 0;
  for (let t = 0; t < index.L; t++) {
    bucketDataSize += (((1 << index.K) + 1) * 4); // offsets
    bucketDataSize += (index.bucketIndices[t].length * 4); // indices
    bucketDataSize += 4; // length prefix for indices
  }

  const totalSize = headerSize + vectorsSize + labelsSize + projectionsSize + thresholdsSize + bucketDataSize;
  const buffer = Buffer.alloc(totalSize);
  let off = 0;

  buffer.writeInt32LE(index.n, off); off += 4;
  buffer.writeInt32LE(index.L, off); off += 4;
  buffer.writeInt32LE(index.K, off); off += 4;
  buffer.writeInt32LE(0, off); off += 4;

  Buffer.from(index.vectors.buffer).copy(buffer, off, 0, vectorsSize); off += vectorsSize;
  Buffer.from(index.labels.buffer).copy(buffer, off, 0, labelsSize); off += labelsSize;
  Buffer.from(index.projections.buffer).copy(buffer, off, 0, projectionsSize); off += projectionsSize;
  Buffer.from(index.thresholds.buffer).copy(buffer, off, 0, thresholdsSize); off += thresholdsSize;

  // Bucket data per table
  const nBuckets = 1 << index.K;
  for (let t = 0; t < index.L; t++) {
    const offsetsSize = (nBuckets + 1) * 4;
    Buffer.from(index.bucketOffsets[t].buffer).copy(buffer, off, 0, offsetsSize); off += offsetsSize;
    
    const indicesLen = index.bucketIndices[t].length;
    buffer.writeInt32LE(indicesLen, off); off += 4;
    Buffer.from(index.bucketIndices[t].buffer).copy(buffer, off, 0, indicesLen * 4); off += indicesLen * 4;
  }

  return buffer;
}

/**
 * Deserialize LSH index.
 */
export function deserializeLSHIndex(buffer: Buffer): LSHIndex {
  let off = 0;

  const n = buffer.readInt32LE(off); off += 4;
  const L = buffer.readInt32LE(off); off += 4;
  const K = buffer.readInt32LE(off); off += 4;
  off += 4;

  const vectorsSize = n * DIMENSIONS;
  const vectors = new Uint8Array(vectorsSize);
  buffer.copy(Buffer.from(vectors.buffer), 0, off, off + vectorsSize); off += vectorsSize;

  const labels = new Uint8Array(n);
  buffer.copy(Buffer.from(labels.buffer), 0, off, off + n); off += n;

  const projectionsSize = L * K * DIMENSIONS * 4;
  const projBuf = new ArrayBuffer(projectionsSize);
  buffer.copy(Buffer.from(projBuf), 0, off, off + projectionsSize);
  const projections = new Float32Array(projBuf);
  off += projectionsSize;

  const thresholdsSize = L * K * 4;
  const threshBuf = new ArrayBuffer(thresholdsSize);
  buffer.copy(Buffer.from(threshBuf), 0, off, off + thresholdsSize);
  const thresholds = new Float32Array(threshBuf);
  off += thresholdsSize;

  const nBuckets = 1 << K;
  const bucketOffsets: Int32Array[] = [];
  const bucketIndices: Int32Array[] = [];

  for (let t = 0; t < L; t++) {
    const offsetsSize = (nBuckets + 1) * 4;
    const offsetsBuf = new ArrayBuffer(offsetsSize);
    buffer.copy(Buffer.from(offsetsBuf), 0, off, off + offsetsSize);
    bucketOffsets.push(new Int32Array(offsetsBuf));
    off += offsetsSize;

    const indicesLen = buffer.readInt32LE(off); off += 4;
    const indicesBuf = new ArrayBuffer(indicesLen * 4);
    buffer.copy(Buffer.from(indicesBuf), 0, off, off + indicesLen * 4);
    bucketIndices.push(new Int32Array(indicesBuf));
    off += indicesLen * 4;
  }

  return { vectors, labels, bucketIndices, bucketOffsets, projections, thresholds, L, K, n };
}
