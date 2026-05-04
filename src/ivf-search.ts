/**
 * IVF (Inverted File Index) Search - Approach 1
 * 
 * Strategy: Cluster 3M vectors into ~1024 centroids using K-means.
 * At query time, only search vectors in the nProbe nearest clusters.
 * This reduces search from 3M to ~60K vectors per query (~30-50x speedup).
 */

const DIMENSIONS = 14;
const K = 5; // Number of nearest neighbors

export interface IVFIndex {
  centroids: Uint8Array;       // nClusters * 14 bytes
  vectors: Uint8Array;         // n * 14 bytes (reordered by cluster)
  labels: Uint8Array;          // n bytes (reordered by cluster)
  clusterOffsets: Int32Array;  // nClusters + 1 (start offset for each cluster)
  nClusters: number;
  n: number;
  nProbe: number;
}

/**
 * Simple K-means on Uint8 vectors. Uses random initialization + Lloyd's iterations.
 */
function kmeans(
  vectors: Uint8Array,
  n: number,
  nClusters: number,
  maxIter: number = 15
): { centroids: Uint8Array; assignments: Int32Array } {
  const centroids = new Uint8Array(nClusters * DIMENSIONS);
  const assignments = new Int32Array(n);

  // Random initialization: pick nClusters random vectors as initial centroids
  const used = new Set<number>();
  for (let c = 0; c < nClusters; c++) {
    let idx: number;
    do {
      idx = Math.floor(Math.random() * n);
    } while (used.has(idx));
    used.add(idx);
    const srcOffset = idx * DIMENSIONS;
    const dstOffset = c * DIMENSIONS;
    for (let d = 0; d < DIMENSIONS; d++) {
      centroids[dstOffset + d] = vectors[srcOffset + d];
    }
  }

  // Accumulator arrays for centroid updates
  const sums = new Float64Array(nClusters * DIMENSIONS);
  const counts = new Int32Array(nClusters);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign each vector to nearest centroid
    for (let i = 0; i < n; i++) {
      const vOffset = i * DIMENSIONS;
      let bestDist = Infinity;
      let bestC = 0;

      for (let c = 0; c < nClusters; c++) {
        const cOffset = c * DIMENSIONS;
        let dist = 0;
        for (let d = 0; d < DIMENSIONS; d++) {
          const diff = vectors[vOffset + d] - centroids[cOffset + d];
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestC = c;
        }
      }
      assignments[i] = bestC;
    }

    // Update centroids
    sums.fill(0);
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      counts[c]++;
      const vOffset = i * DIMENSIONS;
      const cOffset = c * DIMENSIONS;
      for (let d = 0; d < DIMENSIONS; d++) {
        sums[cOffset + d] += vectors[vOffset + d];
      }
    }

    for (let c = 0; c < nClusters; c++) {
      const cOffset = c * DIMENSIONS;
      if (counts[c] > 0) {
        for (let d = 0; d < DIMENSIONS; d++) {
          centroids[cOffset + d] = Math.round(sums[cOffset + d] / counts[c]);
        }
      }
    }

    console.log(`  K-means iter ${iter + 1}/${maxIter}, cluster sizes: min=${Math.min(...counts)}, max=${Math.max(...counts)}`);
  }

  return { centroids, assignments };
}

/**
 * Build IVF index from quantized vectors.
 */
export function buildIVFIndex(
  vectors: Uint8Array,
  labels: Uint8Array,
  n: number,
  nClusters: number = 1024,
  nProbe: number = 20
): IVFIndex {
  console.log(`Building IVF index: ${n} vectors, ${nClusters} clusters...`);

  // Run K-means on a subsample for speed (use all vectors for assignment)
  const sampleSize = Math.min(200000, n);
  const sampleVectors = new Uint8Array(sampleSize * DIMENSIONS);
  const sampleIndices = new Set<number>();
  
  while (sampleIndices.size < sampleSize) {
    sampleIndices.add(Math.floor(Math.random() * n));
  }
  
  let si = 0;
  for (const idx of sampleIndices) {
    const srcOffset = idx * DIMENSIONS;
    const dstOffset = si * DIMENSIONS;
    for (let d = 0; d < DIMENSIONS; d++) {
      sampleVectors[dstOffset + d] = vectors[srcOffset + d];
    }
    si++;
  }

  // Train centroids on subsample
  console.log(`  Training K-means on ${sampleSize} samples...`);
  const { centroids } = kmeans(sampleVectors, sampleSize, nClusters, 15);

  // Assign ALL vectors to nearest centroid
  console.log(`  Assigning ${n} vectors to clusters...`);
  const assignments = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const vOffset = i * DIMENSIONS;
    let bestDist = Infinity;
    let bestC = 0;
    for (let c = 0; c < nClusters; c++) {
      const cOffset = c * DIMENSIONS;
      let dist = 0;
      for (let d = 0; d < DIMENSIONS; d++) {
        const diff = vectors[vOffset + d] - centroids[cOffset + d];
        dist += diff * diff;
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestC = c;
      }
    }
    assignments[i] = bestC;
  }

  // Count cluster sizes
  const clusterCounts = new Int32Array(nClusters);
  for (let i = 0; i < n; i++) {
    clusterCounts[assignments[i]]++;
  }

  // Build cluster offsets
  const clusterOffsets = new Int32Array(nClusters + 1);
  for (let c = 0; c < nClusters; c++) {
    clusterOffsets[c + 1] = clusterOffsets[c] + clusterCounts[c];
  }

  // Reorder vectors and labels by cluster
  const reorderedVectors = new Uint8Array(n * DIMENSIONS);
  const reorderedLabels = new Uint8Array(n);
  const clusterPos = new Int32Array(nClusters); // current write position per cluster

  for (let i = 0; i < n; i++) {
    const c = assignments[i];
    const writeIdx = clusterOffsets[c] + clusterPos[c];
    clusterPos[c]++;

    const srcOffset = i * DIMENSIONS;
    const dstOffset = writeIdx * DIMENSIONS;
    for (let d = 0; d < DIMENSIONS; d++) {
      reorderedVectors[dstOffset + d] = vectors[srcOffset + d];
    }
    reorderedLabels[writeIdx] = labels[i];
  }

  return {
    centroids,
    vectors: reorderedVectors,
    labels: reorderedLabels,
    clusterOffsets,
    nClusters,
    n,
    nProbe,
  };
}

/**
 * Find K nearest neighbors using IVF.
 * Only searches vectors in the nProbe nearest clusters.
 */
export function findFraudCountIVF(index: IVFIndex, query: Uint8Array): number {
  const { centroids, vectors, labels, clusterOffsets, nClusters, nProbe } = index;

  // Find nProbe nearest centroids
  const centroidDists: Array<{ cluster: number; dist: number }> = new Array(nClusters);
  for (let c = 0; c < nClusters; c++) {
    const cOffset = c * DIMENSIONS;
    let dist = 0;
    for (let d = 0; d < DIMENSIONS; d++) {
      const diff = query[d] - centroids[cOffset + d];
      dist += diff * diff;
    }
    centroidDists[c] = { cluster: c, dist };
  }

  // Partial sort: find nProbe smallest
  centroidDists.sort((a, b) => a.dist - b.dist);

  // Search vectors in nearest nProbe clusters
  const topDists = new Int32Array(K).fill(0x7FFFFFFF);
  const topLabels = new Uint8Array(K);

  for (let p = 0; p < nProbe; p++) {
    const c = centroidDists[p].cluster;
    const start = clusterOffsets[c];
    const end = clusterOffsets[c + 1];

    for (let i = start; i < end; i++) {
      const offset = i * DIMENSIONS;
      let dist = 0;
      for (let d = 0; d < DIMENSIONS; d++) {
        const diff = query[d] - vectors[offset + d];
        dist += diff * diff;
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
  }

  let fraudCount = 0;
  for (let i = 0; i < K; i++) {
    fraudCount += topLabels[i];
  }
  return fraudCount;
}

/**
 * Serialize IVF index to binary.
 */
export function serializeIVFIndex(index: IVFIndex): Buffer {
  const headerSize = 16; // n, nClusters, nProbe, reserved
  const centroidsSize = index.nClusters * DIMENSIONS;
  const vectorsSize = index.n * DIMENSIONS;
  const labelsSize = index.n;
  const offsetsSize = (index.nClusters + 1) * 4;
  const totalSize = headerSize + centroidsSize + vectorsSize + labelsSize + offsetsSize;

  const buffer = Buffer.alloc(totalSize);
  let off = 0;

  buffer.writeInt32LE(index.n, off); off += 4;
  buffer.writeInt32LE(index.nClusters, off); off += 4;
  buffer.writeInt32LE(index.nProbe, off); off += 4;
  buffer.writeInt32LE(0, off); off += 4;

  Buffer.from(index.centroids.buffer).copy(buffer, off, 0, centroidsSize); off += centroidsSize;
  Buffer.from(index.vectors.buffer).copy(buffer, off, 0, vectorsSize); off += vectorsSize;
  Buffer.from(index.labels.buffer).copy(buffer, off, 0, labelsSize); off += labelsSize;
  Buffer.from(index.clusterOffsets.buffer).copy(buffer, off, 0, offsetsSize);

  return buffer;
}

/**
 * Deserialize IVF index from binary.
 */
export function deserializeIVFIndex(buffer: Buffer): IVFIndex {
  let off = 0;

  const n = buffer.readInt32LE(off); off += 4;
  const nClusters = buffer.readInt32LE(off); off += 4;
  const nProbe = buffer.readInt32LE(off); off += 4;
  off += 4; // reserved

  const centroidsSize = nClusters * DIMENSIONS;
  const centroids = new Uint8Array(centroidsSize);
  buffer.copy(Buffer.from(centroids.buffer), 0, off, off + centroidsSize); off += centroidsSize;

  const vectorsSize = n * DIMENSIONS;
  const vectors = new Uint8Array(vectorsSize);
  buffer.copy(Buffer.from(vectors.buffer), 0, off, off + vectorsSize); off += vectorsSize;

  const labels = new Uint8Array(n);
  buffer.copy(Buffer.from(labels.buffer), 0, off, off + n); off += n;

  const offsetsSize = (nClusters + 1) * 4;
  const offsetsBuf = new ArrayBuffer(offsetsSize);
  buffer.copy(Buffer.from(offsetsBuf), 0, off, off + offsetsSize);
  const clusterOffsets = new Int32Array(offsetsBuf);

  return { centroids, vectors, labels, clusterOffsets, nClusters, n, nProbe };
}
