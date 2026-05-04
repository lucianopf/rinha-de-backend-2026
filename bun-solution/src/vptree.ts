/**
 * VP-Tree (Vantage Point Tree) implementation for exact nearest neighbor search.
 * 
 * VP-Tree is ideal for this use case because:
 * - It provides exact KNN results (no approximation error)
 * - It's much faster than brute force O(N*D) for each query
 * - With 14 dimensions, it still provides good pruning
 * - It can be pre-built and serialized for fast startup
 */

const DIMENSIONS = 14;

export interface ReferenceVector {
  vector: Float32Array;
  isFraud: boolean;
}

interface VPNode {
  point: number; // index into vectors array
  radius: number;
  left: VPNode | null;
  right: VPNode | null;
}

/** Compute squared Euclidean distance between two vectors */
function distanceSquared(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < DIMENSIONS; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

/** Compute Euclidean distance */
export function distance(a: Float32Array, b: Float32Array): number {
  return Math.sqrt(distanceSquared(a, b));
}

interface Neighbor {
  index: number;
  dist: number;
}

/**
 * Flat VP-Tree stored in typed arrays for cache efficiency.
 * Uses a pre-allocated array layout similar to a binary heap.
 */
export class VPTree {
  private vectors: Float32Array; // all vectors flattened (N * 14)
  private labels: Uint8Array;   // 1 = fraud, 0 = legit
  private nodePoints: Int32Array;  // point index for each node
  private nodeRadii: Float32Array; // radius for each node
  private nodeLeft: Int32Array;    // left child index (-1 = none)
  private nodeRight: Int32Array;   // right child index (-1 = none)
  private nodeCount: number;
  private readonly n: number;

  constructor(references: ReferenceVector[]) {
    this.n = references.length;

    // Flatten vectors into a single Float32Array for cache locality
    this.vectors = new Float32Array(this.n * DIMENSIONS);
    this.labels = new Uint8Array(this.n);
    for (let i = 0; i < this.n; i++) {
      this.vectors.set(references[i].vector, i * DIMENSIONS);
      this.labels[i] = references[i].isFraud ? 1 : 0;
    }

    // Allocate node arrays (max nodes = n)
    this.nodePoints = new Int32Array(this.n);
    this.nodeRadii = new Float32Array(this.n);
    this.nodeLeft = new Int32Array(this.n).fill(-1);
    this.nodeRight = new Int32Array(this.n).fill(-1);
    this.nodeCount = 0;

    // Build the tree
    const indices = new Int32Array(this.n);
    for (let i = 0; i < this.n; i++) indices[i] = i;
    this.buildNode(indices, 0, this.n);
  }

  private getVector(index: number): Float32Array {
    return this.vectors.subarray(index * DIMENSIONS, (index + 1) * DIMENSIONS);
  }

  private distBetween(a: number, b: number): number {
    let sum = 0;
    const offsetA = a * DIMENSIONS;
    const offsetB = b * DIMENSIONS;
    for (let i = 0; i < DIMENSIONS; i++) {
      const d = this.vectors[offsetA + i] - this.vectors[offsetB + i];
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  private distToQuery(queryVec: Float32Array, b: number): number {
    let sum = 0;
    const offsetB = b * DIMENSIONS;
    for (let i = 0; i < DIMENSIONS; i++) {
      const d = queryVec[i] - this.vectors[offsetB + i];
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  private buildNode(indices: Int32Array, start: number, end: number): number {
    if (start >= end) return -1;

    const nodeIdx = this.nodeCount++;

    if (end - start === 1) {
      this.nodePoints[nodeIdx] = indices[start];
      this.nodeRadii[nodeIdx] = 0;
      return nodeIdx;
    }

    // Choose vantage point: use random selection for better balance
    const vpIdx = start + Math.floor(Math.random() * (end - start));
    // Swap vantage point to start
    const tmp = indices[start];
    indices[start] = indices[vpIdx];
    indices[vpIdx] = tmp;

    const vpPoint = indices[start];
    this.nodePoints[nodeIdx] = vpPoint;

    // Compute distances from vantage point to all other points
    const count = end - start - 1;
    const dists = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      dists[i] = this.distBetween(vpPoint, indices[start + 1 + i]);
    }

    // Find median distance using partial sort (nth_element equivalent)
    const median = this.selectMedian(dists, indices, start + 1, end);
    this.nodeRadii[nodeIdx] = median;

    // Partition: points with dist <= median go left, > median go right
    const mid = start + 1 + Math.floor(count / 2);

    this.nodeLeft[nodeIdx] = this.buildNode(indices, start + 1, mid);
    this.nodeRight[nodeIdx] = this.buildNode(indices, mid, end);

    return nodeIdx;
  }

  private selectMedian(
    dists: Float32Array,
    indices: Int32Array,
    start: number,
    end: number
  ): number {
    const count = end - start;
    // Create index array for sorting
    const order = new Int32Array(count);
    for (let i = 0; i < count; i++) order[i] = i;

    // Sort by distance
    order.sort((a, b) => dists[a] - dists[b]);

    // Reorder indices according to sort
    const tempIndices = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      tempIndices[i] = indices[start + order[i]];
    }
    for (let i = 0; i < count; i++) {
      indices[start + i] = tempIndices[i];
    }

    // Return median distance
    return dists[order[Math.floor(count / 2)]];
  }

  /**
   * Find K nearest neighbors to the query vector.
   * Uses iterative approach with explicit stack to avoid stack overflow with 3M nodes.
   * Returns array of {index, dist} sorted by distance.
   */
  knn(query: Float32Array, k: number): Neighbor[] {
    const neighbors: Neighbor[] = [];
    let tau = Infinity; // current furthest distance in neighbors

    // Iterative DFS with explicit stack
    const stack: number[] = [0];

    while (stack.length > 0) {
      const nodeIdx = stack.pop()!;
      if (nodeIdx === -1) continue;

      const point = this.nodePoints[nodeIdx];
      const dist = this.distToQuery(query, point);

      // Check if this point should be added to neighbors
      if (neighbors.length < k) {
        neighbors.push({ index: point, dist });
        if (neighbors.length === k) {
          neighbors.sort((a, b) => a.dist - b.dist);
          tau = neighbors[k - 1].dist;
        }
      } else if (dist < tau) {
        neighbors[k - 1] = { index: point, dist };
        neighbors.sort((a, b) => a.dist - b.dist);
        tau = neighbors[k - 1].dist;
      }

      const radius = this.nodeRadii[nodeIdx];
      const left = this.nodeLeft[nodeIdx];
      const right = this.nodeRight[nodeIdx];

      // Decide which subtrees to search (push in reverse order for DFS priority)
      if (dist < radius) {
        // Query is inside the radius - search left first (push right, then left)
        if (dist + tau >= radius && right !== -1) {
          stack.push(right);
        }
        if (dist - tau < radius && left !== -1) {
          stack.push(left);
        }
      } else {
        // Query is outside the radius - search right first (push left, then right)
        if (dist - tau < radius && left !== -1) {
          stack.push(left);
        }
        if (dist + tau >= radius && right !== -1) {
          stack.push(right);
        }
      }
    }

    // Final sort
    neighbors.sort((a, b) => a.dist - b.dist);
    return neighbors;
  }

  /**
   * Compute fraud score for a query vector.
   * Returns fraction of fraud labels among 5 nearest neighbors.
   */
  fraudScore(query: Float32Array): number {
    const neighbors = this.knn(query, 5);
    let fraudCount = 0;
    for (const n of neighbors) {
      if (this.labels[n.index]) fraudCount++;
    }
    return fraudCount / 5;
  }

  /**
   * Count fraud labels among 5 nearest neighbors.
   * Returns integer 0-5 (avoids floating point division in hot path).
   */
  fraudCount(query: Float32Array): number {
    const neighbors = this.knn(query, 5);
    let count = 0;
    for (const n of neighbors) {
      if (this.labels[n.index]) count++;
    }
    return count;
  }

  /** Get the label of a reference by index */
  getLabel(index: number): boolean {
    return this.labels[index] === 1;
  }

  /** Serialize tree structure for faster loading */
  serialize(): Buffer {
    const headerSize = 4 * 4; // n, nodeCount, + 2 reserved
    const vectorsSize = this.n * DIMENSIONS * 4;
    const labelsSize = this.n;
    const nodesSize = this.nodeCount * (4 + 4 + 4 + 4); // point, radius, left, right

    const totalSize = headerSize + vectorsSize + labelsSize + nodesSize;
    const buffer = Buffer.alloc(totalSize);
    let offset = 0;

    // Header
    buffer.writeInt32LE(this.n, offset); offset += 4;
    buffer.writeInt32LE(this.nodeCount, offset); offset += 4;
    buffer.writeInt32LE(0, offset); offset += 4; // reserved
    buffer.writeInt32LE(0, offset); offset += 4; // reserved

    // Vectors
    Buffer.from(this.vectors.buffer).copy(buffer, offset);
    offset += vectorsSize;

    // Labels
    Buffer.from(this.labels.buffer).copy(buffer, offset);
    offset += labelsSize;

    // Node data
    for (let i = 0; i < this.nodeCount; i++) {
      buffer.writeInt32LE(this.nodePoints[i], offset); offset += 4;
      buffer.writeFloatLE(this.nodeRadii[i], offset); offset += 4;
      buffer.writeInt32LE(this.nodeLeft[i], offset); offset += 4;
      buffer.writeInt32LE(this.nodeRight[i], offset); offset += 4;
    }

    return buffer;
  }

  /** Deserialize from buffer */
  static deserialize(buffer: Buffer): VPTree {
    const tree = Object.create(VPTree.prototype) as VPTree;
    let offset = 0;

    // Header
    const n = buffer.readInt32LE(offset); offset += 4;
    const nodeCount = buffer.readInt32LE(offset); offset += 4;
    offset += 8; // reserved

    (tree as any).n = n;
    (tree as any).nodeCount = nodeCount;

    // Vectors - copy into new typed arrays to avoid alignment issues
    const vectorsSize = n * DIMENSIONS * 4;
    const vectorsBuf = new ArrayBuffer(vectorsSize);
    buffer.copy(Buffer.from(vectorsBuf), 0, offset, offset + vectorsSize);
    (tree as any).vectors = new Float32Array(vectorsBuf);
    offset += vectorsSize;

    // Labels
    const labelsBuf = new ArrayBuffer(n);
    buffer.copy(Buffer.from(labelsBuf), 0, offset, offset + n);
    (tree as any).labels = new Uint8Array(labelsBuf);
    offset += n;

    // Node data
    (tree as any).nodePoints = new Int32Array(n);
    (tree as any).nodeRadii = new Float32Array(n);
    (tree as any).nodeLeft = new Int32Array(n).fill(-1);
    (tree as any).nodeRight = new Int32Array(n).fill(-1);

    for (let i = 0; i < nodeCount; i++) {
      tree.nodePoints[i] = buffer.readInt32LE(offset); offset += 4;
      tree.nodeRadii[i] = buffer.readFloatLE(offset); offset += 4;
      tree.nodeLeft[i] = buffer.readInt32LE(offset); offset += 4;
      tree.nodeRight[i] = buffer.readInt32LE(offset); offset += 4;
    }

    return tree;
  }
}

/**
 * Brute-force KNN for validation and fallback.
 * Useful for testing correctness of VP-Tree results.
 */
export function bruteForceKnn(
  query: Float32Array,
  vectors: Float32Array,
  labels: Uint8Array,
  n: number,
  k: number
): { fraudCount: number; neighbors: { dist: number; isFraud: boolean }[] } {
  const results: { dist: number; isFraud: boolean }[] = [];

  for (let i = 0; i < n; i++) {
    const offset = i * DIMENSIONS;
    let sum = 0;
    for (let d = 0; d < DIMENSIONS; d++) {
      const diff = query[d] - vectors[offset + d];
      sum += diff * diff;
    }
    const dist = Math.sqrt(sum);
    
    if (results.length < k) {
      results.push({ dist, isFraud: labels[i] === 1 });
      if (results.length === k) {
        results.sort((a, b) => a.dist - b.dist);
      }
    } else if (dist < results[k - 1].dist) {
      results[k - 1] = { dist, isFraud: labels[i] === 1 };
      results.sort((a, b) => a.dist - b.dist);
    }
  }

  if (results.length < k) {
    results.sort((a, b) => a.dist - b.dist);
  }

  let fraudCount = 0;
  for (const r of results) {
    if (r.isFraud) fraudCount++;
  }

  return { fraudCount, neighbors: results };
}
