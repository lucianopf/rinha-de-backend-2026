/**
 * Tests for the VP-Tree implementation.
 * Validates KNN search correctness against brute-force baseline.
 */
import { describe, test, expect } from "bun:test";
import { VPTree, bruteForceKnn, distance, type ReferenceVector } from "../src/vptree";

function makeVector(values: number[]): Float32Array {
  return new Float32Array(values.length === 14 ? values : [...values, ...new Array(14 - values.length).fill(0)]);
}

function makeRef(values: number[], isFraud: boolean): ReferenceVector {
  return { vector: makeVector(values), isFraud };
}

describe("VPTree", () => {
  test("basic KNN search with small dataset", () => {
    const refs: ReferenceVector[] = [
      makeRef([0.01, 0.08, 0.05, 0.83, 0.17, -1, -1, 0.04, 0.25, 0, 1, 0, 0.2, 0.04], false),
      makeRef([0.58, 0.92, 1.0, 0.04, 0, 0.01, 0.44, 0.46, 0.4, 1, 0, 1, 0.85, 0.003], true),
      makeRef([0.004, 0.17, 0.05, 0.39, 0.67, 0.3, 0.01, 0.02, 0.15, 0, 1, 0, 0.2, 0.03], false),
      makeRef([0.97, 1.0, 1.0, 0.04, 0, 0.01, 0.5, 0.5, 0.5, 1, 0, 1, 0.8, 0.005], true),
      makeRef([0.41, 1.0, 1.0, 0.1, 0.5, 0.02, 0.3, 0.3, 0.3, 1, 0, 1, 0.75, 0.004], true),
      makeRef([0.009, 0.08, 0.05, 0.7, 0.3, -1, -1, 0.03, 0.2, 0, 1, 0, 0.15, 0.05], false),
    ];

    const tree = new VPTree(refs);

    // Query close to ref[0] and ref[5] (legitimate transactions)
    const query = makeVector([0.01, 0.08, 0.05, 0.83, 0.17, -1, -1, 0.04, 0.25, 0, 1, 0, 0.2, 0.04]);
    const neighbors = tree.knn(query, 3);

    expect(neighbors.length).toBe(3);
    // The closest should be ref[0] (exact match)
    expect(neighbors[0].dist).toBeCloseTo(0, 3);
  });

  test("fraud score calculation", () => {
    const refs: ReferenceVector[] = [
      makeRef([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 1, 0, 0.1, 0.1], false),
      makeRef([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 1, 0, 0.1, 0.11], false),
      makeRef([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 1, 0, 0.1, 0.12], false),
      makeRef([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1, 0, 1, 0.9, 0.9], true),
      makeRef([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1, 0, 1, 0.9, 0.91], true),
      makeRef([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1, 0, 1, 0.9, 0.92], true),
      makeRef([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0.5, 0.5], false),
      makeRef([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0.5, 0.51], true),
    ];

    const tree = new VPTree(refs);

    // Query close to fraud cluster
    const fraudQuery = makeVector([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1, 0, 1, 0.9, 0.9]);
    const fraudScore = tree.fraudScore(fraudQuery);
    expect(fraudScore).toBeGreaterThanOrEqual(0.6); // Should be mostly fraud neighbors

    // Query close to legit cluster
    const legitQuery = makeVector([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 1, 0, 0.1, 0.1]);
    const legitScore = tree.fraudScore(legitQuery);
    expect(legitScore).toBeLessThan(0.6); // Should be mostly legit neighbors
  });

  test("VP-Tree matches brute force results", () => {
    // Generate random reference vectors
    const n = 1000;
    const refs: ReferenceVector[] = [];
    for (let i = 0; i < n; i++) {
      const vec = new Float32Array(14);
      for (let d = 0; d < 14; d++) {
        vec[d] = Math.random();
      }
      refs.push({ vector: vec, isFraud: Math.random() > 0.5 });
    }

    const tree = new VPTree(refs);

    // Flatten for brute force
    const flatVectors = new Float32Array(n * 14);
    const flatLabels = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      flatVectors.set(refs[i].vector, i * 14);
      flatLabels[i] = refs[i].isFraud ? 1 : 0;
    }

    // Test multiple random queries
    for (let q = 0; q < 20; q++) {
      const query = new Float32Array(14);
      for (let d = 0; d < 14; d++) {
        query[d] = Math.random();
      }

      const treeResult = tree.knn(query, 5);
      const bruteResult = bruteForceKnn(query, flatVectors, flatLabels, n, 5);

      // Distances should match (allowing for floating point differences)
      for (let i = 0; i < 5; i++) {
        expect(treeResult[i].dist).toBeCloseTo(bruteResult.neighbors[i].dist, 4);
      }
    }
  });

  test("serialization and deserialization preserves results", () => {
    const refs: ReferenceVector[] = [];
    for (let i = 0; i < 100; i++) {
      const vec = new Float32Array(14);
      for (let d = 0; d < 14; d++) {
        vec[d] = Math.random();
      }
      refs.push({ vector: vec, isFraud: Math.random() > 0.5 });
    }

    const tree = new VPTree(refs);
    const serialized = tree.serialize();
    const deserialized = VPTree.deserialize(serialized);

    // Test that deserialized tree gives same results
    for (let q = 0; q < 10; q++) {
      const query = new Float32Array(14);
      for (let d = 0; d < 14; d++) {
        query[d] = Math.random();
      }

      const origResult = tree.knn(query, 5);
      const deserResult = deserialized.knn(query, 5);

      for (let i = 0; i < 5; i++) {
        expect(deserResult[i].dist).toBeCloseTo(origResult[i].dist, 4);
        expect(deserResult[i].index).toBe(origResult[i].index);
      }
    }
  });

  test("handles edge case with fewer than K references", () => {
    const refs: ReferenceVector[] = [
      makeRef([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 1, 0, 0.5, 0.5], false),
      makeRef([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1, 0, 1, 0.6, 0.6], true),
    ];

    const tree = new VPTree(refs);
    const query = makeVector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 1, 0, 0.5, 0.5]);
    const neighbors = tree.knn(query, 5);

    // Should return only 2 neighbors since we only have 2 references
    expect(neighbors.length).toBe(2);
  });

  test("distance function correctness", () => {
    const a = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const b = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(distance(a, b)).toBeCloseTo(1.0, 5);

    const c = new Float32Array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const d = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(distance(c, d)).toBeCloseTo(Math.sqrt(2), 5);
  });

  test("sentinel value -1 handling in vectors", () => {
    // Vectors with -1 sentinel should be naturally close to each other
    const refs: ReferenceVector[] = [
      makeRef([0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 1, 0, 0.5, 0.5], false),
      makeRef([0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 1, 0, 0.5, 0.51], false),
      makeRef([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 1, 0, 0.5, 0.5], true),
      makeRef([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 1, 0, 0.5, 0.51], true),
      makeRef([0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 1, 0, 0.5, 0.52], false),
    ];

    const tree = new VPTree(refs);

    // Query with -1 sentinel should find other -1 sentinel vectors closer
    const query = makeVector([0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 1, 0, 0.5, 0.5]);
    const neighbors = tree.knn(query, 3);

    // The 3 nearest should be the ones with -1 sentinel (indices 0, 1, 4)
    const sentinelRefs = [0, 1, 4];
    for (const n of neighbors) {
      expect(sentinelRefs).toContain(n.index);
    }
  });
});
