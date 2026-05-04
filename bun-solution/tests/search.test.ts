/**
 * Tests for the quantized search module.
 */
import { describe, test, expect } from "bun:test";
import { buildIndex, findFraudCount, quantizeVector, serializeIndex, deserializeIndex } from "../src/search";

describe("Quantized Search", () => {
  test("quantizeVector maps [-1, 1] to [0, 255]", () => {
    const vec = new Float32Array([-1, 0, 1, 0.5, -0.5, -1, -1, 0, 0, 0, 0, 0, 0, 0]);
    const quantized = quantizeVector(vec);

    expect(quantized[0]).toBe(0);     // -1 → 0
    expect(quantized[1]).toBe(127);   // 0 → 127 or 128
    expect(quantized[2]).toBe(255);   // 1 → 255
    expect(quantized[3]).toBeGreaterThan(180); // 0.5 → ~191
    expect(quantized[4]).toBeLessThan(75);     // -0.5 → ~64
  });

  test("buildIndex creates correct structure", () => {
    const vectors = [
      new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 1, 0, 0.5, 0.5]),
      new Float32Array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1, 0, 1, 0.8, 0.8]),
    ];
    const labels = [false, true];

    const index = buildIndex(vectors, labels);

    expect(index.n).toBe(2);
    expect(index.vectors.length).toBe(2 * 14);
    expect(index.labels[0]).toBe(0);
    expect(index.labels[1]).toBe(1);
  });

  test("findFraudCount returns correct count with clear clusters", () => {
    // Create clear fraud and legit clusters
    const vectors: Float32Array[] = [];
    const labels: boolean[] = [];

    // Legit cluster: all values near 0.1
    for (let i = 0; i < 10; i++) {
      const v = new Float32Array(14);
      for (let d = 0; d < 14; d++) v[d] = 0.1 + Math.random() * 0.05;
      vectors.push(v);
      labels.push(false);
    }

    // Fraud cluster: all values near 0.9
    for (let i = 0; i < 10; i++) {
      const v = new Float32Array(14);
      for (let d = 0; d < 14; d++) v[d] = 0.9 + Math.random() * 0.05;
      vectors.push(v);
      labels.push(true);
    }

    const index = buildIndex(vectors, labels);

    // Query near legit cluster
    const legitQuery = quantizeVector(new Float32Array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]));
    const legitCount = findFraudCount(index, legitQuery);
    expect(legitCount).toBe(0); // All 5 nearest should be legit

    // Query near fraud cluster
    const fraudQuery = quantizeVector(new Float32Array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]));
    const fraudCount = findFraudCount(index, fraudQuery);
    expect(fraudCount).toBe(5); // All 5 nearest should be fraud
  });

  test("serialization and deserialization preserves data", () => {
    const vectors = [
      new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, -1, -1, 0.8, 0.9, 0, 1, 0, 0.5, 0.5]),
      new Float32Array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1, 0, 1, 0.8, 0.8]),
    ];
    const labels = [false, true];

    const index = buildIndex(vectors, labels);
    const serialized = serializeIndex(index);
    const deserialized = deserializeIndex(serialized);

    expect(deserialized.n).toBe(index.n);
    expect(deserialized.labels[0]).toBe(index.labels[0]);
    expect(deserialized.labels[1]).toBe(index.labels[1]);

    // Vectors should be identical
    for (let i = 0; i < index.vectors.length; i++) {
      expect(deserialized.vectors[i]).toBe(index.vectors[i]);
    }
  });

  test("handles sentinel value -1 correctly", () => {
    // -1 maps to 0 in quantized space
    const vec = new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 1, 0, 0.5, 0.5]);
    const quantized = quantizeVector(vec);

    expect(quantized[5]).toBe(0);  // -1 → 0
    expect(quantized[6]).toBe(0);  // -1 → 0
  });

  test("fraud count is always between 0 and 5", () => {
    const vectors: Float32Array[] = [];
    const labels: boolean[] = [];

    for (let i = 0; i < 100; i++) {
      const v = new Float32Array(14);
      for (let d = 0; d < 14; d++) v[d] = Math.random();
      vectors.push(v);
      labels.push(Math.random() > 0.5);
    }

    const index = buildIndex(vectors, labels);

    for (let i = 0; i < 20; i++) {
      const query = new Float32Array(14);
      for (let d = 0; d < 14; d++) query[d] = Math.random();
      const qQuery = quantizeVector(query);
      const count = findFraudCount(index, qQuery);
      expect(count).toBeGreaterThanOrEqual(0);
      expect(count).toBeLessThanOrEqual(5);
    }
  });
});
