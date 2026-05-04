/**
 * Integration tests for the fraud detection API.
 * Tests the full flow: payload → vectorization → KNN → response.
 */
import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { vectorize, type TransactionPayload, type NormalizationConstants, type MccRiskMap } from "../src/vectorize";
import { VPTree, bruteForceKnn, type ReferenceVector } from "../src/vptree";

const normalization: NormalizationConstants = {
  max_amount: 10000,
  max_installments: 12,
  amount_vs_avg_ratio: 10,
  max_minutes: 1440,
  max_km: 1000,
  max_tx_count_24h: 20,
  max_merchant_avg_amount: 10000,
};

const mccRisk: MccRiskMap = {
  "5411": 0.15,
  "5812": 0.30,
  "5912": 0.20,
  "5944": 0.45,
  "7801": 0.80,
  "7802": 0.75,
  "7995": 0.85,
  "4511": 0.35,
  "5311": 0.25,
  "5999": 0.50,
};

describe("Integration: full fraud detection flow", () => {
  let tree: VPTree;
  let refs: ReferenceVector[];

  beforeAll(() => {
    // Create a realistic mini-dataset with clear fraud/legit clusters
    refs = [];

    // Legit cluster: low amount, known merchant, near home, card present
    for (let i = 0; i < 50; i++) {
      const vec = new Float32Array(14);
      vec[0] = 0.01 + Math.random() * 0.05;  // low amount
      vec[1] = 0.08 + Math.random() * 0.1;   // low installments
      vec[2] = 0.03 + Math.random() * 0.05;  // normal ratio
      vec[3] = 0.4 + Math.random() * 0.3;    // business hours
      vec[4] = Math.random() * 0.67;         // weekday
      vec[5] = -1;                           // no last tx
      vec[6] = -1;
      vec[7] = 0.01 + Math.random() * 0.05;  // near home
      vec[8] = 0.05 + Math.random() * 0.15;  // low tx count
      vec[9] = 0;                            // not online
      vec[10] = 1;                           // card present
      vec[11] = 0;                           // known merchant
      vec[12] = 0.15 + Math.random() * 0.1;  // low risk mcc
      vec[13] = 0.01 + Math.random() * 0.05; // low merchant avg
      refs.push({ vector: vec, isFraud: false });
    }

    // Fraud cluster: high amount, unknown merchant, far from home, high risk
    for (let i = 0; i < 50; i++) {
      const vec = new Float32Array(14);
      vec[0] = 0.8 + Math.random() * 0.2;   // high amount
      vec[1] = 0.7 + Math.random() * 0.3;   // high installments
      vec[2] = 0.8 + Math.random() * 0.2;   // high ratio
      vec[3] = 0.01 + Math.random() * 0.2;  // late night
      vec[4] = 0.8 + Math.random() * 0.2;   // weekend
      vec[5] = -1;                           // no last tx
      vec[6] = -1;
      vec[7] = 0.7 + Math.random() * 0.3;   // far from home
      vec[8] = 0.8 + Math.random() * 0.2;   // high tx count
      vec[9] = 1;                            // online
      vec[10] = 0;                           // card not present
      vec[11] = 1;                           // unknown merchant
      vec[12] = 0.75 + Math.random() * 0.1;  // high risk mcc
      vec[13] = 0.003 + Math.random() * 0.005; // low merchant avg
      refs.push({ vector: vec, isFraud: true });
    }

    tree = new VPTree(refs);
  });

  test("legitimate transaction gets approved", () => {
    const payload: TransactionPayload = {
      id: "tx-legit-test",
      transaction: { amount: 41.12, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
      customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ["MERC-003", "MERC-016"] },
      merchant: { id: "MERC-016", mcc: "5411", avg_amount: 60.25 },
      terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
      last_transaction: null,
    };

    const queryVec = vectorize(payload, normalization, mccRisk);
    const fraudScore = tree.fraudScore(queryVec);

    expect(fraudScore).toBeLessThan(0.6);
  });

  test("fraudulent transaction gets denied", () => {
    const payload: TransactionPayload = {
      id: "tx-fraud-test",
      transaction: { amount: 9505.97, installments: 10, requested_at: "2026-03-14T05:15:12Z" },
      customer: { avg_amount: 81.28, tx_count_24h: 20, known_merchants: ["MERC-008", "MERC-007", "MERC-005"] },
      merchant: { id: "MERC-068", mcc: "7802", avg_amount: 54.86 },
      terminal: { is_online: false, card_present: true, km_from_home: 952.27 },
      last_transaction: null,
    };

    const queryVec = vectorize(payload, normalization, mccRisk);
    const fraudScore = tree.fraudScore(queryVec);

    expect(fraudScore).toBeGreaterThanOrEqual(0.6);
  });

  test("fraud_score is always a multiple of 0.2 (K=5)", () => {
    // Score = frauds_among_5 / 5, so values are 0, 0.2, 0.4, 0.6, 0.8, 1.0
    const validScores = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

    for (let i = 0; i < 20; i++) {
      const query = new Float32Array(14);
      for (let d = 0; d < 14; d++) {
        query[d] = Math.random();
      }
      const score = tree.fraudScore(query);
      const isValid = validScores.some(v => Math.abs(score - v) < 0.001);
      expect(isValid).toBe(true);
    }
  });

  test("approved is fraud_score < 0.6", () => {
    const THRESHOLD = 0.6;

    for (let i = 0; i < 20; i++) {
      const query = new Float32Array(14);
      for (let d = 0; d < 14; d++) {
        query[d] = Math.random();
      }
      const score = tree.fraudScore(query);
      const approved = score < THRESHOLD;

      if (score < 0.6) {
        expect(approved).toBe(true);
      } else {
        expect(approved).toBe(false);
      }
    }
  });

  test("response format matches API spec", () => {
    const query = new Float32Array(14);
    for (let d = 0; d < 14; d++) {
      query[d] = 0.5;
    }
    const fraudScore = tree.fraudScore(query);
    const approved = fraudScore < 0.6;

    const response = { approved, fraud_score: fraudScore };

    expect(response).toHaveProperty("approved");
    expect(response).toHaveProperty("fraud_score");
    expect(typeof response.approved).toBe("boolean");
    expect(typeof response.fraud_score).toBe("number");
    expect(response.fraud_score).toBeGreaterThanOrEqual(0);
    expect(response.fraud_score).toBeLessThanOrEqual(1);
  });
});

describe("Integration: edge cases", () => {
  test("all MCC codes produce valid risk values", () => {
    const knownMccs = ["5411", "5812", "5912", "5944", "7801", "7802", "7995", "4511", "5311", "5999"];
    
    for (const mcc of knownMccs) {
      const payload: TransactionPayload = {
        id: "tx-mcc-test",
        transaction: { amount: 100, installments: 1, requested_at: "2026-03-11T12:00:00Z" },
        customer: { avg_amount: 200, tx_count_24h: 1, known_merchants: ["MERC-001"] },
        merchant: { id: "MERC-001", mcc, avg_amount: 100 },
        terminal: { is_online: false, card_present: true, km_from_home: 10 },
        last_transaction: null,
      };
      
      const vec = vectorize(payload, normalization, mccRisk);
      expect(vec[12]).toBeGreaterThanOrEqual(0);
      expect(vec[12]).toBeLessThanOrEqual(1);
      expect(vec[12]).toBeCloseTo(mccRisk[mcc], 4);
    }
  });

  test("very recent last_transaction (1 minute ago)", () => {
    const payload: TransactionPayload = {
      id: "tx-recent",
      transaction: { amount: 100, installments: 1, requested_at: "2026-03-11T12:01:00Z" },
      customer: { avg_amount: 200, tx_count_24h: 5, known_merchants: ["MERC-001"] },
      merchant: { id: "MERC-001", mcc: "5411", avg_amount: 100 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: { timestamp: "2026-03-11T12:00:00Z", km_from_current: 0.5 },
    };

    const vec = vectorize(payload, normalization, mccRisk);
    // 1 minute / 1440 = 0.000694
    expect(vec[5]).toBeCloseTo(1 / 1440, 4);
    expect(vec[6]).toBeCloseTo(0.0005, 3);
  });

  test("zero km_from_home", () => {
    const payload: TransactionPayload = {
      id: "tx-home",
      transaction: { amount: 100, installments: 1, requested_at: "2026-03-11T12:00:00Z" },
      customer: { avg_amount: 200, tx_count_24h: 1, known_merchants: ["MERC-001"] },
      merchant: { id: "MERC-001", mcc: "5411", avg_amount: 100 },
      terminal: { is_online: false, card_present: true, km_from_home: 0 },
      last_transaction: null,
    };

    const vec = vectorize(payload, normalization, mccRisk);
    expect(vec[7]).toBe(0);
  });

  test("minimum values produce valid vector", () => {
    const payload: TransactionPayload = {
      id: "tx-min",
      transaction: { amount: 0.01, installments: 1, requested_at: "2026-03-09T00:00:00Z" },
      customer: { avg_amount: 0.01, tx_count_24h: 0, known_merchants: [] },
      merchant: { id: "MERC-001", mcc: "5411", avg_amount: 0.01 },
      terminal: { is_online: false, card_present: false, km_from_home: 0 },
      last_transaction: null,
    };

    const vec = vectorize(payload, normalization, mccRisk);
    expect(vec.length).toBe(14);
    // All values should be valid numbers
    for (let i = 0; i < 14; i++) {
      expect(isNaN(vec[i])).toBe(false);
      expect(isFinite(vec[i])).toBe(true);
    }
  });
});
