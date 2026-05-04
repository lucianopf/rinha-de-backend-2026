/**
 * Tests for the vectorization module.
 * Validates all 14 dimensions, edge cases, and the examples from the documentation.
 */
import { describe, test, expect } from "bun:test";
import { vectorize, type TransactionPayload, type NormalizationConstants, type MccRiskMap } from "../src/vectorize";

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

describe("vectorize", () => {
  test("legitimate transaction from documentation example", () => {
    // From REGRAS_DE_DETECCAO.md - legitimate transaction
    const payload: TransactionPayload = {
      id: "tx-1329056812",
      transaction: { amount: 41.12, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
      customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ["MERC-003", "MERC-016"] },
      merchant: { id: "MERC-016", mcc: "5411", avg_amount: 60.25 },
      terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
      last_transaction: null,
    };

    const vec = vectorize(payload, normalization, mccRisk);

    // Expected: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
    expect(vec[0]).toBeCloseTo(0.0041, 3);    // amount: 41.12/10000
    expect(vec[1]).toBeCloseTo(0.1667, 3);    // installments: 2/12
    expect(vec[2]).toBeCloseTo(0.05, 3);      // amount_vs_avg: (41.12/82.24)/10
    expect(vec[3]).toBeCloseTo(18 / 23, 3);   // hour: 18/23 = 0.7826
    // 2026-03-11 is a Wednesday -> dayOfWeek=2 -> 2/6 = 0.3333
    expect(vec[4]).toBeCloseTo(2 / 6, 3);     // day_of_week
    expect(vec[5]).toBe(-1);                  // last_transaction null
    expect(vec[6]).toBe(-1);                  // last_transaction null
    expect(vec[7]).toBeCloseTo(0.02923, 3);   // km_from_home: 29.23/1000
    expect(vec[8]).toBeCloseTo(0.15, 3);      // tx_count_24h: 3/20
    expect(vec[9]).toBe(0);                   // is_online: false
    expect(vec[10]).toBe(1);                  // card_present: true
    expect(vec[11]).toBe(0);                  // known merchant (MERC-016 in list)
    expect(vec[12]).toBeCloseTo(0.15, 3);     // mcc_risk: 5411
    expect(vec[13]).toBeCloseTo(0.006025, 3); // merchant_avg: 60.25/10000
  });

  test("fraudulent transaction from documentation example", () => {
    // From REGRAS_DE_DETECCAO.md - fraudulent transaction
    const payload: TransactionPayload = {
      id: "tx-3330991687",
      transaction: { amount: 9505.97, installments: 10, requested_at: "2026-03-14T05:15:12Z" },
      customer: { avg_amount: 81.28, tx_count_24h: 20, known_merchants: ["MERC-008", "MERC-007", "MERC-005"] },
      merchant: { id: "MERC-068", mcc: "7802", avg_amount: 54.86 },
      terminal: { is_online: false, card_present: true, km_from_home: 952.27 },
      last_transaction: null,
    };

    const vec = vectorize(payload, normalization, mccRisk);

    // Expected: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
    expect(vec[0]).toBeCloseTo(0.9506, 3);    // amount
    expect(vec[1]).toBeCloseTo(0.8333, 3);    // installments: 10/12
    expect(vec[2]).toBeCloseTo(1.0, 3);       // amount_vs_avg: clamped (9505.97/81.28)/10 > 1
    expect(vec[3]).toBeCloseTo(5 / 23, 3);    // hour: 5/23 = 0.2174
    // 2026-03-14 is a Saturday -> jsDay=6, dayOfWeek=5 -> 5/6 = 0.8333
    expect(vec[4]).toBeCloseTo(5 / 6, 3);     // day_of_week
    expect(vec[5]).toBe(-1);
    expect(vec[6]).toBe(-1);
    expect(vec[7]).toBeCloseTo(0.95227, 3);   // km_from_home
    expect(vec[8]).toBeCloseTo(1.0, 3);       // tx_count_24h: 20/20 = 1.0
    expect(vec[9]).toBe(0);                   // is_online: false
    expect(vec[10]).toBe(1);                  // card_present: true
    expect(vec[11]).toBe(1);                  // unknown merchant
    expect(vec[12]).toBeCloseTo(0.75, 3);     // mcc_risk: 7802
    expect(vec[13]).toBeCloseTo(0.005486, 3); // merchant_avg
  });

  test("transaction with last_transaction present", () => {
    const payload: TransactionPayload = {
      id: "tx-123",
      transaction: { amount: 384.88, installments: 3, requested_at: "2026-03-11T20:23:35Z" },
      customer: { avg_amount: 769.76, tx_count_24h: 3, known_merchants: ["MERC-009", "MERC-001"] },
      merchant: { id: "MERC-001", mcc: "5912", avg_amount: 298.95 },
      terminal: { is_online: false, card_present: true, km_from_home: 13.71 },
      last_transaction: { timestamp: "2026-03-11T14:58:35Z", km_from_current: 18.86 },
    };

    const vec = vectorize(payload, normalization, mccRisk);

    // minutes_since_last_tx: (20:23:35 - 14:58:35) = 5h 25m = 325 min
    // 325 / 1440 = 0.2257
    expect(vec[5]).toBeCloseTo(325 / 1440, 3);
    // km_from_last_tx: 18.86 / 1000 = 0.01886
    expect(vec[6]).toBeCloseTo(0.01886, 3);
  });

  test("clamping works for values above maximum", () => {
    const payload: TransactionPayload = {
      id: "tx-clamp",
      transaction: { amount: 50000, installments: 24, requested_at: "2026-03-11T12:00:00Z" },
      customer: { avg_amount: 1, tx_count_24h: 100, known_merchants: [] },
      merchant: { id: "MERC-X", mcc: "9999", avg_amount: 99999 },
      terminal: { is_online: true, card_present: false, km_from_home: 5000 },
      last_transaction: { timestamp: "2026-03-10T12:00:00Z", km_from_current: 2000 },
    };

    const vec = vectorize(payload, normalization, mccRisk);

    expect(vec[0]).toBe(1.0);  // amount clamped
    expect(vec[1]).toBe(1.0);  // installments clamped
    expect(vec[2]).toBe(1.0);  // amount_vs_avg clamped
    expect(vec[7]).toBe(1.0);  // km_from_home clamped
    expect(vec[8]).toBe(1.0);  // tx_count_24h clamped
    expect(vec[6]).toBe(1.0);  // km_from_last_tx clamped
    expect(vec[13]).toBe(1.0); // merchant_avg clamped
  });

  test("unknown MCC defaults to 0.5", () => {
    const payload: TransactionPayload = {
      id: "tx-unknown-mcc",
      transaction: { amount: 100, installments: 1, requested_at: "2026-03-11T12:00:00Z" },
      customer: { avg_amount: 200, tx_count_24h: 1, known_merchants: ["MERC-001"] },
      merchant: { id: "MERC-001", mcc: "1234", avg_amount: 100 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: null,
    };

    const vec = vectorize(payload, normalization, mccRisk);
    expect(vec[12]).toBe(0.5);
  });

  test("is_online and card_present boolean flags", () => {
    const makePayload = (isOnline: boolean, cardPresent: boolean): TransactionPayload => ({
      id: "tx-bool",
      transaction: { amount: 100, installments: 1, requested_at: "2026-03-11T12:00:00Z" },
      customer: { avg_amount: 200, tx_count_24h: 1, known_merchants: ["MERC-001"] },
      merchant: { id: "MERC-001", mcc: "5411", avg_amount: 100 },
      terminal: { is_online: isOnline, card_present: cardPresent, km_from_home: 10 },
      last_transaction: null,
    });

    const vec1 = vectorize(makePayload(true, true), normalization, mccRisk);
    expect(vec1[9]).toBe(1);
    expect(vec1[10]).toBe(1);

    const vec2 = vectorize(makePayload(false, false), normalization, mccRisk);
    expect(vec2[9]).toBe(0);
    expect(vec2[10]).toBe(0);
  });

  test("day of week mapping (Mon=0 to Sun=6)", () => {
    // Test each day of the week
    const days = [
      { date: "2026-03-09T12:00:00Z", expected: 0 }, // Monday
      { date: "2026-03-10T12:00:00Z", expected: 1 }, // Tuesday
      { date: "2026-03-11T12:00:00Z", expected: 2 }, // Wednesday
      { date: "2026-03-12T12:00:00Z", expected: 3 }, // Thursday
      { date: "2026-03-13T12:00:00Z", expected: 4 }, // Friday
      { date: "2026-03-14T12:00:00Z", expected: 5 }, // Saturday
      { date: "2026-03-15T12:00:00Z", expected: 6 }, // Sunday
    ];

    for (const { date, expected } of days) {
      const payload: TransactionPayload = {
        id: "tx-day",
        transaction: { amount: 100, installments: 1, requested_at: date },
        customer: { avg_amount: 200, tx_count_24h: 1, known_merchants: ["MERC-001"] },
        merchant: { id: "MERC-001", mcc: "5411", avg_amount: 100 },
        terminal: { is_online: false, card_present: true, km_from_home: 10 },
        last_transaction: null,
      };

      const vec = vectorize(payload, normalization, mccRisk);
      expect(vec[4]).toBeCloseTo(expected / 6, 4);
    }
  });

  test("unknown_merchant flag", () => {
    const payload: TransactionPayload = {
      id: "tx-merchant",
      transaction: { amount: 100, installments: 1, requested_at: "2026-03-11T12:00:00Z" },
      customer: { avg_amount: 200, tx_count_24h: 1, known_merchants: ["MERC-001", "MERC-002"] },
      merchant: { id: "MERC-003", mcc: "5411", avg_amount: 100 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: null,
    };

    const vec = vectorize(payload, normalization, mccRisk);
    expect(vec[11]).toBe(1); // MERC-003 not in known_merchants

    // Now test with known merchant
    payload.merchant.id = "MERC-001";
    const vec2 = vectorize(payload, normalization, mccRisk);
    expect(vec2[11]).toBe(0); // MERC-001 is in known_merchants
  });

  test("hour_of_day boundary values", () => {
    const makePayload = (hour: string): TransactionPayload => ({
      id: "tx-hour",
      transaction: { amount: 100, installments: 1, requested_at: `2026-03-11T${hour}:00:00Z` },
      customer: { avg_amount: 200, tx_count_24h: 1, known_merchants: ["MERC-001"] },
      merchant: { id: "MERC-001", mcc: "5411", avg_amount: 100 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: null,
    });

    const vec0 = vectorize(makePayload("00"), normalization, mccRisk);
    expect(vec0[3]).toBe(0); // 0/23

    const vec23 = vectorize(makePayload("23"), normalization, mccRisk);
    expect(vec23[3]).toBe(1); // 23/23
  });

  test("minutes_since_last_tx boundary: exactly 24 hours", () => {
    const payload: TransactionPayload = {
      id: "tx-24h",
      transaction: { amount: 100, installments: 1, requested_at: "2026-03-12T12:00:00Z" },
      customer: { avg_amount: 200, tx_count_24h: 1, known_merchants: ["MERC-001"] },
      merchant: { id: "MERC-001", mcc: "5411", avg_amount: 100 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: { timestamp: "2026-03-11T12:00:00Z", km_from_current: 100 },
    };

    const vec = vectorize(payload, normalization, mccRisk);
    // 24h = 1440 min / 1440 = 1.0
    expect(vec[5]).toBeCloseTo(1.0, 3);
  });

  test("vector dimensions are exactly 14", () => {
    const payload: TransactionPayload = {
      id: "tx-dim",
      transaction: { amount: 100, installments: 1, requested_at: "2026-03-11T12:00:00Z" },
      customer: { avg_amount: 200, tx_count_24h: 1, known_merchants: ["MERC-001"] },
      merchant: { id: "MERC-001", mcc: "5411", avg_amount: 100 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: null,
    };

    const vec = vectorize(payload, normalization, mccRisk);
    expect(vec.length).toBe(14);
  });
});
