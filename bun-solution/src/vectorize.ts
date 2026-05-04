/**
 * Vectorization module - transforms transaction payloads into 14-dimensional vectors
 * following the normalization rules defined in REGRAS_DE_DETECCAO.md
 */

export interface TransactionPayload {
  id: string;
  transaction: {
    amount: number;
    installments: number;
    requested_at: string;
  };
  customer: {
    avg_amount: number;
    tx_count_24h: number;
    known_merchants: string[];
  };
  merchant: {
    id: string;
    mcc: string;
    avg_amount: number;
  };
  terminal: {
    is_online: boolean;
    card_present: boolean;
    km_from_home: number;
  };
  last_transaction: {
    timestamp: string;
    km_from_current: number;
  } | null;
}

export interface NormalizationConstants {
  max_amount: number;
  max_installments: number;
  amount_vs_avg_ratio: number;
  max_minutes: number;
  max_km: number;
  max_tx_count_24h: number;
  max_merchant_avg_amount: number;
}

export type MccRiskMap = Record<string, number>;

const DEFAULT_MCC_RISK = 0.5;

/** Clamp value to [0.0, 1.0] */
function clamp(x: number): number {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

/**
 * Vectorize a transaction payload into a 14-dimensional normalized vector.
 */
export function vectorize(
  payload: TransactionPayload,
  normalization: NormalizationConstants,
  mccRisk: MccRiskMap
): Float32Array {
  const vec = new Float32Array(14);

  // 0: amount
  vec[0] = clamp(payload.transaction.amount / normalization.max_amount);

  // 1: installments
  vec[1] = clamp(payload.transaction.installments / normalization.max_installments);

  // 2: amount_vs_avg
  vec[2] = clamp(
    (payload.transaction.amount / payload.customer.avg_amount) /
      normalization.amount_vs_avg_ratio
  );

  // 3: hour_of_day (UTC)
  const requestedAt = new Date(payload.transaction.requested_at);
  vec[3] = requestedAt.getUTCHours() / 23;

  // 4: day_of_week (Mon=0, Sun=6)
  // JS getUTCDay: Sun=0, Mon=1, ..., Sat=6
  // We need: Mon=0, Tue=1, ..., Sun=6
  const jsDay = requestedAt.getUTCDay();
  const dayOfWeek = jsDay === 0 ? 6 : jsDay - 1;
  vec[4] = dayOfWeek / 6;

  // 5: minutes_since_last_tx
  if (payload.last_transaction === null) {
    vec[5] = -1;
  } else {
    const lastTs = new Date(payload.last_transaction.timestamp);
    const minutes = (requestedAt.getTime() - lastTs.getTime()) / 60000;
    vec[5] = clamp(minutes / normalization.max_minutes);
  }

  // 6: km_from_last_tx
  if (payload.last_transaction === null) {
    vec[6] = -1;
  } else {
    vec[6] = clamp(payload.last_transaction.km_from_current / normalization.max_km);
  }

  // 7: km_from_home
  vec[7] = clamp(payload.terminal.km_from_home / normalization.max_km);

  // 8: tx_count_24h
  vec[8] = clamp(payload.customer.tx_count_24h / normalization.max_tx_count_24h);

  // 9: is_online
  vec[9] = payload.terminal.is_online ? 1 : 0;

  // 10: card_present
  vec[10] = payload.terminal.card_present ? 1 : 0;

  // 11: unknown_merchant (1 = merchant not in known_merchants)
  vec[11] = payload.customer.known_merchants.includes(payload.merchant.id) ? 0 : 1;

  // 12: mcc_risk
  vec[12] = mccRisk[payload.merchant.mcc] ?? DEFAULT_MCC_RISK;

  // 13: merchant_avg_amount
  vec[13] = clamp(payload.merchant.avg_amount / normalization.max_merchant_avg_amount);

  return vec;
}
