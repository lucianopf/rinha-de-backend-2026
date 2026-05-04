# Rinha de Backend 2026 – @lucianopf

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).
Detecção de fraude em transações de cartão por busca vetorial (k-NN).

## Stack

| Camada | Tecnologia | Motivo |
|--------|-----------|--------|
| Load Balancer | Nginx 1.27 alpine | round-robin com keepalive 256 |
| API ×2 | Bun 1.2 + TypeScript | Runtime rápido, worker threads nativo |
| Busca Vetorial | IVF (Inverted File Index) | K-means 1024 clusters, nProbe=20 |
| Dataset | `references.json.gz` | Pré-processado no build para `ivf-index.bin` |

## Abordagem

**IVF + Worker Threads** — a melhor combinação que testamos (score 3412, p99=23ms):

1. **Pré-processamento (build time):** K-means agrupa 3M vetores em 1024 clusters. Vetores quantizados de Float32 para Uint8 (42MB).
2. **Runtime:** Cada instância roda 2 worker threads. O main thread faz I/O (parse JSON, vectorize), workers fazem a busca IVF.
3. **Busca IVF:** Para cada query, encontra os 20 clusters mais próximos e busca só neles (~60K vetores vs 3M). Retorna os 5 vizinhos mais próximos.

### Resultados nos benchmarks

| Concorrência | p99 | Score | Throughput |
|-------------|-----|-------|-----------|
| 20 | 23ms | 3412 | 1318 req/s |
| 50 | 50ms | 2956 | ~2500 req/s |
| 100 | 95ms | 2622 | ~2700 req/s |

## Como rodar

```bash
docker compose up -d --build
curl -fsS http://localhost:9999/ready
```

## Como testar

```bash
curl -X POST http://localhost:9999/fraud-score \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tx-123",
    "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2025-01-15T14:30:00Z" },
    "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-001"] },
    "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
    "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.7 },
    "last_transaction": { "timestamp": "2025-01-15T12:00:00Z", "km_from_current": 18.8 }
  }'
```

## Limites de recursos

| Serviço | CPU | Memória |
|---------|-----|---------|
| Nginx | 0.10 | 16MB |
| API 1 | 0.45 | 167MB |
| API 2 | 0.45 | 167MB |
| **Total** | **1.00** | **350MB** |

## Estrutura

```
├── src/
│   ├── server-ivf-workers.ts   # Server principal (IVF + Workers)
│   ├── ivf-search-worker.ts    # Worker thread para busca IVF
│   ├── ivf-search.ts           # IVF index build + search
│   ├── ivf-preprocess.ts       # Pré-processamento (build time)
│   ├── worker-pool.ts          # Pool de workers
│   ├── vectorize.ts            # Payload → vetor 14D
│   └── search.ts               # Quantização de vetores
├── resources/                   # Dados de referência do desafio
├── docker-compose.yml
├── Dockerfile
├── nginx.conf
├── info.json
└── package.json
```
