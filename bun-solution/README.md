# Rinha de Backend 2026 - Bun.js Solution

Implementação em **Bun.js** (TypeScript) para a Rinha de Backend 2026, com foco em performance e clareza educativa.

## Arquitetura

```
┌─────────────┐     ┌──────────┐     ┌──────────┐
│   nginx     │────▶│  api 1   │     │  api 2   │
│ (port 9999) │────▶│ (Bun.js) │     │ (Bun.js) │
└─────────────┘     └──────────┘     └──────────┘
```

- **nginx**: Load balancer com round-robin (0.10 CPU, 20MB)
- **api1/api2**: Instâncias Bun.js (0.45 CPU, 165MB cada)
- **Total**: 1.0 CPU, 350MB (dentro dos limites)

## Estratégia de Detecção de Fraude

1. **Pré-processamento (build time)**: Os 3 milhões de vetores de referência são carregados, **quantizados de Float32 para Uint8** (reduzindo de 168MB para 42MB), e serializados em arquivo binário.

2. **Runtime**: Para cada requisição:
   - O payload é transformado em um vetor de 14 dimensões (normalizado)
   - O vetor é quantizado para Uint8
   - Busca brute-force nos 3M vetores quantizados para encontrar os 5 vizinhos mais próximos
   - O fraud_score é calculado como fração de fraudes entre os 5 vizinhos
   - A decisão é: `approved = fraud_score < 0.6`

### Por que Brute Force com Quantização?

A escolha de brute force + quantização Uint8 é intencional:

- **Memória**: 42MB por instância vs 168MB+ para Float32, possibilitando 2 instâncias em 350MB total
- **Precisão**: Quantização Uint8 mantém precisão suficiente para classificação KNN (testado com dataset real)
- **Cache locality**: Array contíguo de Uint8 maximiza utilização do cache L1/L2
- **Simplicidade**: Código simples e auditável, sem estruturas complexas
- **Startup rápido**: Carregamento do índice em ~70ms (vs segundos para árvores complexas)

### Também incluído: VP-Tree (alternativa)

O repositório inclui uma implementação completa de **VP-Tree** (Vantage Point Tree) em `src/vptree.ts` como alternativa educativa. A VP-Tree oferece busca exata sem brute force, mas usa mais memória (~209MB) que não cabe no limite.

## Como executar

### Desenvolvimento local

```bash
cd bun-solution
bun install
RESOURCES_DIR=../resources OUTPUT_DIR=./data bun run src/preprocess.ts
DATA_DIR=./data bun run src/server.ts
```

### Com Docker

```bash
docker compose -f bun-solution/docker-compose.yml up --build
```

### Testes

```bash
cd bun-solution
bun test
```

## Estrutura do projeto

```
bun-solution/
├── src/
│   ├── server.ts       # HTTP server (Bun.serve)
│   ├── vectorize.ts    # Vetorização do payload (14 dimensões)
│   ├── search.ts       # Busca brute-force com vetores quantizados Uint8
│   ├── vptree.ts       # VP-Tree (alternativa educativa)
│   └── preprocess.ts   # Script de pré-processamento
├── tests/
│   ├── vectorize.test.ts   # Testes de vetorização (11 testes)
│   ├── search.test.ts      # Testes da busca quantizada (6 testes)
│   ├── vptree.test.ts      # Testes da VP-Tree (7 testes)
│   └── integration.test.ts # Testes de integração (9 testes)
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
└── package.json
```

## Decisões de Design

1. **Quantização Uint8**: Reduz memória de 168MB para 42MB por instância com perda mínima de precisão
2. **Pré-processamento no build**: A quantização e serialização acontecem no Docker multi-stage build
3. **Respostas pré-computadas**: As 6 respostas possíveis são strings constantes (zero alocação no hot path)
4. **Fallback gracioso**: Em caso de erro no parsing, retorna `approved: true` para evitar erro HTTP (peso 5 na pontuação)
5. **Brute force cache-friendly**: O loop interno acessa memória sequencialmente, maximizando prefetch do CPU

## Performance

- **Startup**: ~70ms para carregar o índice
- **Latência**: ~48ms por requisição (brute force 3M × 14 dims quantizados)
- **Memória runtime**: ~45MB por instância
- **Throughput**: ~20 req/s por instância (limitado por CPU, não memória)
