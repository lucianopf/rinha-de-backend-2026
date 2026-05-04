# Stage 1: Pre-process - build quantized search index from references
FROM oven/bun:1.2-alpine AS preprocess

WORKDIR /app

# Copy source and package files
COPY bun-solution/package.json bun-solution/tsconfig.json ./
COPY bun-solution/src/ ./src/

# Copy resource files from repository root
COPY resources/references.json.gz ./resources/references.json.gz
COPY resources/normalization.json ./resources/normalization.json
COPY resources/mcc_risk.json ./resources/mcc_risk.json

# Install dependencies
RUN bun install --frozen-lockfile 2>/dev/null || bun install

# Run pre-processing to build quantized search index
ENV RESOURCES_DIR=./resources
ENV OUTPUT_DIR=./data
RUN bun run src/preprocess.ts

# Stage 2: Runtime - lightweight image with pre-processed data
FROM oven/bun:1.2-alpine AS runtime

WORKDIR /app

COPY bun-solution/package.json bun-solution/tsconfig.json ./
COPY bun-solution/src/ ./src/

RUN bun install --frozen-lockfile --production 2>/dev/null || bun install --production

COPY --from=preprocess /app/data/ ./data/

ENV DATA_DIR=./data
ENV PORT=9999

EXPOSE 9999

CMD ["bun", "run", "src/server.ts"]
