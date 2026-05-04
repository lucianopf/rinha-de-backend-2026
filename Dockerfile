# Stage 1: Pre-process - build IVF search index from references
FROM oven/bun:1.2-alpine AS preprocess

WORKDIR /app

COPY package.json tsconfig.json ./
COPY src/ ./src/

# Copy resource files (provided by the competition runner)
COPY resources/references.json.gz ./resources/references.json.gz
COPY resources/normalization.json ./resources/normalization.json
COPY resources/mcc_risk.json ./resources/mcc_risk.json

# Install dependencies
RUN bun install --frozen-lockfile 2>/dev/null || bun install

# Run IVF pre-processing to build clustered search index
ENV RESOURCES_DIR=./resources
ENV OUTPUT_DIR=./data
RUN bun run src/ivf-preprocess.ts

# Stage 2: Runtime - lightweight image with pre-processed data
FROM oven/bun:1.2-alpine AS runtime

WORKDIR /app

COPY package.json tsconfig.json ./
COPY src/ ./src/

RUN bun install --frozen-lockfile --production 2>/dev/null || bun install --production

COPY --from=preprocess /app/data/ ./data/

ENV DATA_DIR=./data
ENV PORT=9999
ENV NUM_WORKERS=2

EXPOSE 9999

CMD ["bun", "run", "src/server-ivf-workers.ts"]
