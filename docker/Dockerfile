# Dockerfile — BioStructurePrediction inference container
#
# Base: CUDA 12.6 + cuDNN 9 on Ubuntu 22.04
# Installs: Julia 1.10, HMMER 3.4, zstd, wget, and all Julia package deps.
#
# Build:
#   docker build -t biostructure:latest -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -v /data/models:/models -v /data/dbs:/databases \
#     -v /data/input:/input -v /data/output:/output \
#     biostructure:latest \
#     julia run_prediction.jl \
#       --json_path /input/fold_input.json \
#       --output_dir /output \
#       --model_dir /models \
#       --db_dir /databases

FROM nvidia/cuda:12.6.0-cudnn9-runtime-ubuntu22.04

LABEL maintainer="BioStructurePrediction"
LABEL description="AlphaFold 3-style structure prediction in Julia"
LABEL version="3.0.2"

# ── Build arguments ────────────────────────────────────────────────────────────
ARG JULIA_VERSION=1.10.4
ARG HMMER_VERSION=3.4
ARG JULIA_URL=https://julialang-s3.julialang.org/bin/linux/x64/1.10
ARG HMMER_URL=http://eddylab.org/software/hmmer

# ── Environment ────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV JULIA_HOME=/opt/julia
ENV PATH="${JULIA_HOME}/bin:${PATH}"
ENV JULIA_DEPOT_PATH=/opt/julia_depot
ENV JULIA_NUM_THREADS=8
ENV JULIA_CUDA_USE_BINARYBUILDER=false
ENV MODEL_PARAMETERS_PATH=/models
ENV DB_DIR=/databases

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl ca-certificates \
        git \
        build-essential \
        libhdf5-dev \
        zstd \
        libgomp1 \
        # For HMMER build
        make \
        gcc \
        # Useful utilities
        less \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Julia installation ────────────────────────────────────────────────────────
RUN wget -q "${JULIA_URL}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" -O /tmp/julia.tar.gz \
    && tar -xzf /tmp/julia.tar.gz -C /opt \
    && mv "/opt/julia-${JULIA_VERSION}" "$JULIA_HOME" \
    && rm /tmp/julia.tar.gz \
    && julia --version

# ── HMMER installation ─────────────────────────────────────────────────────────
RUN wget -q "${HMMER_URL}/hmmer-${HMMER_VERSION}.tar.gz" -O /tmp/hmmer.tar.gz \
    && tar -xzf /tmp/hmmer.tar.gz -C /tmp \
    && cd "/tmp/hmmer-${HMMER_VERSION}" \
    # Apply sequence limit patch if present
    && if [ -f /build/docker/jackhmmer_seq_limit.patch ]; then \
           patch -p1 < /build/docker/jackhmmer_seq_limit.patch; \
       fi \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && cd / \
    && rm -rf /tmp/hmmer.tar.gz "/tmp/hmmer-${HMMER_VERSION}" \
    && jackhmmer -h | head -3 \
    && nhmmer  -h | head -3 \
    && hmmbuild -h | head -3

ENV JACKHMMER_BINARY_PATH=/usr/local/bin/jackhmmer
ENV NHMMER_BINARY_PATH=/usr/local/bin/nhmmer
ENV HMMBUILD_BINARY_PATH=/usr/local/bin/hmmbuild
ENV HMMSEARCH_BINARY_PATH=/usr/local/bin/hmmsearch
ENV HMMALIGN_BINARY_PATH=/usr/local/bin/hmmalign

# ── Copy project ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app/

# ── Julia package pre-installation ────────────────────────────────────────────
RUN mkdir -p "$JULIA_DEPOT_PATH" \
    && julia --project=/app -e 'using Pkg; Pkg.instantiate()' \
    && julia --project=/app -e '
        using Pkg
        Pkg.precompile()
        # Test basic imports
        include("src/BioStructurePrediction/__init__.jl")
        using .BioStructurePrediction
        @info "Package loaded OK: v$(BioStructurePrediction.VERSION_STRING)"
    '

# ── Create volume mount points ─────────────────────────────────────────────────
RUN mkdir -p /models /databases /input /output

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD julia --project=/app -e 'using Pkg; @info "healthy"' || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["julia", "--project=/app", "run_prediction.jl"]
CMD ["--help"]
