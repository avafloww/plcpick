# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is plcpick?

A vanity `did:plc` identifier miner for AT Protocol (Bluesky). It brute-force searches for secp256k1 keys whose resulting DID matches a user-supplied glob pattern (e.g. `grug*`, `*cool*`). DID suffixes are 24-char base32-encoded truncated SHA256 hashes of a signed CBOR PLC operation.

## Build & Test Commands

```bash
# Build (CPU-only, default)
cargo build --release

# Build with CUDA backend (requires nvcc / CUDA toolkit)
cargo build --release --features cuda

# Build with Vulkan backend (requires Vulkan SDK / shaderc)
cargo build --release --features vulkan

# Build with all GPU backends
cargo build --release --features gpu

# Run tests (CPU-only)
cargo test

# Run tests including CUDA (requires GPU)
cargo test --features cuda

# Run tests including Vulkan (requires GPU)
cargo test --features vulkan

# Run a single test
cargo test test_name
cargo test --features vulkan test_name

# Run the miner (auto-detects best backend: CUDA > Vulkan > CPU)
cargo run --release --features gpu -- 'pattern*' --handle user.bsky.social --pds https://bsky.social
cargo run --release --features gpu -- 'pattern*' --placeholder   # placeholder mode

# Explicit backend selection
cargo run --release --features vulkan -- 'pattern*' --placeholder --backend vulkan
cargo run --release -- 'pattern*' --placeholder --backend cpu
```

## Architecture

### Mining Pipeline

Each attempt: generate secp256k1 key → build signed PLC operation (CBOR) → SHA256 hash → truncate to 15 bytes → base32-encode → glob-match against pattern. The CPU backend does this naively per-thread. The GPU backends (CUDA and Vulkan) run the entire pipeline on-GPU with a CBOR template patching strategy to avoid re-encoding the full operation each iteration.

### Module Layout

- **`src/main.rs`** — CLI (clap), backend auto-detection (`select_backend()`), progress spinner loop, match output
- **`src/plc.rs`** — `PlcOperation` struct, `build_signed_op()`, `did_suffix()`, `encode_did_key()`, `CborTemplate` (byte-offset patching system for GPU)
- **`src/pattern.rs`** — Pattern validation (base32 charset + wildcards only), `glob_match()`, `difficulty()` estimation
- **`src/output.rs`** — Terminal formatting (console/indicatif), `register_did()` HTTP call to plc.directory
- **`src/mining/mod.rs`** — `MiningBackend` trait, `MiningConfig`, `Match` struct
- **`src/mining/cpu.rs`** — Multi-threaded CPU backend
- **`src/mining/cuda.rs`** — CUDA backend via cudarc; includes extensive GPU-vs-CPU verification tests
- **`src/mining/vulkan.rs`** — Vulkan backend via vulkano; GLSL compute shaders compiled to SPIR-V at build time

### CUDA Kernel (`cuda/`)

- **`kernel.cu`** — Main `mine_kernel` entry point, `KernelParams` struct (must match Rust-side `KernelParams` exactly)
- **`secp256k1.cuh`** — 256-bit field/scalar arithmetic, Jacobian EC point operations, ECDSA signing
- **`sha256.cuh`** — SHA256 implementation
- **`hmac_drbg.cuh`** — HMAC-SHA256 + RFC 6979 deterministic nonce generation
- **`encoding.cuh`** — Base32, base58, base64url encoders
- **`pattern.cuh`** — Glob pattern matching

### Vulkan Shaders (`vulkan/`)

- **`mine.comp`** — Main mining kernel (same pipeline as CUDA), 9 SSBOs + push constants
- **`init_g_table.comp`** — Single-invocation shader to populate G table with [1*G..15*G]
- **`stride_g.comp`** — Computes stride*G for incremental key optimization
- **`secp256k1.glsl`** — 256-bit field/scalar arithmetic, Jacobian EC point operations, 4-bit windowed scalar_mul_G
- **`sha256.glsl`** — SHA256 with `SHA256_MAX_INPUT` (512) byte buffer for CBOR templates
- **`hmac_drbg.glsl`** — HMAC-SHA256 + RFC 6979 deterministic nonce generation
- **`encoding.glsl`** — Base32, base58, base64url encoders
- **`pattern.glsl`** — Glob pattern matching
- **`test_*.comp`** — Per-component test shaders for GPU-vs-CPU verification

Both GPU backends use an incremental key strategy: each thread starts with `base_key + thread_id`, and between launches advances by `stride = total_threads`. Only the first launch does a full scalar multiplication; subsequent launches do a single point addition (`pubkey += stride_G`).

### Key Invariants

- CBOR byte length is **stable** for a given handle/PDS pair regardless of key — the `CborTemplate` system relies on this (see `cbor_template_length_stable` test which checks 1000 random keys)
- CBOR templates are ~306 bytes (unsigned) and ~398 bytes (signed) — GPU SHA256 buffers must be ≥512 bytes (`SHA256_MAX_INPUT`)
- Base58 pubkey payload is always 48 chars; base64url signature is always 86 chars
- `did:key` encoding is always 57 chars (`did:key:z` + 48 base58)
- DID suffix is always 24 lowercase base32 chars (SHA256 of signed CBOR, truncated to 15 bytes)
- Rust-side `KernelParams` struct layout must match GPU-side exactly (repr(C), manual padding for CUDA; std430 SSBO for Vulkan)

### Feature Gating

- **`cuda`** — CUDA backend via cudarc (requires NVIDIA GPU + CUDA toolkit)
- **`vulkan`** — Vulkan backend via vulkano + shaderc build-time GLSL→SPIR-V compilation
- **`gpu`** — Convenience feature enabling both `cuda` and `vulkan`

Backend auto-detection (`--backend auto`, the default) probes CUDA first, then Vulkan, falling back to CPU.

### Vulkan-Specific Notes

- Requires `GL_EXT_shader_explicit_arithmetic_types_int8`, `GL_EXT_shader_8bit_storage`, `GL_EXT_shader_explicit_arithmetic_types_int64`
- Device features: `shaderInt8`, `storageBuffer8BitAccess`, `uniformAndStorageBuffer8BitAccess`, `shaderInt64`
- All data passes through std430 SSBOs (not std140 UBOs) to avoid alignment traps
- Push constants used for per-dispatch values (`is_first_launch`, `iterations_per_thread`, `max_matches`)
- GLSL `#include` directives resolved by shaderc at build time from `vulkan/` directory

## Rust Edition

Uses Rust **2024 edition** (`edition = "2024"` in Cargo.toml).
