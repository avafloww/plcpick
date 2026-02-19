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

# Build with wgpu backend (cross-platform GPU via Vulkan/Metal/DX12)
cargo build --release --features wgpu

# Build with all GPU backends
cargo build --release --features gpu

# Run tests (CPU-only)
cargo test

# Run tests including CUDA (requires GPU)
cargo test --features cuda

# Run tests including wgpu (requires GPU)
cargo test --features wgpu

# Run a single test
cargo test test_name
cargo test --features wgpu test_name

# Run the miner (auto-detects best backend: CUDA > wgpu > CPU)
cargo run --release --features gpu -- 'pattern*' --handle user.bsky.social --pds https://bsky.social
cargo run --release --features gpu -- 'pattern*' --placeholder   # placeholder mode

# Explicit backend selection
cargo run --release --features wgpu -- 'pattern*' --placeholder --backend wgpu
cargo run --release -- 'pattern*' --placeholder --backend cpu
```

## Architecture

### Mining Pipeline

Each attempt: generate secp256k1 key → build signed PLC operation (CBOR) → SHA256 hash → truncate to 15 bytes → base32-encode → glob-match against pattern. The CPU backend does this naively per-thread. The GPU backends (CUDA and wgpu) run the entire pipeline on-GPU with a CBOR template patching strategy to avoid re-encoding the full operation each iteration.

### Module Layout

- **`src/main.rs`** — CLI (clap), backend auto-detection (`select_backend()`), progress spinner loop, match output
- **`src/plc.rs`** — `PlcOperation` struct, `build_signed_op()`, `did_suffix()`, `encode_did_key()`, `CborTemplate` (byte-offset patching system for GPU)
- **`src/pattern.rs`** — Pattern validation (base32 charset + wildcards only), `glob_match()`, `difficulty()` estimation
- **`src/output.rs`** — Terminal formatting (console/indicatif), `register_did()` HTTP call to plc.directory
- **`src/mining/mod.rs`** — `MiningBackend` trait, `MiningConfig`, `Match` struct
- **`src/mining/cpu.rs`** — Multi-threaded CPU backend
- **`src/mining/cuda.rs`** — CUDA backend via cudarc; includes extensive GPU-vs-CPU verification tests
- **`src/mining/wgpu_backend.rs`** — wgpu backend; two-pass WGSL compute pipeline (ec_pass + hash_pass)

### CUDA Kernel (`cuda/`)

- **`kernel.cu`** — Main `mine_kernel` entry point, `KernelParams` struct (must match Rust-side `KernelParams` exactly)
- **`secp256k1.cuh`** — 256-bit field/scalar arithmetic, Jacobian EC point operations, ECDSA signing
- **`sha256.cuh`** — SHA256 implementation
- **`hmac_drbg.cuh`** — HMAC-SHA256 + RFC 6979 deterministic nonce generation
- **`encoding.cuh`** — Base32, base58, base64url encoders
- **`pattern.cuh`** — Glob pattern matching

### wgpu Shaders (`wgpu/shaders/`)

Two-pass pipeline composed from modular WGSL files (concatenated at compile time via `include_str!()`):

- **`field.wgsl`** — u32-native secp256k1 field arithmetic (vendored from kangaroo, `mul32`-based)
- **`curve.wgsl`** — Jacobian point operations, `jac_to_affine`, 4-bit windowed `scalar_mul_g_windowed`
- **`scalar.wgsl`** — Scalar mod-n arithmetic for ECDSA (mul, inv via Fermat's little theorem, add, sub)
- **`sha256.wgsl`** — SHA256 (one byte per u32 element, no u8/u64)
- **`hmac_drbg.wgsl`** — HMAC-SHA256 + RFC 6979 deterministic nonce generation
- **`encoding.wgsl`** — Base32, base58, base64url encoders
- **`pattern.wgsl`** — Glob pattern matching
- **`ec_pass.wgsl`** — Pass 1: keygen + ECDSA signing (heavy EC math)
- **`hash_pass.wgsl`** — Pass 2: CBOR patching + SHA256 + base32 + pattern match (lightweight)

Both GPU backends use an incremental key strategy: each thread starts with `base_key + thread_id`, and between launches advances by `stride = total_threads`. Only the first launch does a full scalar multiplication; subsequent launches do a single point addition (`pubkey += stride_G`).

### Key Invariants

- CBOR byte length is **stable** for a given handle/PDS pair regardless of key — the `CborTemplate` system relies on this (see `cbor_template_length_stable` test which checks 1000 random keys)
- CBOR templates are ~306 bytes (unsigned) and ~398 bytes (signed) — GPU SHA256 buffers must be ≥512 bytes
- Base58 pubkey payload is always 48 chars; base64url signature is always 86 chars
- `did:key` encoding is always 57 chars (`did:key:z` + 48 base58)
- DID suffix is always 24 lowercase base32 chars (SHA256 of signed CBOR, truncated to 15 bytes)
- Rust-side `KernelParams` struct layout must match GPU-side exactly (repr(C), manual padding for CUDA)

### Feature Gating

- **`cuda`** — CUDA backend via cudarc (requires NVIDIA GPU + CUDA toolkit)
- **`wgpu`** — wgpu backend (cross-platform: Vulkan, Metal, DX12, WebGPU)
- **`gpu`** — Convenience feature enabling both `cuda` and `wgpu`

Backend auto-detection (`--backend auto`, the default) probes CUDA first, then wgpu, falling back to CPU.

### wgpu-Specific Notes

- WGSL has no u8 or u64 types — all byte values stored as individual u32 elements
- Shader modules composed via `include_str!()` concatenation (no `#include` in WGSL)
- G table [1*G..15*G] computed on CPU, uploaded as SSBO
- Two-pass pipeline: ec_pass (142 u32s per thread output) → hash_pass (reads ec_pass output)
- Match results read back via staging buffer with `map_async` + polling

## Rust Edition

Uses Rust **2024 edition** (`edition = "2024"` in Cargo.toml).
