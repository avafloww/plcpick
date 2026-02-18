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

# Run tests (CPU-only)
cargo test

# Run tests including CUDA (requires GPU)
cargo test --features cuda

# Run a single test
cargo test test_name
cargo test --features cuda test_name

# Run the miner
cargo run --release -- 'pattern*' --handle user.bsky.social --pds https://bsky.social
cargo run --release -- 'pattern*' --placeholder   # placeholder mode, no real handle/PDS
```

## Architecture

### Mining Pipeline

Each attempt: generate secp256k1 key → build signed PLC operation (CBOR) → SHA256 hash → truncate to 15 bytes → base32-encode → glob-match against pattern. The CPU backend does this naively per-thread. The CUDA backend runs the entire pipeline on-GPU with a CBOR template patching strategy to avoid re-encoding the full operation each iteration.

### Module Layout

- **`src/main.rs`** — CLI (clap), backend selection, progress spinner loop, match output
- **`src/plc.rs`** — `PlcOperation` struct, `build_signed_op()`, `did_suffix()`, `encode_did_key()`, `CborTemplate` (byte-offset patching system for GPU)
- **`src/pattern.rs`** — Pattern validation (base32 charset + wildcards only), `glob_match()`, `difficulty()` estimation
- **`src/output.rs`** — Terminal formatting (console/indicatif), `register_did()` HTTP call to plc.directory
- **`src/mining/mod.rs`** — `MiningBackend` trait, `MiningConfig`, `Match` struct
- **`src/mining/cpu.rs`** — Multi-threaded CPU backend
- **`src/mining/cuda.rs`** — CUDA backend via cudarc; includes extensive GPU-vs-CPU verification tests

### CUDA Kernel (`cuda/`)

- **`kernel.cu`** — Main `mine_kernel` entry point, `KernelParams` struct (must match Rust-side `KernelParams` exactly)
- **`secp256k1.cuh`** — 256-bit field/scalar arithmetic, Jacobian EC point operations, ECDSA signing
- **`sha256.cuh`** — SHA256 implementation
- **`hmac_drbg.cuh`** — HMAC-SHA256 + RFC 6979 deterministic nonce generation
- **`encoding.cuh`** — Base32, base58, base64url encoders
- **`pattern.cuh`** — Glob pattern matching

The GPU uses an incremental key strategy: each thread starts with `base_key + thread_id`, and between launches advances by `stride = total_threads`. Only the first launch does a full scalar multiplication; subsequent launches do a single point addition (`pubkey += stride_G`).

### Key Invariants

- CBOR byte length is **stable** for a given handle/PDS pair regardless of key — the `CborTemplate` system relies on this (see `cbor_template_length_stable` test which checks 1000 random keys)
- Base58 pubkey payload is always 48 chars; base64url signature is always 86 chars
- `did:key` encoding is always 57 chars (`did:key:z` + 48 base58)
- DID suffix is always 24 lowercase base32 chars (SHA256 of signed CBOR, truncated to 15 bytes)
- Rust-side `KernelParams` struct layout must match CUDA-side exactly (repr(C), manual padding)

### Feature Gating

The `cuda` feature gates all GPU code. Without it, only the CPU backend is available. CUDA code uses `#[cfg(feature = "cuda")]` and `#[cfg_attr(not(feature = "cuda"), allow(dead_code))]` for types shared between backends.

## Rust Edition

Uses Rust **2024 edition** (`edition = "2024"` in Cargo.toml).
