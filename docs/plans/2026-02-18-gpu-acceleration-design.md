# GPU Acceleration Design for plcpick

## Overview

Add CUDA GPU acceleration to plcpick's vanity DID:PLC mining loop. The entire mining pipeline (key generation, CBOR encoding, ECDSA signing, SHA256 hashing, pattern matching) runs on the GPU for maximum throughput. Architecture supports future backends (Vulkan, OpenCL, etc.) via a pluggable trait.

## Architecture

### Module Structure

```
src/
  main.rs              CLI, progress display, backend selection
  pattern.rs           Pattern validation + glob matching
  plc.rs               PLC op types, did_suffix, encode_did_key, build_signed_op
  output.rs            Styles, print_match, fmt_count, human, register_did
  mining/
    mod.rs             MiningBackend trait, MiningConfig, Match
    cpu.rs             CPU backend (refactored existing loop)
    cuda.rs            CUDA backend wrapper (feature = "cuda")
cuda/
  kernel.cu            Main mining kernel entry point
  secp256k1.cuh        256-bit field math + EC point operations
  sha256.cuh           SHA256
  hmac_drbg.cuh        HMAC-SHA256 + RFC 6979 nonce generation
  encoding.cuh         Base32, Base58, Base64url
  pattern.cuh          Glob matching
build.rs               Compiles .cu → .ptx via nvcc (only when cuda feature on)
```

### Feature Flags

- **default**: CPU backend only. Pure Rust, no external dependencies beyond current crates.
- **`cuda`**: Adds CUDA backend. Requires `nvcc` at build time. Adds `cudarc` as a dependency.
- Tests for CUDA gated behind `#[cfg(feature = "cuda")]`.

### MiningBackend Trait

```rust
pub struct MiningConfig {
    pub pattern: Vec<u8>,
    pub handle: String,
    pub pds: String,
    pub keep_going: bool,
}

pub struct Match {
    pub did: String,
    pub private_key_hex: String,
    pub signed_op: PlcOperation,
    pub attempts: u64,
    pub elapsed: Duration,
}

pub trait MiningBackend: Send + Sync {
    fn name(&self) -> &str;

    /// Start mining. Sends matches via tx, bumps total counter,
    /// stops when stop flag is set.
    fn run(
        &self,
        config: &MiningConfig,
        stop: &AtomicBool,
        total: &AtomicU64,
        tx: mpsc::Sender<Match>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
```

The `stop` flag is owned by the main thread. Main receives matches, and if `!keep_going`, sets `stop = true`. Backend checks `stop` periodically and exits.

## CBOR Templating Strategy

The CBOR byte layout is fixed-length for a given handle/PDS because all variable parts have constant encoded length:

- Compressed secp256k1 pubkey: always 33 bytes
- Multicodec prefix `(0xe7, 0x01)` + pubkey = always 35 bytes input to base58
- First byte 0xe7 (never 0x00): no leading-zero special cases in base58
- base58(35 bytes starting with 0xe7) = always 48 chars
- `did:key:z` + 48 chars = always 57 chars
- ECDSA signature: always 64 bytes → base64url_nopad = always 86 chars

Since string lengths are constant, DAG-CBOR length prefixes are constant (both 57 and 86 fall in the 24-255 range → 2-byte CBOR prefix each), and the total byte layout is fixed.

DAG-CBOR sorts map keys by length first, then lexicographically. In the signed template, `sig` (3 chars) is the shortest key and appears **first** in the map — right after the map header byte. This makes the signature patch location predictable and early in the byte stream.

### Template Construction (CPU side, once at startup)

1. Generate a dummy signing key
2. Build a complete signed `PlcOperation` using the dummy key
3. CBOR-encode the unsigned version (sig = None, field skipped)
4. CBOR-encode the signed version (sig = Some(base64url_string))
5. Locate byte offsets of:
   - Base58-encoded pubkey payload (2 locations in unsigned template: `rotationKeys[0]` and `verificationMethods.atproto`)
   - Base58-encoded pubkey payload (2 locations in signed template, same fields)
   - Base64url signature payload (1 location in signed template: `sig` field)
6. Pass templates + offsets to GPU as constant/global memory

### Template Patching (GPU side, per iteration)

1. Copy unsigned template to thread-local buffer
2. Patch base58 pubkey at 2 offsets
3. SHA256 → message hash (used for signing)
4. Copy signed template to thread-local buffer
5. Patch base58 pubkey at 2 offsets + base64url signature at 1 offset
6. SHA256 → DID hash
7. Base32 first 15 bytes → suffix
8. Pattern match

No CBOR serialization logic on GPU. Just memcpy + patch + hash.

**Note:** This assumption (fixed-length CBOR) is mathematically proven (see above) but we add a belt-and-suspenders startup assertion: CBOR-encode at least 1,000 random keys and confirm all produce identical template lengths. If the assertion fails, panic with a clear message rather than silently producing wrong DIDs.

## CUDA Kernel Design

### secp256k1 on GPU

**256-bit field arithmetic** (mod p, where p is the secp256k1 field prime):
- Represented as 8 × uint32 limbs
- Operations: add, subtract, multiply with Montgomery reduction, modular inverse via Fermat's little theorem
- Montgomery multiplication is the standard GPU approach (avoids expensive division)
- **Endianness:** secp256k1 scalars are big-endian by convention. CUDA is natively little-endian. All host↔device data transfer of scalars, coordinates, and signatures must include explicit byte-order conversion. This applies to: initial scalar setup, match data readback, and precomputed table values.

**EC point operations** (Jacobian coordinates to avoid per-operation modular inverse):
- Point addition
- Point doubling
- Scalar multiplication via 4-bit windowed method with precomputed table

**Precomputed G table** (GPU constant memory initially, may move to shared memory):
- Window size 4 bits: store [1·G, 2·G, ..., 15·G] in Jacobian coordinates
- Scalar mul processes 256 bits in 64 windows: 256 doublings + up to 64 additions
- Table size: 15 points × 96 bytes (3 × 32-byte coordinates) ≈ 1.4 KB
- Note: threads in a warp access different table indices (different nonces), which can serialize constant memory reads. If profiling shows stalls, move table to shared memory (loaded once per block).

### Incremental Key Optimization

Instead of generating a fully random key each iteration (requiring expensive full scalar multiplication for the pubkey), we use an incremental approach:

- Each GPU thread starts with scalar `base + thread_id`
- First iteration: full scalar multiplication `P = scalar * G` (expensive, done once)
- Subsequent iterations: `scalar += stride` (stride = total_threads), `P += stride_G` (one EC point addition)
- `stride_G = stride * G` is precomputed on CPU and passed as a kernel parameter
- Scalar increment must handle wrap at group order `n`: if `scalar >= n`, reduce modulo `n`

This makes pubkey derivation per iteration ~1 point addition instead of ~256 doublings + 64 additions.

ECDSA signing still requires one full scalar multiplication per iteration (for the nonce point `R = k * G`), so the overall savings are roughly halving the EC math cost.

### ECDSA Signing Per Iteration

1. SHA256 the patched unsigned CBOR template → `msg_hash`
2. RFC 6979 deterministic nonce: HMAC-DRBG(privkey, msg_hash) → nonce `k`. Must retry if `k` falls outside `[1, n)` (extremely rare for secp256k1 but must be handled correctly).
3. `R = k * G` (full scalar multiplication using windowed method + precomputed table)
4. `r = R.x mod n`
5. `s = k⁻¹ * (msg_hash + r * privkey) mod n`
6. **Low-s normalization (BIP-62):** if `s > n/2`, set `s = n - s`. ATProto/PLC expects low-s signatures.
7. Signature = `(r, s)` as 64 bytes, base64url-encoded to 86 chars

### Kernel Launch & Batching

CUDA kernels run in batches to avoid GPU watchdog timeouts and keep the system responsive.

**Launch model:**
1. Host launches kernel with N threads × B iterations per thread
2. Kernel runs batch, writes matches to output buffer in device memory
3. Host synchronizes, reads match count + match data
4. Host reconstructs full `PlcOperation` for any matches (for display/registration), then **re-hashes to verify** the DID suffix matches what the GPU reported. This catches any GPU-side bugs before presenting results to the user.
5. Host updates progress counter: `total += N * B`
6. Host checks stop flag, repeats

**Configuration:**
- N = auto-detected based on GPU SM count and occupancy, or user-configurable
- B = iterations per thread per launch (default ~256-1024, tunable)

**Thread state between launches:**
- Each thread's current scalar persists in device memory
- No re-initialization needed between launches
- First launch: host initializes per-thread scalars with random base + offsets

**Output buffer:**
- Fixed-size array in device memory: 64 match slots
- Each slot: 32 bytes (privkey scalar) + 64 bytes (signature) + 24 bytes (DID suffix) = 120 bytes
- Atomic match counter in device memory
- Host reads back after each kernel launch

## Testing Strategy

### Unit Tests (always compiled)

- `pattern.rs`: glob matching correctness, pattern validation, edge cases
- `plc.rs`: `did_suffix` produces correct output, `build_signed_op` round-trips, `encode_did_key` correctness
- `mining/cpu.rs`: CPU backend finds matches for trivial patterns

### CUDA Integration Tests (`#[cfg(feature = "cuda")]`)

- **Correctness test**: Give GPU a known private key scalar, verify it produces the same DID suffix as the CPU code path. This is the critical test — proves CUDA reimplementation matches Rust crypto libraries.
- **Match test**: Mine with an easy pattern (like `a*`), verify CUDA finds valid matches that CPU confirms.
- **Consistency test**: Run the same starting scalar through CPU and CUDA paths, compare intermediate values (pubkey, unsigned CBOR hash, signature, signed CBOR hash, suffix).
- **Endianness round-trip test**: Pass a known scalar from Rust to CUDA and back, verify byte order is preserved correctly through the host↔device boundary.
- **Low-s normalization test**: Verify GPU-produced signatures always have `s <= n/2` and that CPU-side verification (k256 crate) accepts them.
- **Scalar wrap test**: Initialize a thread's scalar near the group order `n`, run several increments, verify modular reduction produces correct pubkeys.

### CUDA Component Tests (`#[cfg(feature = "cuda")]`)

Small "test kernels" that exercise individual GPU functions and write results to a host-readable buffer:

- secp256k1 point addition/doubling against known test vectors
- Scalar multiplication against known test vectors
- SHA256 against NIST test vectors
- Base58/Base64/Base32 encoding against known values
- RFC 6979 nonce generation against test vectors from the RFC

## Build System

### `build.rs` (when `cuda` feature enabled)

1. Find `nvcc` in PATH
2. Compile `cuda/kernel.cu` → PTX
3. Place PTX file in `OUT_DIR`
4. `cuda.rs` loads PTX at runtime via `cudarc`

### Dependencies Added (cuda feature only)

- `cudarc`: Rust bindings for CUDA driver API (kernel loading, memory management, launch)

## CLI Changes

- Add `--backend` flag: `cpu` (default) or `cuda` (requires cuda feature)
- CUDA backend auto-detects GPU and prints device info in header
- Progress display works the same regardless of backend

## Future Optimizations

These are not part of the initial implementation but worth noting for later:

- **SHA256 midstate**: The CBOR template has a constant prefix before the first variable field. This prefix can be pre-hashed into a SHA256 midstate, so each GPU thread starts hashing from the first variable block instead of byte 0. Easy win once the basic kernel works.
- **G-table in shared memory**: If profiling shows constant memory bank conflicts during scalar multiplication, move the precomputed G table to shared memory (loaded once per block at kernel start).
- **Larger window for scalar mul**: 8-bit window (256 table entries, ~24 KB) reduces scalar mul to 32 additions + 256 doublings. Trades shared memory for fewer operations.

## Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| CBOR template length assumption wrong | Startup assertion checks multiple random keys. Fall back to GPU-side CBOR if needed. |
| secp256k1 GPU implementation has subtle bugs | Extensive test vectors. CPU vs GPU consistency checks. |
| RFC 6979 nonce gen is complex to implement in CUDA | Use test vectors from the RFC itself. Compare against k256 crate output. |
| GPU watchdog kills long kernels | Batched launch model with tunable iterations per batch. |
| cudarc API instability | Pin version in Cargo.toml. |
| Host↔device endianness mismatch | Explicit byte-order conversion at all transfer points. Round-trip test. |
| Scalar increment wraps past group order | Modular reduction on increment. Test with scalar near n. |
