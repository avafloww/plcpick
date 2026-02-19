# wgpu Migration Design

## Overview

Replace the vulkano-based Vulkan backend with a wgpu-based GPU backend. The key change is both the host API (vulkano → wgpu) and the shader arithmetic (GLSL uint64 → WGSL u32-native), which should unlock real GPU performance for the secp256k1 mining pipeline.

## Why

The current Vulkan backend produces correct results but runs at ~1 key/sec — the same speed whether dispatching 1 thread or 256. Root cause: GLSL `uint64_t` multiplication is silently emulated on most GPUs via multiple u32 ops. The arithmetic representation (not memory layout or workgroup count) is the bottleneck.

wgpu with WGSL forces u32-native arithmetic, matching what the hardware actually does. The kangaroo project (github.com/oritwoen/kangaroo) proves this works for secp256k1 at real speed across AMD, NVIDIA, Intel, and Apple Silicon.

## Dependencies

Replace:
- `vulkano = "0.35"` → `wgpu = "28"`
- `shaderc` (build-dep) → removed (WGSL is included as strings, no compilation step)
- Add `pollster = "0.4"` for async wgpu init

Feature flags:
- `vulkan` → `wgpu`
- `gpu = ["cuda", "wgpu"]`
- CLI: `--backend auto|cpu|cuda|wgpu`

## Shader Architecture: Two-Pass Pipeline

Instead of one monolithic mining kernel, split into two compute passes to avoid TDR (GPU timeout) on heavy EC math.

### Pass 1: EC Operations (`ec_pass.wgsl`)

Input: scalars (private keys), g_table, stride/stride_G, CBOR template metadata
Does: scalar_mul_G → compressed pubkey → HMAC-DRBG nonce → ECDSA sign
Output: per-thread (privkey_bytes[32], pubkey_bytes[33], signature_bytes[64])

This is the heavy pass — hundreds of u32 field multiplications per thread for EC point operations and modular inverse.

### Pass 2: Hash + Match (`hash_pass.wgsl`)

Input: CBOR templates, pubkeys + signatures from pass 1, pattern
Does: patch CBOR templates with pubkey/signature → SHA256 hash → base32 encode → glob match
Output: match results (privkey + DID suffix for any pattern hits)

This is lightweight — SHA256 is cheap compared to EC math.

### Shared Shader Libraries

```
wgpu/shaders/
  field.wgsl          — vendored from kangaroo: secp256k1 field arithmetic (u32-native)
  curve.wgsl          — vendored from kangaroo: Jacobian EC point operations
  sha256.wgsl         — SHA256 (u32-native, no uint64)
  hmac_drbg.wgsl      — HMAC-SHA256 + RFC 6979 nonce generation
  encoding.wgsl       — base32, base58, base64url
  pattern.wgsl        — glob matching
  ec_pass.wgsl        — Pass 1: keygen + ECDSA signing
  hash_pass.wgsl      — Pass 2: template hash + pattern match
```

### Field Arithmetic (from kangaroo)

Key difference from our current GLSL approach:
- 8 x u32 limbs (same representation)
- `mul32()` decomposes 32x32→64 using 16-bit half-words in pure u32
- No uint64 anywhere — all carry propagation in u32
- Optimized `fe_square()` with cross-term symmetry
- Addition-chain `fe_inv()` (255 squarings + 15 multiplications)

### Buffer Layout Between Passes

One persistent "thread results" buffer:
```
Per thread (129 bytes, padded to 132 for alignment):
  [0..32)   privkey (32 bytes, big-endian)
  [32..65)  compressed pubkey (33 bytes)
  [65..129) signature (64 bytes)
  [129..132) padding
```

Pass 1 writes this buffer, Pass 2 reads it.

## Host-Side: `src/mining/wgpu.rs`

```rust
pub struct WgpuBackend { pub device_index: usize }

impl MiningBackend for WgpuBackend {
    fn name(&self) -> &str { "wgpu" }
    fn run(&self, config, stop, total, tx) -> Result<()> {
        // 1. Init wgpu: instance → adapter → device + queue
        // 2. Create shader modules via include_str!()
        // 3. Create 2 compute pipelines (ec_pass, hash_pass)
        // 4. Allocate GPU buffers
        // 5. Init g_table (single dispatch or CPU-side)
        // 6. Generate initial scalars, upload
        // 7. Main loop:
        //    a. Dispatch ec_pass
        //    b. Dispatch hash_pass
        //    c. Read match_count, extract matches
        //    d. CPU re-verification
        //    e. Update total counter
    }
}
```

wgpu API is simpler than vulkano: no descriptor set allocators, no command buffer builders. Just pipelines, bind groups, and encoder.dispatch_workgroups().

## Auto-Detection

```rust
fn select_backend(threads: usize) -> Box<dyn MiningBackend> {
    #[cfg(feature = "cuda")]  { /* probe CUDA */ }
    #[cfg(feature = "wgpu")]  { /* probe wgpu adapter */ }
    Box::new(CpuBackend { threads })
}
```

Priority: CUDA > wgpu > CPU (CUDA likely faster on NVIDIA due to native uint64 and PTX).

## Testing

Same strategy as current Vulkan backend:
- Per-component test shaders: field_mul, field_sqr, SHA256, ECDSA, encoding, pattern
- Each pass testable independently (ec_pass output verification, hash_pass with known inputs)
- End-to-end diagnostic: pattern "*" with 256 threads, verify GPU suffix matches CPU

## What Gets Removed

- `vulkan/` directory (all GLSL shaders)
- `src/mining/vulkan.rs`
- vulkano + shaderc dependencies
- GLSL compilation in build.rs

## What Gets Kept

- All Rust host-side logic: CborTemplate, pattern matching, match verification, CLI
- CUDA backend (unchanged)
- CPU backend (unchanged)
- Mining algorithm: incremental key strategy, CBOR template patching
- Test infrastructure patterns (VulkanTestContext → WgpuTestContext)
