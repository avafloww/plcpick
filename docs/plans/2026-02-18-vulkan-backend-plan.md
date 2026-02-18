# Vulkan Backend + Auto-Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Vulkan compute backend so plcpick works on non-NVIDIA GPUs, plus automatic backend detection.

**Architecture:** Port the existing CUDA mining kernel to GLSL compute shaders compiled to SPIR-V via shaderc. Use vulkano for Vulkan host-side integration. Auto-detection tries CUDA first (NVIDIA), then Vulkan (any GPU), then CPU fallback.

**Tech Stack:** vulkano 0.35, shaderc 0.10, GLSL 450 with `GL_EXT_shader_explicit_arithmetic_types_int8` + `GL_EXT_shader_8bit_storage` + `GL_EXT_shader_explicit_arithmetic_types_int64` extensions, SPIR-V

**Design doc:** `docs/plans/2026-02-18-vulkan-backend-design.md`

---

## Phase 1: Foundation

### Task 1: Add Feature Flags and Dependencies

**Files:**
- Modify: `Cargo.toml`

**Step 1: Add vulkan dependencies and feature flags**

Add to `Cargo.toml`:

```toml
[features]
cuda = ["dep:cudarc"]
vulkan = ["dep:vulkano", "dep:shaderc"]
gpu = ["cuda", "vulkan"]

[dependencies]
vulkano = { version = "0.35", optional = true }

[build-dependencies]
shaderc = { version = "0.10", optional = true }

[features]
vulkan = ["dep:vulkano", "dep:shaderc"]
```

Note: `shaderc` is a build dependency (used in build.rs to compile GLSL → SPIR-V), but it needs to be listed under `[dependencies]` with `optional = true` for the feature flag to work. Alternatively, we can use the feature flag in build.rs via `cfg`. Actually — `shaderc` is only needed at build time. We'll check `cfg(feature = "vulkan")` in build.rs and use `shaderc` as a build-dependency. Build dependencies can't be optional via features in the normal way, so we'll make `shaderc` a regular optional dependency and use it in build.rs.

**Step 2: Verify it compiles**

Run: `cargo check`
Expected: Compiles with no errors (no vulkan code yet, just deps declared)

**Step 3: Verify feature combinations**

Run: `cargo check --features vulkan && cargo check --features cuda && cargo check --features gpu`
Expected: All compile cleanly

**Step 4: Commit**

```
feat: add vulkan and gpu feature flags to Cargo.toml
```

---

### Task 2: Build System — GLSL to SPIR-V Compilation

**Files:**
- Modify: `build.rs`
- Create: `vulkan/mine.comp` (stub)

**Step 1: Create a minimal stub shader**

Create `vulkan/mine.comp`:
```glsl
#version 450
layout(local_size_x = 256) in;
void main() {
    // stub
}
```

**Step 2: Add Vulkan build step to build.rs**

Add a `build_vulkan()` function that uses `shaderc` to compile all `.comp` files in `vulkan/` to SPIR-V, writing them to `OUT_DIR`. Support `#include "file.glsl"` via shaderc's include callback, resolving relative to the `vulkan/` directory.

```rust
#[cfg(feature = "vulkan")]
fn build_vulkan() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let compiler = shaderc::Compiler::new().expect("failed to create shaderc compiler");
    let mut options = shaderc::CompileOptions::new().expect("failed to create compile options");

    // Set up include resolution from vulkan/ directory
    options.set_include_callback(|name, _type, _source, _depth| {
        let path = format!("vulkan/{name}");
        match std::fs::read_to_string(&path) {
            Ok(content) => Ok(shaderc::ResolvedInclude {
                resolved_name: path,
                content,
            }),
            Err(e) => Err(format!("Failed to include {path}: {e}")),
        }
    });

    // Compile each .comp file
    for entry in std::fs::read_dir("vulkan").expect("vulkan/ directory") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "comp") {
            let name = path.file_name().unwrap().to_str().unwrap();
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
            let spirv = compiler
                .compile_into_spirv(
                    &source,
                    shaderc::ShaderKind::Compute,
                    name,
                    "main",
                    Some(&options),
                )
                .unwrap_or_else(|e| panic!("failed to compile {name}: {e}"));
            let spv_path = format!("{out_dir}/{name}.spv");
            std::fs::write(&spv_path, spirv.as_binary_u8())
                .unwrap_or_else(|e| panic!("failed to write {spv_path}: {e}"));
        }
    }

    println!("cargo:rerun-if-changed=vulkan/");
}
```

Update `main()` in build.rs:
```rust
fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
    #[cfg(feature = "vulkan")]
    build_vulkan();
}
```

**Step 3: Verify the stub compiles to SPIR-V**

Run: `cargo build --features vulkan 2>&1`
Expected: Builds successfully, `target/*/build/plcpick-*/out/mine.comp.spv` exists

**Step 4: Commit**

```
feat: add shaderc GLSL→SPIR-V build pipeline for Vulkan shaders
```

---

### Task 3: Vulkan Test Infrastructure + Trivial Shader Test

**Files:**
- Create: `src/mining/vulkan.rs`
- Modify: `src/mining/mod.rs`

**Step 1: Create vulkan.rs with test helper and trivial test**

Create `src/mining/vulkan.rs` with:
- A `setup()` helper function that creates a Vulkan instance, selects the first compute-capable physical device, creates a logical device + compute queue
- A trivial test that loads the stub mine.comp SPIR-V, creates a compute pipeline, dispatches it with 1 workgroup, and verifies it completes without error
- The `VulkanBackend` struct (empty impl of `MiningBackend` for now — just return `Ok(())`)

This validates the entire Vulkan pipeline: instance → device → shader loading → pipeline → command buffer → dispatch → fence-wait.

**Step 2: Add vulkan module to mod.rs**

Add to `src/mining/mod.rs`:
```rust
#[cfg(feature = "vulkan")]
pub mod vulkan;
```

**Step 3: Run the test**

Run: `cargo test --features vulkan vulkan_trivial_dispatch`
Expected: PASS (requires a Vulkan-capable GPU)

**Step 4: Commit**

```
feat: Vulkan test infrastructure and trivial dispatch test
```

---

## Phase 2: GLSL Shader Porting

Each shader file is ported from the corresponding CUDA header. Tests dispatch small test shaders and compare results against CPU reference values (same approach as the CUDA tests).

**GLSL conventions:**
- `#version 450`
- `#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require` for `uint8_t`
- `#extension GL_EXT_shader_8bit_storage : require` for uint8 in SSBOs
- `#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require` for `uint64_t` (needed by field_mul/scalar_mul 64-bit intermediate products)
- U256 represented as `uint d[8]` (little-endian limbs, same as CUDA)
- JacobianPoint as `struct { uint x[8]; uint y[8]; uint z[8]; }`
- AffinePoint as `struct { uint x[8]; uint y[8]; }`

**Required Vulkan device features** (checked during auto-detection and device creation):
- `shaderInt8` — 8-bit integer arithmetic in shaders
- `storageBuffer8BitAccess` — uint8 in SSBOs
- `shaderInt64` — 64-bit integer arithmetic (critical for field/scalar multiplication)

### Task 4: secp256k1.glsl — U256 Type and Field Arithmetic

**Files:**
- Create: `vulkan/secp256k1.glsl`
- Create: `vulkan/test_field_mul.comp` (test shader)
- Modify: `src/mining/vulkan.rs` (add test)

**Step 1: Write test_field_mul.comp test shader**

A compute shader that reads two U256 values from an SSBO, calls `field_mul`, writes the result back. This tests field multiplication (mod p).

**Step 2: Write the Rust test `vulkan_field_mul_gx_gy`**

Same test logic as `gpu_field_mul_gx_gy` in cuda.rs: multiply GX * GY mod p, compare against num-bigint reference.

**Step 3: Port secp256k1.glsl — U256 basics + field operations**

Port from `cuda/secp256k1.cuh`:
- U256 struct (8 × uint LE limbs)
- `add256`, `sub256`, `cmp` — 256-bit basic arithmetic
- `field_mul` — multiplication mod p (secp256k1 field prime)
- `field_sqr`, `field_inv` — squaring and inverse via Fermat's little theorem
- `field_add`, `field_sub` — addition/subtraction mod p
- Constants: `P` (field prime), `GX`, `GY`, `N`, `N_HALF`
- `load_scalar`, `store_scalar` — byte↔limb conversion

**Step 4: Run test**

Run: `cargo test --features vulkan vulkan_field_mul_gx_gy`
Expected: PASS

**Step 5: Commit**

```
feat(vulkan): port secp256k1 U256 type and field arithmetic to GLSL
```

---

### Task 5: secp256k1.glsl — Scalar Arithmetic

**Files:**
- Modify: `vulkan/secp256k1.glsl`
- Create: `vulkan/test_scalar_mul.comp`, `vulkan/test_scalar_inv.comp`
- Modify: `src/mining/vulkan.rs` (add tests)

**Step 1: Write test shaders and Rust tests**

Port from CUDA tests:
- `vulkan_scalar_mul_mod_n` — scalar_mul(GX, GY) mod n
- `vulkan_scalar_mul_squaring` — scalar_mul(n-1, n-1) mod n
- `vulkan_scalar_inv_mod_n` — scalar_inv(7) mod n

**Step 2: Port scalar operations to secp256k1.glsl**

- `scalar_mul` — multiplication mod n (group order)
- `scalar_add` — addition mod n
- `scalar_inv` — inverse mod n via Fermat's little theorem (n-2 exponentiation)

**Step 3: Run tests**

Run: `cargo test --features vulkan vulkan_scalar`
Expected: All PASS

**Step 4: Commit**

```
feat(vulkan): port secp256k1 scalar arithmetic to GLSL
```

---

### Task 6: secp256k1.glsl — EC Point Operations

**Files:**
- Modify: `vulkan/secp256k1.glsl`
- Create: `vulkan/test_point_double_g.comp`, `vulkan/test_scalar_mul_G.comp`
- Modify: `src/mining/vulkan.rs` (add tests)

**Step 1: Write test shaders and Rust tests**

Port from CUDA tests:
- `vulkan_point_double_g` — point_double(G) must equal known 2*G
- `vulkan_scalar_mul_g_identity` — 1*G must equal G
- `vulkan_scalar_mul_g_small_scalars` — test scalars 2, 3, 15, 16, 17, 256
- `vulkan_scalar_mul_g_matches_k256` — random scalar, compare against k256 crate

**Step 2: Port EC point operations to secp256k1.glsl**

- JacobianPoint struct
- AffinePoint struct + G_TABLE (shared via SSBO, initialized by init_g_table)
- `point_double` — Jacobian doubling
- `point_add` — Jacobian addition
- `point_add_affine` — mixed Jacobian + affine addition
- `jacobian_to_affine` — convert to affine (requires field_inv)
- `get_compressed_pubkey` — serialize to 33 bytes (02/03 prefix + x)
- `scalar_mul_G` — 4-bit windowed method using G_TABLE

**Step 3: Run tests**

Run: `cargo test --features vulkan vulkan_point && cargo test --features vulkan vulkan_scalar_mul_g`
Expected: All PASS

**Step 4: Commit**

```
feat(vulkan): port secp256k1 EC point operations to GLSL
```

---

### Task 7: sha256.glsl

**Files:**
- Create: `vulkan/sha256.glsl`
- Create: `vulkan/test_sha256.comp`
- Modify: `src/mining/vulkan.rs` (add tests)

**Step 1: Write test shader and Rust tests**

Port from CUDA tests:
- `vulkan_sha256_matches_cpu` — SHA256("abc") matches CPU
- `vulkan_sha256_empty_input` — SHA256("") matches CPU

**Step 2: Port SHA256 to GLSL**

Port from `cuda/sha256.cuh`:
- SHA256 constants (K table, initial hash values)
- `sha256_hash(uint8_t[] data, uint len, uint8_t[32] out)` function
- Message padding, block processing, final hash

Note: GLSL doesn't have dynamic arrays on stack. Use fixed-size arrays sized for the max template length (512 bytes is sufficient — the CUDA kernel uses the same limit).

**Step 3: Run tests**

Run: `cargo test --features vulkan vulkan_sha256`
Expected: PASS

**Step 4: Commit**

```
feat(vulkan): port SHA256 to GLSL compute shader
```

---

### Task 8: hmac_drbg.glsl

**Files:**
- Create: `vulkan/hmac_drbg.glsl`
- Create: `vulkan/test_ecdsa_sign.comp`
- Modify: `src/mining/vulkan.rs` (add test)

**Step 1: Write test shader and Rust test**

Port from CUDA tests:
- `vulkan_ecdsa_sign_matches_k256` — full ECDSA sign (RFC 6979 nonce + sign), compare against k256

This tests HMAC-DRBG + ECDSA together, since the nonce generation is only useful in the context of signing.

**Step 2: Port HMAC-DRBG to GLSL**

Port from `cuda/hmac_drbg.cuh`:
- `hmac_sha256` — HMAC using SHA256
- `rfc6979_nonce` — deterministic nonce generation per RFC 6979

**Step 3: Run test**

Run: `cargo test --features vulkan vulkan_ecdsa_sign`
Expected: PASS

**Step 4: Commit**

```
feat(vulkan): port HMAC-DRBG and RFC 6979 nonce generation to GLSL
```

---

### Task 9: encoding.glsl

**Files:**
- Create: `vulkan/encoding.glsl`
- Create: `vulkan/test_base58.comp`, `vulkan/test_base64url.comp`, `vulkan/test_base32.comp`
- Modify: `src/mining/vulkan.rs` (add tests)

**Step 1: Write test shaders and Rust tests**

Port from CUDA tests:
- `vulkan_base58_matches_cpu` — base58 encode 35-byte multicodec pubkey
- `vulkan_base64url_matches_cpu` — base64url encode 64-byte signature
- `vulkan_base32_matches_cpu` — base32 encode 15-byte SHA256 prefix

**Step 2: Port encoding functions to GLSL**

Port from `cuda/encoding.cuh`:
- `base58_encode_35bytes` — fixed-size base58 encoding for multicodec pubkey
- `base64url_encode_64bytes` — fixed-size base64url encoding for signature
- `base32_encode_15bytes` — fixed-size base32 encoding for DID suffix

**Step 3: Run tests**

Run: `cargo test --features vulkan vulkan_base`
Expected: All PASS

**Step 4: Commit**

```
feat(vulkan): port base32/base58/base64url encoding to GLSL
```

---

### Task 10: pattern.glsl

**Files:**
- Create: `vulkan/pattern.glsl`

**Step 1: Port glob matching to GLSL**

Port from `cuda/pattern.cuh`:
- `glob_match(pattern, pattern_len, text, text_len)` → bool

This is straightforward — simple character comparison and `*` wildcard handling. No test shader needed since the pattern matching is simple and will be validated end-to-end.

**Step 2: Commit**

```
feat(vulkan): port glob pattern matching to GLSL
```

---

## Phase 3: Mining Kernels

### Task 11: Utility Compute Shaders (init_g_table + stride_g)

**Files:**
- Create: `vulkan/init_g_table.comp`
- Create: `vulkan/stride_g.comp`

**Step 1: Write init_g_table.comp**

Single-invocation compute shader that populates a G_TABLE SSBO with [1*G, 2*G, ..., 15*G] as affine points. Port from CUDA's `init_g_table` kernel.

Layout:
```glsl
layout(set = 0, binding = 0) buffer GTable { uint data[]; } g_table;
```

**Step 2: Write stride_g.comp**

Single-invocation compute shader that computes `stride * G` for the incremental key optimization. Port from CUDA's `compute_stride_g` kernel.

Layout:
```glsl
layout(push_constant) uniform PushConstants { uint stride_val; };
layout(set = 0, binding = 0) buffer GTable { uint data[]; } g_table; // readonly, from init_g_table
layout(set = 0, binding = 1) buffer StrideOut { uint data[8]; } stride_out;
layout(set = 0, binding = 2) buffer StrideGOut { uint data[24]; } stride_g_out;
```

**Step 3: Verify compilation**

Run: `cargo build --features vulkan`
Expected: Compiles (SPIR-V generated for all .comp files)

**Step 4: Commit**

```
feat(vulkan): add init_g_table and compute_stride_g compute shaders
```

---

### Task 12: Main Mining Shader (mine.comp)

**Files:**
- Modify: `vulkan/mine.comp` (replace stub)

**Step 1: Write mine.comp**

Replace the stub with the full mining kernel. This is the GLSL equivalent of `mine_kernel` in kernel.cu. It includes all the .glsl files and implements the full pipeline:

1. Load scalar, load/compute pubkey
2. Per-iteration loop:
   a. Get compressed pubkey
   b. Base58 encode multicodec pubkey
   c. Copy unsigned template, patch pubkey at 2 offsets
   d. SHA256 → message hash
   e. RFC 6979 nonce, ECDSA sign
   f. Base64url encode signature
   g. Copy signed template, patch pubkey + signature
   h. SHA256 → DID hash
   i. Base32 encode first 15 bytes → suffix
   j. Pattern match → if match, write to output buffer
   k. Increment scalar + pubkey
3. Save state for next dispatch

Buffer layout (all SSBOs use `std430` — no UBOs, to avoid `std140` alignment traps with scalar arrays):
```glsl
// Per-dispatch values via push constants (fast path, no buffer update needed)
layout(push_constant) uniform PushConstants {
    uint is_first_launch;
    uint iterations_per_thread;
    uint max_matches;
} pc;

// Static params (set once at startup)
layout(std430, set = 0, binding = 0) readonly buffer Params {
    uint unsigned_template_len;
    uint signed_template_len;
    uint unsigned_pubkey_offsets[2];
    uint signed_pubkey_offsets[2];
    uint signed_sig_offset;
    uint pattern_len;
    uint stride[8];       // U256
    uint stride_g[24];    // JacobianPoint (3 x U256)
} params;

layout(std430, set = 0, binding = 1) readonly buffer UnsignedTemplate { uint8_t data[]; } unsigned_tmpl;
layout(std430, set = 0, binding = 2) readonly buffer SignedTemplate { uint8_t data[]; } signed_tmpl;
layout(std430, set = 0, binding = 3) readonly buffer Pattern { uint8_t data[]; } pattern_buf;
layout(std430, set = 0, binding = 4) buffer Scalars { uint8_t data[]; } scalars;
layout(std430, set = 0, binding = 5) buffer Pubkeys { uint data[]; } pubkeys;
layout(std430, set = 0, binding = 6) buffer Matches { ... } matches;
layout(std430, set = 0, binding = 7) buffer MatchCount { uint count; } match_count;
layout(std430, set = 0, binding = 8) readonly buffer GTable { uint data[]; } g_table;
```

**Step 2: Verify compilation**

Run: `cargo build --features vulkan`
Expected: Compiles to SPIR-V

**Step 3: Commit**

```
feat(vulkan): implement main mining compute shader
```

---

## Phase 4: Host Integration

### Task 13: VulkanBackend Implementation

**Files:**
- Modify: `src/mining/vulkan.rs`

**Step 1: Write end-to-end mining test**

Port `gpu_mining_finds_valid_match` from CUDA tests:
```rust
#[test]
fn vulkan_mining_finds_valid_match() {
    let config = MiningConfig {
        pattern: b"a*".to_vec(),
        handle: "test.bsky.social".into(),
        pds: "https://bsky.social".into(),
        keep_going: false,
    };
    let stop = AtomicBool::new(false);
    let total = AtomicU64::new(0);
    let (tx, rx) = mpsc::channel();

    let backend = VulkanBackend { device_index: 0 };
    backend.run(&config, &stop, &total, tx).expect("Vulkan mining should succeed");

    let m = rx.try_recv().expect("should have found a match");
    assert!(m.did.starts_with("did:plc:a"));

    // Re-verify on CPU
    let key = SigningKey::from_bytes(
        (&data_encoding::HEXLOWER.decode(m.key_hex.as_bytes()).unwrap()[..]).into()
    ).unwrap();
    let op = build_signed_op(&key, "test.bsky.social", "https://bsky.social");
    let suffix = did_suffix(&op);
    assert_eq!(m.did, format!("did:plc:{suffix}"));
}
```

**Step 2: Implement VulkanBackend::run()**

Full implementation mirroring CudaBackend::run():

1. Create Vulkan instance (no validation layers in release, enable in debug)
2. Select physical device by `device_index`, verify compute queue support
3. Create logical device + compute queue
4. Load all 3 SPIR-V modules (init_g_table, stride_g, mine)
5. Create compute pipelines for each
6. Build CBOR template (reuse `CborTemplate::new()`)
7. Allocate and upload buffers (templates, pattern, scalars, pubkeys, matches, match_count, g_table)
8. Dispatch init_g_table (1 invocation), fence-wait
9. Dispatch stride_g (1 invocation, push constant = TOTAL_THREADS), fence-wait, read back stride values
10. Generate initial scalars (same as CUDA: base_key + tid), upload
11. Main mining loop:
    a. Reset match_count to 0
    b. Update params uniform buffer (is_first_launch, etc.)
    c. Record command buffer: bind pipeline, bind descriptor sets, dispatch mine.comp
    d. Submit + fence-wait
    e. Read back match_count; if > 0, read match buffer
    f. CPU-side re-verification (identical to CUDA)
    g. Update total counter, check stop flag
    h. Set is_first_launch = 0 after first iteration

**Step 3: Run the end-to-end test**

Run: `cargo test --features vulkan vulkan_mining_finds_valid_match`
Expected: PASS

**Step 4: Commit**

```
feat(vulkan): implement VulkanBackend with full mining pipeline
```

---

### Task 14: Auto-Detection and CLI Integration

**Files:**
- Modify: `src/main.rs`

**Step 1: Add Backend enum and auto-detection**

Replace the `backend: String` CLI field with a proper enum:

```rust
#[derive(Clone, Debug, clap::ValueEnum)]
enum Backend {
    Auto,
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "vulkan")]
    Vulkan,
}
```

Default to `Auto`.

Implement `select_backend()`:

```rust
fn select_backend(requested: Backend, threads: usize) -> Box<dyn MiningBackend> {
    match requested {
        Backend::Auto => {
            // Try CUDA first (NVIDIA-optimized path)
            #[cfg(feature = "cuda")]
            if let Ok(_) = cudarc::driver::CudaContext::new(0) {
                eprintln!("  auto-detected CUDA GPU");
                return Box::new(mining::cuda::CudaBackend { device_id: 0 });
            }

            // Try Vulkan next (any GPU)
            #[cfg(feature = "vulkan")]
            if let Some(device_index) = detect_vulkan_compute_device() {
                eprintln!("  auto-detected Vulkan GPU");
                return Box::new(mining::vulkan::VulkanBackend { device_index });
            }

            // Fallback to CPU
            Box::new(mining::cpu::CpuBackend { threads })
        }
        Backend::Cpu => Box::new(mining::cpu::CpuBackend { threads }),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Box::new(mining::cuda::CudaBackend { device_id: 0 }),
        #[cfg(feature = "vulkan")]
        Backend::Vulkan => Box::new(mining::vulkan::VulkanBackend { device_index: 0 }),
    }
}
```

`detect_vulkan_compute_device()` enumerates Vulkan physical devices and returns the index of the best one, or None. Selection logic:
1. Filter to devices that have a compute queue family AND support all required features (`shaderInt8`, `storageBuffer8BitAccess`, `shaderInt64`)
2. Score remaining devices: `DiscreteGpu` = 1000, `IntegratedGpu` = 100, `VirtualGpu` = 10, `Cpu` = 1
3. Return the highest-scored device index, or None if no devices qualify

**Step 2: Update CLI and header output**

- Change `--backend` default from `"cpu"` to `Auto`
- Update the header to show the auto-detected backend name
- Show GPU device name when using CUDA or Vulkan

**Step 3: Test auto-detection manually**

Run: `cargo run --release --features gpu -- 'test*' --placeholder`
Expected: Auto-detects the best available backend, shows it in the header

**Step 4: Commit**

```
feat: add automatic GPU backend detection (CUDA > Vulkan > CPU)
```

---

### Task 15: Update CLAUDE.md and Final Polish

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Add Vulkan backend info to the architecture docs:
- New module: `src/mining/vulkan.rs`
- New directory: `vulkan/` with GLSL shader files
- New features: `vulkan`, `gpu`
- Auto-detection behavior
- Build commands with `--features vulkan` and `--features gpu`

**Step 2: Run full test suite**

Run: `cargo test && cargo test --features vulkan && cargo test --features gpu`
Expected: All pass

**Step 3: Commit**

```
docs: update CLAUDE.md with Vulkan backend architecture
```

---

## Key Implementation Notes

### Byte Packing in GLSL

The CUDA code uses `uint8_t` arrays extensively. In GLSL, we use `GL_EXT_shader_8bit_storage` and `GL_EXT_shader_explicit_arithmetic_types_int8` extensions to get `uint8_t` type support. This is supported on Vulkan 1.2+ devices (2020+), which covers all relevant GPUs.

If a device lacks these extensions, we fall back to `uint` arrays with byte pack/unpack helpers. The auto-detection can filter out devices without these extensions.

### Vulkano Buffer Types

- `Buffer::new(allocator, BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, .. }, AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, .. }, data)` for GPU-local storage buffers
- `Buffer::from_iter(allocator, BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST, .. }, AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_HOST, .. }, data)` for host-visible readback buffers
- Use staging buffers + copy commands for uploading to device-local memory

### Remove `dead_code` Attributes

The `CborTemplate` type and `find_all` function currently have `#[cfg_attr(not(feature = "cuda"), allow(dead_code))]`. Update these to also account for the `vulkan` feature:
```rust
#[cfg_attr(not(any(feature = "cuda", feature = "vulkan")), allow(dead_code))]
```

### SSBO Alignment and Push Constants

All buffers use `std430` SSBOs (not UBOs with `std140`) to avoid alignment traps — `std140` pads scalar arrays to 16-byte stride, which would silently misalign `uint[8]` U256 limb arrays.

Per-dispatch values (`is_first_launch`, `iterations_per_thread`, `max_matches`) use **push constants** for minimal overhead — no buffer writes needed between dispatches.

Static params (template offsets, stride, stride_G) go in a read-only SSBO written once at setup.

### Synchronization Between Dispatches

Use **pipeline barriers** (not full fence-waits) between `init_g_table` → `stride_g` → `mine` dispatches within the same command buffer when possible. For the main mining loop, a fence-wait per dispatch is fine since we need to read back match results on the host anyway.

### Register Pressure and Specialization Constants

GLSL local arrays (like `unsigned_buf[512]`) may end up in registers or "private" memory. On some architectures (AMD), large local arrays reduce occupancy.

Use **specialization constants** to size local arrays to the exact template length:
```glsl
layout(constant_id = 0) const uint UNSIGNED_TMPL_LEN = 512;
layout(constant_id = 1) const uint SIGNED_TMPL_LEN = 512;
```
Then in `vulkan.rs`, set these during pipeline creation to the actual template lengths (typically ~300 bytes). The GPU compiler can then optimize register allocation based on the real sizes.

### shaderc Build Dependencies

`shaderc` requires `cmake` (and sometimes `python`/`ninja`) on the build host. The build.rs error message should be helpful if shaderc fails to compile — suggest installing `cmake` and the system's Vulkan SDK.
