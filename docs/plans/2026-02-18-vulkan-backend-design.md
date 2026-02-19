# Vulkan Backend + Auto-Detection Design

## Overview

Add a Vulkan compute backend to plcpick so it can mine on non-NVIDIA GPUs (AMD, Intel, etc.). Also add automatic backend detection so users don't need to manually specify `--backend`. The Vulkan backend implements the same mining pipeline as the existing CUDA backend, ported to GLSL compute shaders compiled to SPIR-V.

## Architecture

### New Module Structure

```
src/mining/
  vulkan.rs          Vulkan backend (feature = "vulkan")
vulkan/
  secp256k1.glsl     U256 arithmetic, field/scalar ops, EC point operations
  sha256.glsl        SHA256 implementation
  hmac_drbg.glsl     HMAC-SHA256 + RFC 6979 nonce generation
  encoding.glsl      Base32, base58, base64url encoders
  pattern.glsl       Glob pattern matching
  mine.comp          Main mining kernel (includes all above)
  init_g_table.comp  Precompute [1*G..15*G] table
  stride_g.comp      Compute stride * G
```

### Feature Flags

```toml
[features]
cuda = ["dep:cudarc"]
vulkan = ["dep:vulkano", "dep:shaderc"]
gpu = ["cuda", "vulkan"]  # convenience: all GPU backends
```

All three features are independent. `gpu` activates both GPU backends. Tests are gated behind `#[cfg(feature = "vulkan")]`.

### Backend Auto-Detection

The `--backend` flag changes from defaulting to `"cpu"` to defaulting to `"auto"`:

```
--backend auto|cpu|cuda|vulkan
```

Auto-detection priority (in `select_backend()`):
1. **CUDA** (if `cuda` feature enabled): Try `CudaContext::new(0)`. If succeeds → use CUDA.
2. **Vulkan** (if `vulkan` feature enabled): Enumerate Vulkan physical devices, filter to those with compute queues + required features (`shaderInt8`, `storageBuffer8BitAccess`, `shaderInt64`), score by type (`DiscreteGpu` 1000 > `IntegratedGpu` 100 > `VirtualGpu` 10). Pick highest-scored device.
3. **CPU**: Fallback. Always available.

Rationale for CUDA > Vulkan priority: The CUDA kernel is the more mature/optimized path (original implementation). On NVIDIA hardware where both are available, CUDA is the better choice.

When a specific backend is requested (e.g., `--backend vulkan`), skip auto-detection and use that backend directly, failing with a clear error if the feature isn't compiled in or no suitable device is found.

## Vulkan Compute Shader Design

### GLSL ↔ CUDA Mapping

The GLSL shaders are a direct port of the CUDA headers. Key translations:

| CUDA Concept | GLSL/Vulkan Equivalent |
|---|---|
| `__global__ void kernel(...)` | `void main()` with `layout(local_size_x=N)` |
| `threadIdx.x + blockIdx.x * blockDim.x` | `gl_GlobalInvocationID.x` |
| `__device__` functions | Regular GLSL functions |
| `__constant__` memory | Uniform buffers or push constants |
| `atomicAdd(ptr, 1)` | `atomicAdd(ssbo.field, 1)` |
| `uint8_t` arrays | `uint` arrays (pack 4 bytes per uint) |
| Raw pointer args | Storage buffer bindings (SSBOs) |

### Critical GLSL Differences

**uint8_t via extensions.** We require `GL_EXT_shader_explicit_arithmetic_types_int8` + `GL_EXT_shader_8bit_storage` for `uint8_t` support in local variables and SSBOs. This requires the `shaderInt8` and `storageBuffer8BitAccess` Vulkan device features (Vulkan 1.2 core, covers all 2018+ GPUs).

**uint64_t for intermediate products.** The `field_mul` and `scalar_mul` functions use `(uint64_t)a * b` for 32x32→64 bit multiplication. We require `GL_EXT_shader_explicit_arithmetic_types_int64` and the `shaderInt64` Vulkan device feature.

**No function pointers or virtual dispatch.** All code is statically resolved — same as CUDA.

**No `#include` in vanilla GLSL.** We use shaderc's `#include` support with a custom include resolver. The build system resolves includes from the `vulkan/` directory.

### Buffer Layout

All data buffers use SSBOs with `std430` layout to avoid `std140` alignment traps (which pad scalar arrays to 16-byte stride). Per-dispatch values use push constants.

```
Push Constants:
  - is_first_launch: uint
  - iterations_per_thread: uint
  - max_matches: uint

Set 0, Binding 0: KernelParams (SSBO, readonly, std430) — written once at setup
  - unsigned_template_len: uint
  - signed_template_len: uint
  - unsigned_pubkey_offsets: uint[2]
  - signed_pubkey_offsets: uint[2]
  - signed_sig_offset: uint
  - pattern_len: uint
  - stride: uint[8]        (U256)
  - stride_g: uint[24]     (JacobianPoint = 3 x U256)

Set 0, Binding 1: unsigned_template (SSBO, readonly, uint8_t[])
Set 0, Binding 2: signed_template (SSBO, readonly, uint8_t[])
Set 0, Binding 3: pattern (SSBO, readonly, uint8_t[])
Set 0, Binding 4: scalars (SSBO, read/write, uint8_t[])
Set 0, Binding 5: pubkeys (SSBO, read/write, uint[])
Set 0, Binding 6: matches (SSBO, read/write)
  - Per slot: privkey uint8_t[32], signature uint8_t[64], suffix uint8_t[24], found uint
Set 0, Binding 7: match_count (SSBO, read/write, uint — atomic)
Set 0, Binding 8: g_table (SSBO, readonly, uint[])
  - 15 affine points, each 2 x U256 = 16 uints → 240 uints total
```

### Workgroup Sizing

CUDA uses 256 threads/block × 128 blocks = 32,768 total threads. For Vulkan:
- `local_size_x = 256` (workgroup size, analogous to CUDA block size)
- Dispatch 128 workgroups (analogous to CUDA grid)
- Total invocations: 32,768 (same as CUDA)

These can be tuned per-device. Vulkan exposes `maxComputeWorkGroupSize` and `maxComputeWorkGroupInvocations` limits.

## Rust-Side Integration (vulkan.rs)

### VulkanBackend struct

```rust
pub struct VulkanBackend {
    pub device_index: usize,
}
```

### run() Flow

Mirrors `CudaBackend::run()` exactly:

1. **Init Vulkan**: Create instance → select physical device → create logical device + compute queue
2. **Load shaders**: Load pre-compiled SPIR-V modules, create compute pipelines for all 3 shaders
3. **Allocate buffers**: Create Vulkan buffers for all SSBOs, upload templates + pattern
4. **Init G table**: Dispatch `init_g_table.comp`, fence-wait
5. **Compute stride_G**: Dispatch `stride_g.comp`, fence-wait, read back stride values
6. **Generate scalars**: Same logic as CUDA — `base_key + thread_id` for each thread, upload to device
7. **Main loop**:
   a. Reset match_count to 0
   b. Update KernelParams uniform buffer
   c. Record command buffer: bind pipeline + descriptors, dispatch mine.comp
   d. Submit + fence-wait
   e. Read back match_count, if > 0 read match buffer
   f. CPU-side re-verification of matches (same as CUDA)
   g. Update total counter, check stop flag

### Memory Strategy

- **Device-local SSBOs** for GPU-only data: scalars, pubkeys, templates, g_table, params
- **Host-visible SSBOs** for readback: match output, match count
- **Push constants** for per-dispatch values (is_first_launch, iterations_per_thread, max_matches) — no buffer writes needed between dispatches
- Use staging buffers + copy commands for uploading initial data to device-local memory

### Required Vulkan Device Features

The Vulkan backend requires these device features (checked during auto-detection):
- `shaderInt8` — 8-bit integer arithmetic in shaders
- `storageBuffer8BitAccess` — uint8_t in SSBOs
- `shaderInt64` — 64-bit integer arithmetic for field/scalar multiplication intermediates

## Build System Changes

### build.rs additions

```rust
#[cfg(feature = "vulkan")]
fn build_vulkan() {
    // Use shaderc to compile GLSL → SPIR-V
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_include_callback(/* resolve vulkan/ directory includes */);

    // Compile each .comp file
    for shader in ["mine.comp", "init_g_table.comp", "stride_g.comp"] {
        let source = std::fs::read_to_string(format!("vulkan/{shader}")).unwrap();
        let spirv = compiler.compile_into_spirv(&source, ShaderKind::Compute, shader, "main", Some(&options)).unwrap();
        std::fs::write(format!("{out_dir}/{shader}.spv"), spirv.as_binary_u8()).unwrap();
    }

    println!("cargo:rerun-if-changed=vulkan/");
}
```

SPIR-V binaries are embedded in the Rust binary via `include_bytes!`.

## Testing Strategy

Port the same verification tests from CUDA:

- `gpu_sha256_matches_cpu` — SHA256 correctness
- `gpu_scalar_mul_g_matches_k256` — EC scalar multiplication
- `gpu_ecdsa_sign_matches_k256` — Full ECDSA signing
- `gpu_base58_matches_cpu` — Base58 encoding
- `gpu_base64url_matches_cpu` — Base64url encoding
- `gpu_base32_matches_cpu` — Base32 encoding
- `gpu_mining_finds_valid_match` — End-to-end mining correctness

Each test dispatches a small compute shader, reads back results, and compares against CPU reference. Tests are gated behind `#[cfg(feature = "vulkan")]`.

Additionally, test auto-detection logic with unit tests (mock feature flags and device availability).

## Risk & Mitigation

| Risk | Mitigation |
|---|---|
| GLSL uint8/uint64 extensions not available | Require shaderInt8, storageBuffer8BitAccess, shaderInt64 — filter devices during auto-detection, clear error on explicit `--backend vulkan`. Covers all 2018+ discrete GPUs. |
| std140 alignment silently breaks uint[8] arrays | Use std430 SSBOs exclusively, no UBOs. |
| Register pressure from large local arrays (512-byte buffers) | Size local arrays to exact template length. Monitor occupancy. |
| 64-bit intermediates slower on some GPUs | Required for correctness in field_mul/scalar_mul. No alternative without manual 32x32→64 splitting (much worse). |
| Vulkan setup boilerplate | vulkano handles most of it. Compute-only pipeline is simpler than graphics. |
| GPU auto-detection picks wrong device (iGPU vs dGPU) | Score by device type: Discrete > Integrated > Virtual. Check required features. |
| shaderc compilation adds build-time dependency | shaderc-rs bundles the compiler. Build time increases but no system deps needed. |
| Performance may differ from CUDA on NVIDIA hardware | Expected — CUDA path is preferred for NVIDIA via auto-detection. |
