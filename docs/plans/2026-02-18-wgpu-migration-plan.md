# wgpu Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace vulkano + GLSL backend with wgpu + WGSL to eliminate uint64 emulation bottleneck and achieve real GPU mining performance.

**Architecture:** Two-pass compute pipeline (ec_pass: keygen + ECDSA signing, hash_pass: SHA256 + pattern match). Vendor kangaroo project's field/curve WGSL for u32-native secp256k1 arithmetic. wgpu host code replaces vulkano with simpler pipeline/bind-group API.

**Tech Stack:** wgpu 28, pollster 0.4, WGSL shaders (included via `include_str!`), Rust 2024 edition

---

### Task 1: Update Dependencies

**Files:**
- Modify: `Cargo.toml`
- Modify: `build.rs`

**Step 1: Update Cargo.toml**

Replace vulkano/shaderc with wgpu/pollster:

```toml
[features]
cuda = ["dep:cudarc"]
wgpu = ["dep:wgpu", "dep:pollster"]
gpu = ["cuda", "wgpu"]

[dependencies]
# ... existing deps ...
wgpu = { version = "28", optional = true }
pollster = { version = "0.4", optional = true }
# REMOVE: vulkano = { version = "0.35", optional = true }

[build-dependencies]
# REMOVE: shaderc = { version = "0.10", optional = true }
```

**Step 2: Remove vulkan build step from build.rs**

Remove the entire `#[cfg(feature = "vulkan")] fn build_vulkan()` function and its call in `main()`. Keep only the CUDA build step. The `#[cfg(feature = "vulkan")]` reference in `main()` becomes `#[cfg(feature = "wgpu")]` but since WGSL is loaded via `include_str!`, there's nothing to do at build time — so just remove the vulkan block entirely.

```rust
// build.rs — only CUDA remains
#[cfg(feature = "cuda")]
fn build_cuda() {
    // ... unchanged ...
}

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}
```

**Step 3: Update feature references in mining/mod.rs**

```rust
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;
```

**Step 4: Verify it compiles**

Run: `cargo check`
Expected: compiles (wgpu module doesn't exist yet, but it's feature-gated off)

**Step 5: Commit**

```bash
git add Cargo.toml build.rs src/mining/mod.rs
git commit -m "refactor: swap vulkano/shaderc deps for wgpu/pollster"
```

---

### Task 2: Vendor kangaroo Field Arithmetic

**Files:**
- Create: `wgpu/shaders/field.wgsl`

**Step 1: Vendor field.wgsl from kangaroo**

Copy `src/gpu_crypto/shaders/field.wgsl` from [oritwoen/kangaroo](https://github.com/oritwoen/kangaroo) (MIT license). This is 817 lines of u32-native secp256k1 field arithmetic.

Key functions provided:
- `fe_add`, `fe_sub`, `fe_double` — field addition/subtraction
- `mul32(a: u32, b: u32) -> vec2<u32>` — 32×32→64 via 16-bit half-words, pure u32
- `fe_mul(a, b) -> array<u32, 8>` — field multiplication using mul32
- `fe_square(a) -> array<u32, 8>` — optimized squaring with cross-term symmetry
- `fe_inv(a) -> array<u32, 8>` — addition-chain inversion (255 squarings + 15 muls)

Add MIT license header and provenance comment at top of vendored file.

**Step 2: Commit**

```bash
git add wgpu/shaders/field.wgsl
git commit -m "vendor: add kangaroo field.wgsl (MIT, u32-native secp256k1 field arithmetic)"
```

---

### Task 3: Vendor kangaroo Curve Operations + Add Missing Functions

**Files:**
- Create: `wgpu/shaders/curve.wgsl`

**Step 1: Vendor curve.wgsl from kangaroo**

Copy `src/gpu_crypto/shaders/curve.wgsl` (220 lines). Provides:
- `JacobianPoint` struct, `AffinePoint` struct
- `jac_double`, `jac_add_affine` — Jacobian EC point operations
- `scalar_add_256`, `scalar_sub_256` — 256-bit scalar arithmetic

**Step 2: Add missing functions we need**

kangaroo doesn't provide `jac_to_affine` or `scalar_mul_G` — we need these. Add to the vendored curve.wgsl:

**`jac_to_affine`** — converts Jacobian (X,Y,Z) to affine (X/Z², Y/Z³):
```wgsl
fn jac_to_affine(p: JacobianPoint) -> AffinePoint {
    let z_inv = fe_inv(p.z);
    let z_inv2 = fe_square(z_inv);
    let z_inv3 = fe_mul(z_inv2, z_inv);
    var result: AffinePoint;
    result.x = fe_mul(p.x, z_inv2);
    result.y = fe_mul(p.y, z_inv3);
    return result;
}
```

**`scalar_mul_g_windowed`** — 4-bit windowed scalar multiplication using precomputed G table (same algorithm as our GLSL `scalar_mul_G`):
```wgsl
// g_table: array of 15 affine points [1*G, 2*G, ..., 15*G]
// Processes scalar 4 bits at a time, top to bottom
fn scalar_mul_g_windowed(
    scalar: array<u32, 8>,
    g_table_x: ptr<storage, array<u32>, read>,
    g_table_y: ptr<storage, array<u32>, read>
) -> JacobianPoint {
    var result = jac_infinity();
    // Process 64 nibbles (256 bits / 4 bits per nibble)
    for (var i = 63i; i >= 0; i--) {
        // Double 4 times
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);

        // Extract 4-bit nibble
        let limb_idx = u32(i) / 8u;
        let nibble_idx = u32(i) % 8u;
        let nibble = (scalar[limb_idx] >> (nibble_idx * 4u)) & 0xFu;

        if (nibble != 0u) {
            // Look up g_table[nibble - 1]
            let table_offset = (nibble - 1u) * 8u;
            var qx: array<u32, 8>;
            var qy: array<u32, 8>;
            for (var j = 0u; j < 8u; j++) {
                qx[j] = (*g_table_x)[table_offset + j];
                qy[j] = (*g_table_y)[table_offset + j];
            }
            result = jac_add_affine(result, qx, qy);
        }
    }
    return result;
}
```

**Step 3: Commit**

```bash
git add wgpu/shaders/curve.wgsl
git commit -m "vendor: add kangaroo curve.wgsl + scalar_mul_g_windowed, jac_to_affine"
```

---

### Task 4: Port SHA256 to WGSL

**Files:**
- Create: `wgpu/shaders/sha256.wgsl`
- Create: `wgpu/shaders/test_sha256.wgsl`

**Step 1: Write sha256.wgsl**

Port `vulkan/sha256.glsl` (156 lines) to WGSL. Key changes:
- `uint` → `u32`, `uint8_t` → byte arrays stored as `u32` (WGSL has no u8 type in compute)
- `uint64_t bit_len` → two u32 values (`bit_len_hi`, `bit_len_lo`) since inputs ≤512 bytes, `bit_len_lo = len * 8u` and `bit_len_hi = 0u`
- `const uint SHA256_K[64]` → `const SHA256_K: array<u32, 64> = array<u32, 64>(...)`
- Byte arrays become `array<u32, N>` where each u32 packs 4 bytes big-endian
- `sha256_process_block` takes `ptr<function, array<u32, 8>>` for state and reads u32 words directly
- `sha256_hash` takes byte data as packed u32 array + length, outputs 8 x u32 hash words

The WGSL version should work with u32 word arrays throughout (not byte arrays) since WGSL has no u8 type. Pack/unpack helpers convert between byte-level and u32-word-level addressing.

Functions to implement:
- `fn sha256_init() -> array<u32, 8>`
- `fn sha256_process_block(state: ptr<function, array<u32, 8>>, block: array<u32, 16>)`
- `fn sha256_hash(data: ptr<function, array<u32, 128>>, byte_len: u32) -> array<u32, 8>`

Note: Input buffer is `array<u32, 128>` = 512 bytes max, matching `SHA256_MAX_INPUT`.

**Step 2: Write test shader**

`wgpu/shaders/test_sha256.wgsl` — compute shader that reads input bytes from SSBO, runs SHA256, writes output hash to SSBO. Used by Rust test to verify GPU SHA256 matches `sha2` crate.

**Step 3: Commit**

```bash
git add wgpu/shaders/sha256.wgsl wgpu/shaders/test_sha256.wgsl
git commit -m "feat(wgpu): port SHA256 to WGSL (u32-native, no uint64)"
```

---

### Task 5: Port HMAC-DRBG to WGSL

**Files:**
- Create: `wgpu/shaders/hmac_drbg.wgsl`

**Step 1: Write hmac_drbg.wgsl**

Port `vulkan/hmac_drbg.glsl` (190 lines) to WGSL. Depends on sha256.wgsl and field.wgsl types.

Key changes from GLSL:
- All `uint8_t` arrays → `array<u32, N>` with byte packing/unpacking
- `uint64_t` in bit length calculations → `u32` (inputs always ≤128 bytes, so bit_len fits in u32)
- `hmac_sha256` works with u32-packed keys and messages
- `hmac_rfc6979_nonce` uses `array<u32, 8>` for scalar (same as field elements)

Functions to implement:
- `fn hmac_sha256(key: array<u32, 8>, msg: ptr<function, array<u32, 25>>, msg_byte_len: u32) -> array<u32, 8>` — HMAC-SHA256 with 32-byte key, up to 97-byte message
- `fn hmac_rfc6979_nonce(privkey: array<u32, 8>, hash: array<u32, 8>) -> array<u32, 8>` — RFC 6979 deterministic nonce generation

The scalar ↔ bytes helpers (`hmac_load_scalar_bytes`, `hmac_store_scalar_bytes`) become trivial since we work with u32 limbs throughout. The byte order flip (big-endian bytes ↔ little-endian limbs) is still needed.

**Step 2: Commit**

```bash
git add wgpu/shaders/hmac_drbg.wgsl
git commit -m "feat(wgpu): port HMAC-DRBG/RFC6979 to WGSL"
```

---

### Task 6: Port Encoding to WGSL

**Files:**
- Create: `wgpu/shaders/encoding.wgsl`

**Step 1: Write encoding.wgsl**

Port `vulkan/encoding.glsl` (162 lines) to WGSL. Three encoders:

- **Base32**: `fn enc_base32_15bytes(data: array<u32, 4>, out: ptr<function, array<u32, 6>>)` — 15 bytes → 24 chars. Input is 4 x u32 (only 15 bytes used). Output is 6 x u32 (4 ASCII chars packed per u32).
- **Base58**: `fn enc_base58_35bytes(data: array<u32, 9>, out: ptr<function, array<u32, 12>>)` — 35 bytes → 48 chars. Repeated division algorithm.
- **Base64url**: `fn enc_base64url_64bytes(data: array<u32, 16>, out: ptr<function, array<u32, 22>>)` — 64 bytes → 86 chars.

Alphabet constants become `const` arrays of u32. Since we're packing chars into u32s for efficient storage, byte-level indexing uses `(word >> shift) & 0xFF` patterns.

Alternative simpler approach: keep outputs as arrays of individual u32 chars (one char per u32 element) to avoid complex bit packing. This uses more registers but simplifies the logic significantly. Given that pattern matching only needs 24 chars and base58/base64url outputs go directly to CBOR patching, the simpler approach is better.

**Step 2: Commit**

```bash
git add wgpu/shaders/encoding.wgsl
git commit -m "feat(wgpu): port base32/base58/base64url encoding to WGSL"
```

---

### Task 7: Port Pattern Matching to WGSL

**Files:**
- Create: `wgpu/shaders/pattern.wgsl`

**Step 1: Write pattern.wgsl**

Port `vulkan/pattern.glsl` (38 lines) to WGSL. Nearly 1:1 translation.

```wgsl
// Match pattern against text. Pattern supports '*' wildcard.
// pattern/text are u32 arrays (one ASCII char per element).
fn pattern_glob_match(
    pattern: ptr<storage, array<u32>, read>,
    pat_len: u32,
    text: ptr<function, array<u32, 24>>,
    text_len: u32
) -> bool {
    var pi = 0u;
    var ti = 0u;
    var star = 0xFFFFFFFFu;
    var star_t = 0u;

    while (ti < text_len) {
        if (pi < pat_len && (*pattern)[pi] == (*text)[ti]) {
            pi++; ti++;
        } else if (pi < pat_len && (*pattern)[pi] == 0x2Au) { // '*'
            star = pi; star_t = ti; pi++;
        } else if (star != 0xFFFFFFFFu) {
            pi = star + 1u; star_t++; ti = star_t;
        } else {
            return false;
        }
    }
    while (pi < pat_len && (*pattern)[pi] == 0x2Au) { pi++; }
    return pi == pat_len;
}
```

**Step 2: Commit**

```bash
git add wgpu/shaders/pattern.wgsl
git commit -m "feat(wgpu): port glob pattern matching to WGSL"
```

---

### Task 8: Write Scalar Mod N Arithmetic

**Files:**
- Create: `wgpu/shaders/scalar.wgsl`

**Step 1: Write scalar.wgsl**

The kangaroo field.wgsl only has mod-p arithmetic. For ECDSA signing we need mod-n arithmetic (secp256k1 order n). This is a new module.

Functions needed:
- `fn scalar_mod_n_reduce(a: array<u32, 8>) -> array<u32, 8>` — reduce mod n if >= n
- `fn scalar_mod_n_add(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8>` — (a + b) mod n
- `fn scalar_mod_n_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8>` — (a * b) mod n using same mul32 approach as field.wgsl but reducing mod n instead of p
- `fn scalar_mod_n_inv(a: array<u32, 8>) -> array<u32, 8>` — modular inverse via Fermat's little theorem (a^(n-2) mod n)

Constants:
```wgsl
// secp256k1 order n (little-endian limbs)
const N0: u32 = 0xD0364141u;
const N1: u32 = 0xBFD25E8Cu;
const N2: u32 = 0xAF48A03Bu;
const N3: u32 = 0xBAAEDCE6u;
const N4: u32 = 0xFFFFFFFEu;
const N5: u32 = 0xFFFFFFFFu;
const N6: u32 = 0xFFFFFFFFu;
const N7: u32 = 0xFFFFFFFFu;
```

The multiplication and reduction follow the same pattern as field.wgsl's `fe_mul`, but with different modulus constants.

**Step 2: Commit**

```bash
git add wgpu/shaders/scalar.wgsl
git commit -m "feat(wgpu): add scalar mod-n arithmetic for ECDSA"
```

---

### Task 9: Write EC Pass Shader

**Files:**
- Create: `wgpu/shaders/ec_pass.wgsl`

**Step 1: Write ec_pass.wgsl**

This is Pass 1: the heavy compute pass. Each thread does:
1. Load scalar (privkey) from SSBO
2. If first launch: `scalar_mul_g_windowed(scalar)` → Jacobian point → `jac_to_affine`
3. If subsequent launch: `jac_add_affine(prev_pubkey_jac, stride_gx, stride_gy)` → affine
4. Compress pubkey: prefix byte (0x02 or 0x03 based on y parity) + x coordinate
5. HMAC-DRBG nonce: `hmac_rfc6979_nonce(privkey, message_hash)` where message_hash = SHA256 of the unsigned CBOR template
6. ECDSA sign: `(r, s)` where `r = (k*G).x mod n`, `s = k_inv * (hash + r * privkey) mod n`
7. Write to output buffer: privkey (32 bytes) + compressed pubkey (33 bytes) + signature (64 bytes)

Bindings:
```wgsl
@group(0) @binding(0) var<storage, read> scalars: array<u32>;           // per-thread private keys
@group(0) @binding(1) var<storage, read> g_table_x: array<u32>;         // 15 * 8 u32s
@group(0) @binding(2) var<storage, read> g_table_y: array<u32>;         // 15 * 8 u32s
@group(0) @binding(3) var<storage, read> stride_g: array<u32>;          // stride*G affine point (16 u32s)
@group(0) @binding(4) var<storage, read> unsigned_template: array<u32>; // CBOR template for msg hash
@group(0) @binding(5) var<storage, read_write> results: array<u32>;     // output: privkey + pubkey + sig per thread
@group(0) @binding(6) var<uniform> params: EcPassParams;

struct EcPassParams {
    is_first_launch: u32,
    num_threads: u32,
    template_byte_len: u32,
    _pad: u32,
}
```

The shader includes all dependency files via wgpu's shader composition (concatenated as strings on the Rust side via `include_str!`).

**Step 2: Commit**

```bash
git add wgpu/shaders/ec_pass.wgsl
git commit -m "feat(wgpu): write ec_pass shader (keygen + ECDSA signing)"
```

---

### Task 10: Write Hash Pass Shader

**Files:**
- Create: `wgpu/shaders/hash_pass.wgsl`

**Step 1: Write hash_pass.wgsl**

This is Pass 2: the lightweight pass. Each thread does:
1. Read compressed pubkey + signature from ec_pass output buffer
2. Base58-encode pubkey (with multicodec prefix) → 48 chars for did:key
3. Base64url-encode signature → 86 chars
4. Patch CBOR template at known byte offsets with pubkey/signature strings
5. SHA256 hash the patched signed CBOR template
6. Truncate hash to 15 bytes → base32-encode → 24 chars
7. Glob-match against pattern
8. If match: atomically increment match_count, write match result

Bindings:
```wgsl
@group(0) @binding(0) var<storage, read> ec_results: array<u32>;        // from ec_pass
@group(0) @binding(1) var<storage, read> signed_template: array<u32>;   // CBOR template with patch offsets
@group(0) @binding(2) var<storage, read> pattern_buf: array<u32>;       // pattern chars
@group(0) @binding(3) var<storage, read_write> matches: array<u32>;     // match output buffer
@group(0) @binding(4) var<storage, read_write> match_count: atomic<u32>;
@group(0) @binding(5) var<uniform> params: HashPassParams;

struct HashPassParams {
    num_threads: u32,
    pattern_len: u32,
    template_byte_len: u32,
    max_matches: u32,
    pubkey_offset_1: u32,   // byte offset in CBOR template for first did:key occurrence
    pubkey_offset_2: u32,   // byte offset for second did:key occurrence
    sig_offset: u32,        // byte offset for base64url signature
    _pad: u32,
}
```

**Step 2: Commit**

```bash
git add wgpu/shaders/hash_pass.wgsl
git commit -m "feat(wgpu): write hash_pass shader (template hash + pattern match)"
```

---

### Task 11: Write wgpu Host Backend (Skeleton)

**Files:**
- Create: `src/mining/wgpu_backend.rs`

**Step 1: Write the backend struct and MiningBackend impl skeleton**

```rust
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;

use crate::mining::{Match, MiningBackend, MiningConfig};
use crate::plc::CborTemplate;

pub struct WgpuBackend {
    pub device_index: usize,
}

impl MiningBackend for WgpuBackend {
    fn name(&self) -> &str { "wgpu" }

    fn run(
        &self,
        config: &MiningConfig,
        stop: &AtomicBool,
        total: &AtomicU64,
        tx: mpsc::Sender<Match>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Init wgpu
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })).ok_or("no GPU adapter found")?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))?;

        // 2. Compose shaders (concatenate WGSL modules)
        let field_src = include_str!("../../wgpu/shaders/field.wgsl");
        let curve_src = include_str!("../../wgpu/shaders/curve.wgsl");
        let sha256_src = include_str!("../../wgpu/shaders/sha256.wgsl");
        let hmac_src = include_str!("../../wgpu/shaders/hmac_drbg.wgsl");
        let scalar_src = include_str!("../../wgpu/shaders/scalar.wgsl");
        let encoding_src = include_str!("../../wgpu/shaders/encoding.wgsl");
        let pattern_src = include_str!("../../wgpu/shaders/pattern.wgsl");
        let ec_pass_src = include_str!("../../wgpu/shaders/ec_pass.wgsl");
        let hash_pass_src = include_str!("../../wgpu/shaders/hash_pass.wgsl");

        let ec_shader_src = format!("{field_src}\n{curve_src}\n{sha256_src}\n{hmac_src}\n{scalar_src}\n{ec_pass_src}");
        let hash_shader_src = format!("{field_src}\n{sha256_src}\n{encoding_src}\n{pattern_src}\n{hash_pass_src}");

        // 3. Create shader modules + compute pipelines
        // 4. Allocate GPU buffers
        // 5. Init g_table (CPU-side, upload precomputed [1*G..15*G])
        // 6. Build CBOR templates
        // 7. Main mining loop:
        //    a. Generate/advance scalars
        //    b. Upload scalars
        //    c. Dispatch ec_pass
        //    d. Dispatch hash_pass
        //    e. Read back match_count
        //    f. If matches: read match buffer, CPU-verify, send via tx
        //    g. Update total counter
        //    h. Check stop flag

        todo!("implement mining loop")
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo check --features wgpu`
Expected: compiles (with todo!() warning)

**Step 3: Commit**

```bash
git add src/mining/wgpu_backend.rs
git commit -m "feat(wgpu): add wgpu backend skeleton with MiningBackend impl"
```

---

### Task 12: Implement wgpu Host Backend (Full)

**Files:**
- Modify: `src/mining/wgpu_backend.rs`

**Step 1: Implement GPU initialization**

- Create wgpu instance, adapter, device, queue
- Compose shader source strings (concatenate modules)
- Create two compute pipelines: ec_pass and hash_pass
- Create bind group layouts matching shader bindings

**Step 2: Implement buffer allocation**

- Scalars buffer: `num_threads * 8 * 4` bytes (8 u32 limbs per scalar)
- G table buffers: `15 * 8 * 4` bytes each for x and y
- Stride G buffer: `16 * 4` bytes
- Results buffer: `num_threads * 33 * 4` bytes (privkey 8 + pubkey 9 + sig 16 u32s per thread)
- Template buffers: sized to CBOR template length
- Pattern buffer: up to 24 u32s
- Match buffer: `max_matches * (8 + 6) * 4` bytes (privkey + DID suffix per match)
- Match count buffer: 4 bytes (atomic u32)
- Uniform buffers for params structs

**Step 3: Implement G table initialization**

Compute [1*G, 2*G, ..., 15*G] on CPU using `k256` crate, upload to GPU buffers as u32 limbs (little-endian).

**Step 4: Implement main mining loop**

```
loop {
    if stop.load(Relaxed) { break; }

    // Upload scalars
    queue.write_buffer(&scalars_buf, 0, &scalar_data);
    queue.write_buffer(&params_buf, 0, &ec_params_bytes);

    // Dispatch ec_pass
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ec_pipeline);
        pass.set_bind_group(0, &ec_bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // Dispatch hash_pass
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&hash_pipeline);
        pass.set_bind_group(0, &hash_bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // Copy match_count to staging buffer for readback
    encoder.copy_buffer_to_buffer(&match_count_buf, 0, &staging_buf, 0, 4);
    queue.submit(Some(encoder.finish()));

    // Read back match count
    // ... map staging buffer, check count, read matches if any ...

    // Advance scalars for next iteration
    is_first_launch = false;
    total.fetch_add(num_threads as u64, Ordering::Relaxed);
}
```

**Step 5: Implement match readback and CPU verification**

When GPU reports matches:
1. Map match buffer to CPU
2. For each match: extract privkey, reconstruct DID on CPU using existing `plc::did_suffix()`
3. Verify GPU result matches CPU result
4. Send verified matches via `tx`

**Step 6: Run test**

Run: `cargo test --features wgpu test_wgpu_backend_basic`
Expected: PASS (basic init + single dispatch)

**Step 7: Commit**

```bash
git add src/mining/wgpu_backend.rs
git commit -m "feat(wgpu): implement full wgpu mining backend"
```

---

### Task 13: Update CLI and Auto-Detection

**Files:**
- Modify: `src/main.rs`

**Step 1: Update select_backend for wgpu**

Replace the vulkan probe with wgpu adapter probe:

```rust
fn select_backend(threads: usize) -> Box<dyn MiningBackend> {
    #[cfg(feature = "cuda")]
    {
        if cudarc::driver::CudaDevice::new(0).is_ok() {
            return Box::new(mining::cuda::CudaBackend { device_id: 0 });
        }
    }

    #[cfg(feature = "wgpu")]
    {
        let instance = wgpu::Instance::default();
        if pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })).is_some() {
            return Box::new(mining::wgpu_backend::WgpuBackend { device_index: 0 });
        }
    }

    Box::new(mining::cpu::CpuBackend { threads })
}
```

**Step 2: Update CLI backend matching**

```rust
let backend: Box<dyn MiningBackend> = match cli.backend.as_str() {
    "auto" => select_backend(threads),
    "cpu" => Box::new(mining::cpu::CpuBackend { threads }),
    #[cfg(feature = "cuda")]
    "cuda" => Box::new(mining::cuda::CudaBackend { device_id: 0 }),
    #[cfg(feature = "wgpu")]
    "wgpu" => Box::new(mining::wgpu_backend::WgpuBackend { device_index: 0 }),
    other => {
        let mut available = vec!["auto", "cpu"];
        if cfg!(feature = "cuda") { available.push("cuda"); }
        if cfg!(feature = "wgpu") { available.push("wgpu"); }
        eprintln!("error: unknown backend '{other}'. available: {}", available.join(", "));
        std::process::exit(1);
    }
};
```

Update the CLI doc comment: `/// Mining backend to use (auto, cpu, cuda, wgpu)`.

**Step 3: Verify build**

Run: `cargo build --features wgpu`
Expected: compiles

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat(wgpu): update CLI and auto-detection for wgpu backend"
```

---

### Task 14: Add GPU-vs-CPU Verification Tests

**Files:**
- Modify: `src/mining/wgpu_backend.rs` (add tests at bottom)

**Step 1: Write test helper — WgpuTestContext**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct WgpuTestContext {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl WgpuTestContext {
        fn new() -> Self {
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
                .expect("no GPU adapter");
            let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None))
                .expect("failed to create device");
            Self { device, queue }
        }

        fn run_compute(&self, shader_src: &str, buffers: &[&[u8]]) -> Vec<Vec<u8>> {
            // Generic helper: create shader module, pipeline, bind group,
            // dispatch 1 workgroup, read back output buffers
            todo!()
        }
    }
}
```

**Step 2: Write field_mul test**

Test: upload two known field elements, dispatch `test_field_mul.wgsl`, verify GPU result matches CPU `num-bigint` computation.

**Step 3: Write SHA256 test**

Test: upload known message, dispatch `test_sha256.wgsl`, verify GPU hash matches `sha2` crate.

**Step 4: Write ECDSA test**

Test: upload known private key + message hash, dispatch ec_pass with 1 thread, verify GPU signature matches `k256` ECDSA signature.

**Step 5: Write end-to-end test**

Test: dispatch full pipeline (ec_pass + hash_pass) with pattern `"*"` and 1 thread, verify GPU produces a valid DID suffix matching CPU computation.

**Step 6: Run all tests**

Run: `cargo test --features wgpu`
Expected: all PASS

**Step 7: Commit**

```bash
git add src/mining/wgpu_backend.rs wgpu/shaders/test_*.wgsl
git commit -m "test(wgpu): add GPU-vs-CPU verification tests"
```

---

### Task 15: Remove Old Vulkan Backend

**Files:**
- Remove: `vulkan/` directory (all GLSL shaders)
- Remove: `src/mining/vulkan.rs`
- Modify: `src/mining/mod.rs` (remove vulkan module)

**Step 1: Remove vulkan files**

Move `vulkan/` directory and `src/mining/vulkan.rs` to trash (per safety rules — `mv` to `~/.local/trash/`).

**Step 2: Clean up mod.rs**

Remove `#[cfg(feature = "vulkan")] pub mod vulkan;` if it still exists.

**Step 3: Verify build**

Run: `cargo test --features wgpu`
Expected: all PASS, no vulkan references remain

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove old vulkan/GLSL backend (replaced by wgpu/WGSL)"
```

---

### Task 16: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update documentation**

- Replace all `vulkan` references with `wgpu`
- Update build commands: `--features vulkan` → `--features wgpu`
- Update module layout section with `wgpu_backend.rs` and `wgpu/shaders/` structure
- Update feature gating section
- Update key invariants (no more shaderc, SPIR-V, etc.)
- Add note about vendored kangaroo shaders

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for wgpu migration"
```

---

## Execution Notes

**Parallelization opportunities:**
- Tasks 2-3 (vendor field + curve) are independent
- Tasks 4-8 (sha256, hmac, encoding, pattern, scalar) are independent of each other
- Tasks 9-10 (ec_pass, hash_pass) depend on 4-8 but are independent of each other
- Tasks 11-12 (host backend) depend on 9-10
- Task 13 (CLI) depends on 11
- Task 14 (tests) depends on 12
- Tasks 15-16 (cleanup) depend on everything passing

**Risk areas:**
- WGSL shader composition: wgpu doesn't have `#include` — we concatenate shader source strings. Function name collisions between modules must be avoided.
- WGSL limitations: no u8/u64 types, no recursion, no dynamic array indexing in some cases. All byte-level operations must use u32 packing.
- TDR avoidance: keep ec_pass dispatch small initially (256 threads, 1 workgroup). Tune up after correctness is proven.
- kangaroo's field arithmetic assumes no full reduction after add/sub (lazy reduction). This is correct for secp256k1 but must be validated in our ECDSA context.
