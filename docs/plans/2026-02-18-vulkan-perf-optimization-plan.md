# Vulkan Performance Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize the Vulkan mining backend from ~1 key/sec to CUDA-competitive speeds (~10K-100K keys/sec).

**Architecture:** Apply 6 incremental optimizations to the GLSL compute shaders and Rust host code: shared memory g_table, streaming SHA256, multi-workgroup dispatch, addition-chain scalar inverse, dedicated field squaring, and increased iterations per launch. Each phase is independently testable.

**Tech Stack:** GLSL 460 compute shaders (via shaderc), vulkano 0.35, Rust 2024 edition

---

### Task 1: Shared Memory g_table

The biggest register pressure source is each thread loading 15 affine points (960 bytes) from SSBO into local arrays. Use workgroup shared memory instead.

**Files:**
- Modify: `vulkan/mine.comp:195-224` (main function, g_table loading)
- Modify: `vulkan/secp256k1.glsl:644` (secp_scalar_mul_G signature — change from `in AffinePoint g_table[15]` to reading from shared)

**Step 1: Add shared memory g_table declaration to mine.comp**

In `mine.comp`, before `void main()`, add:

```glsl
shared AffinePoint shared_g_table[15];
```

**Step 2: Replace per-thread g_table load with cooperative shared load**

Replace the current g_table loading in `main()` (lines 198-200):

```glsl
// OLD:
AffinePoint g_table[15];
load_g_table(g_table);
```

With cooperative loading:

```glsl
// Load g_table into shared memory cooperatively
uint local_id = gl_LocalInvocationID.x;
if (local_id < 15u) {
    for (int j = 0; j < 8; j++) {
        shared_g_table[local_id].x.d[j] = g_table_data[local_id * 16 + j];
        shared_g_table[local_id].y.d[j] = g_table_data[local_id * 16 + 8 + j];
    }
}
barrier();
```

**Step 3: Update secp_scalar_mul_G to accept shared memory array**

GLSL doesn't allow passing `shared` arrays as function parameters directly. Two options:
- (A) Make `shared_g_table` a file-scope shared variable and access it directly inside `secp_scalar_mul_G` — but secp256k1.glsl doesn't know about it.
- (B) Keep the function parameter and pass shared_g_table — GLSL actually allows this since shared arrays can be passed by reference.

Go with option (A): move `secp_scalar_mul_G` to read from a global `shared` array. Since secp256k1.glsl is `#include`d into mine.comp, declare the shared variable in mine.comp before the include and use a `#define` guard.

Actually, simplest approach: In mine.comp, declare `shared AffinePoint shared_g_table[15]` **before** including secp256k1.glsl. Then modify `secp_scalar_mul_G` to use `shared_g_table` directly when `USE_SHARED_G_TABLE` is defined:

In mine.comp, before includes:
```glsl
#define USE_SHARED_G_TABLE
shared AffinePoint shared_g_table[15];
```

In secp256k1.glsl, change `secp_scalar_mul_G`:
```glsl
#ifdef USE_SHARED_G_TABLE
void secp_scalar_mul_G(inout JacobianPoint r, in U256 k) {
    // Uses shared_g_table directly
#else
void secp_scalar_mul_G(inout JacobianPoint r, in U256 k, in AffinePoint g_table[15]) {
#endif
    for (int i = 0; i < 8; i++) { r.x.d[i] = 0u; r.y.d[i] = 0u; r.z.d[i] = 0u; }

    for (int i = 63; i >= 0; i--) {
        if (i < 63) {
            secp_point_double(r, r);
            secp_point_double(r, r);
            secp_point_double(r, r);
            secp_point_double(r, r);
        }

        int limb_idx = i / 8;
        int bit_offset = (i % 8) * 4;
        uint window = (k.d[limb_idx] >> bit_offset) & 0xFu;

        if (window != 0u) {
#ifdef USE_SHARED_G_TABLE
            AffinePoint pt = shared_g_table[window - 1u];
#else
            AffinePoint pt = g_table[window - 1u];
#endif
            if (secp_point_is_infinity(r)) {
                r.x = pt.x;
                r.y = pt.y;
                r.z = secp_u256_from_u32(1u);
            } else {
                secp_point_add_affine(r, r, pt.x, pt.y);
            }
        }
    }
}
```

**Step 4: Update all callers in mine.comp**

Change `secp_scalar_mul_G(pubkey, scalar, g_table)` to `secp_scalar_mul_G(pubkey, scalar)` (2 occurrences: the initial pubkey computation and the nonce R computation).

Also remove the now-unused `load_g_table()` call and local `AffinePoint g_table[15]` declaration.

**Step 5: Ensure other shaders still compile**

The test shaders (`test_scalar_mul_G.comp`) and utility shaders (`init_g_table.comp`, `stride_g.comp`) don't define `USE_SHARED_G_TABLE`, so they keep the old 3-parameter signature. This means secp256k1.glsl needs BOTH versions.

**Step 6: Build and verify**

Run: `cargo build --release --features vulkan`
Run: `cargo test --features vulkan -- --test-threads=1 vulkan_mine_minimal_diagnostic`
Expected: Compiles, test passes

**Step 7: Commit**

```bash
git add vulkan/mine.comp vulkan/secp256k1.glsl
git commit -m "perf(vulkan): use shared memory for g_table"
```

---

### Task 2: Stream SHA256 from SSBO

Each iteration copies ~306 + ~398 bytes from SSBOs into local buffers just to patch a few bytes and hash them. Instead, modify SHA256 to read blocks directly from the SSBO, with overlay support for patched regions.

**Files:**
- Modify: `vulkan/sha256.glsl` — add `sha256_hash_patched_unsigned` and `sha256_hash_patched_signed` functions
- Modify: `vulkan/mine.comp:242-261, 323-349` — replace local buffer copies with streaming hash calls

**Step 1: Add SSBO-streaming SHA256 variants to sha256.glsl**

The key insight: SHA256 processes data in 64-byte blocks. We can read each block directly from the unsigned/signed template SSBO, overlaying the patched pubkey/signature bytes at the right offsets. This avoids the 512-byte local buffer.

Add to sha256.glsl:

```glsl
// SHA256 block processing that reads from a uint8_t SSBO with byte-level patches.
// Patches are applied inline during the block read.
// This avoids copying the entire template into local memory.
void sha256_process_block_from_ssbo(
    inout uint h[8],
    in uint8_t ssbo_data[],  // can't actually index into SSBO like this in GLSL
    uint block_offset
) {
    // ... read 64 bytes from SSBO starting at block_offset
}
```

Actually, GLSL SSBO arrays can't be passed as function parameters generically. The better approach is to process the template in 64-byte blocks inline in mine.comp, reading from the template SSBO and overlaying patches per-block.

**Revised approach**: Write helper functions directly in mine.comp that process unsigned/signed templates block-by-block, reading from their specific SSBOs and applying patches. This avoids trying to make sha256.glsl SSBO-generic.

In mine.comp, add:

```glsl
// Hash the unsigned template from SSBO with pubkey patches applied
void hash_unsigned_template(
    inout uint8_t hash_out[32],
    in uint8_t base58_pubkey[48]
) {
    uint h[8];
    sha256_init(h);
    uint len = unsigned_template_len;
    uint8_t block[128];

    uint i = 0u;
    while (i + 64u <= len) {
        for (uint j = 0u; j < 64u; j++) {
            uint pos = i + j;
            uint8_t b = unsigned_template[pos];
            // Check if pos falls in a pubkey patch region
            for (int loc = 0; loc < 2; loc++) {
                uint off = unsigned_pubkey_offsets[loc];
                if (pos >= off && pos < off + 48u) {
                    b = base58_pubkey[pos - off];
                }
            }
            block[j] = b;
        }
        sha256_process_block(h, block, 0u);
        i += 64u;
    }

    // Handle padding (remaining bytes + 0x80 + length)
    uint rem = len - i;
    for (uint j = 0u; j < 128u; j++) block[j] = uint8_t(0u);
    for (uint j = 0u; j < rem; j++) {
        uint pos = i + j;
        uint8_t b = unsigned_template[pos];
        for (int loc = 0; loc < 2; loc++) {
            uint off = unsigned_pubkey_offsets[loc];
            if (pos >= off && pos < off + 48u) {
                b = base58_pubkey[pos - off];
            }
        }
        block[j] = b;
    }
    block[rem] = uint8_t(0x80u);
    uint pad_len = (rem < 56u) ? 64u : 128u;
    uint64_t bit_len = uint64_t(len) * 8ul;
    block[pad_len - 8u] = uint8_t(uint(bit_len >> 56) & 0xFFu);
    block[pad_len - 7u] = uint8_t(uint(bit_len >> 48) & 0xFFu);
    block[pad_len - 6u] = uint8_t(uint(bit_len >> 40) & 0xFFu);
    block[pad_len - 5u] = uint8_t(uint(bit_len >> 32) & 0xFFu);
    block[pad_len - 4u] = uint8_t(uint((bit_len >> 24)) & 0xFFu);
    block[pad_len - 3u] = uint8_t(uint((bit_len >> 16)) & 0xFFu);
    block[pad_len - 2u] = uint8_t(uint((bit_len >> 8)) & 0xFFu);
    block[pad_len - 1u] = uint8_t(uint(bit_len) & 0xFFu);
    for (uint j = 0u; j < pad_len; j += 64u) {
        sha256_process_block(h, block, j);
    }

    for (int j = 0; j < 8; j++) {
        hash_out[j * 4]     = uint8_t((h[j] >> 24) & 0xFFu);
        hash_out[j * 4 + 1] = uint8_t((h[j] >> 16) & 0xFFu);
        hash_out[j * 4 + 2] = uint8_t((h[j] >> 8) & 0xFFu);
        hash_out[j * 4 + 3] = uint8_t(h[j] & 0xFFu);
    }
}
```

Similarly for `hash_signed_template` with 3 patch regions (2 pubkey + 1 signature).

**Step 2: Replace local buffer code in mine.comp main loop**

Remove the `uint8_t unsigned_buf[SHA256_MAX_INPUT]` and `uint8_t signed_buf[SHA256_MAX_INPUT]` declarations and their copy/patch loops. Replace with:

```glsl
// 5. Hash unsigned template with pubkey patches
uint8_t msg_hash[32];
hash_unsigned_template(msg_hash, base58_pubkey);

// ... (ECDSA signing steps stay the same) ...

// 14. Hash signed template with pubkey + signature patches
uint8_t did_hash[32];
hash_signed_template(did_hash, base58_pubkey, base64_sig);
```

**Step 3: Build and verify**

Run: `cargo build --release --features vulkan`
Run: `cargo test --features vulkan -- --test-threads=1 vulkan_mine_minimal_diagnostic`
Expected: Compiles, test passes (GPU suffix still matches CPU)

**Step 4: Commit**

```bash
git add vulkan/mine.comp
git commit -m "perf(vulkan): stream SHA256 from SSBO instead of local buffer copy"
```

---

### Task 3: Scale Up Workgroups

With reduced per-thread memory pressure from Tasks 1-2, we can safely dispatch more workgroups.

**Files:**
- Modify: `src/mining/vulkan.rs:26-29` — change constants

**Step 1: Make workgroup count dynamic based on device properties**

In `vulkan.rs`, replace the hardcoded constants:

```rust
const WORKGROUP_SIZE: u32 = 256;
const NUM_WORKGROUPS: u32 = 1;
const TOTAL_THREADS: u32 = WORKGROUP_SIZE * NUM_WORKGROUPS;
const ITERATIONS_PER_LAUNCH: u32 = 1;
```

With device-adaptive values. After creating the device, query `physical_device.properties().max_compute_work_group_count` and compute unit count. For now, start conservative:

```rust
const WORKGROUP_SIZE: u32 = 256;
// These become parameters computed at runtime:
// num_workgroups = 4 (conservative start)
// iterations_per_launch = 4
```

**Step 2: Parameterize the dispatch**

Change the `run()` function to compute `num_workgroups` and `iterations_per_launch` based on the device, rather than using constants. Start with:

```rust
let num_workgroups: u32 = 4;
let iterations_per_launch: u32 = 4;
let total_threads = WORKGROUP_SIZE * num_workgroups;
```

Update all references to `TOTAL_THREADS`, `NUM_WORKGROUPS`, `ITERATIONS_PER_LAUNCH` to use the local variables.

**Step 3: Update scalar buffer and pubkey buffer allocation**

These are sized by `total_threads`, so they need to use the runtime value:

```rust
let mut scalar_data = vec![0u8; total_threads as usize * 32];
let pubkeys_buf = make_buffer_u32(memory_allocator.clone(), &vec![0u32; total_threads as usize * 24]);
```

**Step 4: Update stride computation**

The stride value must be `total_threads`:

```rust
stride_in_data[0] = total_threads;
```

And update `KernelParams` stride field accordingly.

**Step 5: Build, test, and benchmark**

Run: `cargo build --release --features vulkan`
Run: `cargo test --features vulkan -- --test-threads=1 vulkan_mine_minimal_diagnostic`
Run: `cargo run --release --features vulkan -- 'a*' --placeholder --backend vulkan`
Record keys/sec. If dispatch doesn't complete in reasonable time, reduce `num_workgroups` back.

**Step 6: Commit**

```bash
git add src/mining/vulkan.rs
git commit -m "perf(vulkan): scale to multiple workgroups for SM utilization"
```

---

### Task 4: Addition-Chain scalar_inv

The generic binary scalar_inv does ~256 scalar_mul operations. An addition chain can reduce this to ~40.

**Files:**
- Modify: `vulkan/secp256k1.glsl:380-405` — replace `secp_scalar_inv` implementation

**Step 1: Implement addition chain for secp256k1 n-2**

Replace the body of `secp_scalar_inv` with an addition chain similar to the one used for `secp_field_inv` (which already uses an optimized chain for p-2):

```glsl
void secp_scalar_inv(inout U256 r, in U256 a) {
    // Compute a^(n-2) mod n using addition chain
    // n-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD036413F
    //
    // Strategy: build up power-of-2-minus-1 chains, then combine for the tail
    U256 x2, x3, x6, x7, x8, x14, x28, x56, x112, x126, t;

    // x2 = a^3 = a^(2^2-1)
    secp_scalar_mul(t, a, a);      // a^2
    secp_scalar_mul(x2, t, a);     // a^3

    // x3 = a^(2^3-1) = a^7
    secp_scalar_mul(t, x2, x2);    // a^6
    // Actually: t = x2^2 = (a^3)^2 = a^6... no, scalar_mul is multiplication not squaring
    // We need scalar_sqr or just use scalar_mul(t, val, val)
    // Since we don't have scalar_sqr, use scalar_mul(t, t_prev, t_prev)

    // Let me use a cleaner approach matching the secp_field_inv pattern:
    // x2 = a^(2^2-1)
    secp_scalar_mul(t, a, a);       // t = a^2
    secp_scalar_mul(x2, t, a);      // x2 = a^3

    // x3 = a^(2^3-1) = a^7
    secp_scalar_mul(t, x2, x2);     // t = a^6
    // wait, scalar_mul is modular multiplication, not squaring
    // a^6 = a^3 * a^3
    // Then a^7 = a^6 * a
    secp_scalar_mul(x3, t, a);      // x3 = a^7

    // ... (continue building up)
}
```

Actually, since we don't have a separate `secp_scalar_sqr`, and `secp_scalar_mul(r, a, a)` works for squaring, let me write the complete chain. The approach mirrors `secp_field_inv` but for `n-2`:

```glsl
void secp_scalar_inv(inout U256 r, in U256 a) {
    U256 x2, x3, x6, x8, x14, x28, x56, x112, x126, t;

    // x2 = a^(2^2-1) = a^3
    secp_scalar_mul(t, a, a);
    secp_scalar_mul(x2, t, a);

    // x3 = a^(2^3-1) = a^7
    secp_scalar_mul(t, x2, x2);
    secp_scalar_mul(x3, t, a);

    // x6 = a^(2^6-1)
    t = x3;
    for (int i = 0; i < 3; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x6, t, x3);

    // x8 = a^(2^8-1)
    t = x6;
    for (int i = 0; i < 2; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x8, t, x2);

    // x14 = a^(2^14-1)
    t = x8;
    for (int i = 0; i < 6; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x14, t, x6);

    // x28 = a^(2^28-1)
    t = x14;
    for (int i = 0; i < 14; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x28, t, x14);

    // x56 = a^(2^56-1)
    t = x28;
    for (int i = 0; i < 28; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x56, t, x28);

    // x112 = a^(2^112-1)
    t = x56;
    for (int i = 0; i < 56; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x112, t, x56);

    // x126 = a^(2^126-1)
    t = x112;
    for (int i = 0; i < 14; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x126, t, x14);

    // Now build the top 128 bits: n-2 top 128 bits are all 1s except bit 128 is 0
    // n-2 = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_BAAEDCE6_AF48A03B_BFD25E8C_D036413F
    //
    // Top 127 bits (bits 255..129) = all 1s
    // Then we need to handle the specific bit pattern of the lower 129 bits

    // t = x126 ^ (2^2) * x2 = a^(2^128 - 2) * a^3 = a^(2^128 + 1)
    // Actually this gets complex. Use the square-and-multiply for the lower 128 bits.

    // Build a^(2^128 - 1) from x126:
    // x126 is a^(2^126-1). Square twice and mul by a^3:
    secp_scalar_mul(t, x126, x126); // t = a^(2^127 - 2)
    secp_scalar_mul(t, t, t);       // t = a^(2^128 - 4)
    secp_scalar_mul(t, t, x2);      // t = a^(2^128 - 1)

    // Now: a^(n-2) = a^(2^256 - 2^128 - 1) * a^(lower 128 bits of n-2 minus (2^128-1))
    // This is getting complicated. Let's use a hybrid approach:
    // - Use the addition chain for the top 128 bits (all 1s)
    // - Use binary method for the lower 128 bits

    // r = t = a^(2^128-1) at this point
    // Square 128 times to shift up, then multiply by a^(lower 128 bits of n-2)
    for (int i = 0; i < 128; i++) secp_scalar_mul(t, t, t);

    // Now t = a^((2^128-1) * 2^128)
    // We need to multiply by a^(0xBAAEDCE6AF48A03BBFD25E8CD036413F)
    // Use square-and-multiply for these 128 bits

    // lower 128 bits of n-2: 0xBAAEDCE6_AF48A03B_BFD25E8C_D036413F
    const uint low128[4] = uint[4](0xD036413Fu, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u);

    U256 acc = t;
    bool started = true; // we already have the high bits
    for (int limb = 3; limb >= 0; limb--) {
        for (int bit = 31; bit >= 0; bit--) {
            secp_scalar_mul(acc, acc, acc);
            if (((low128[limb] >> bit) & 1u) != 0u) {
                secp_scalar_mul(acc, acc, a);
            }
        }
    }

    r = acc;
}
```

This uses ~126 + 2 + 128 + ~80 (avg half bits set) = ~336 scalar_muls. Still better than 512, but not great. However, note that the top 128 squarings with the addition chain cut it from 256 + 256 = 512 to ~336. More sophisticated chains could reduce further, but this is a good start.

**Actually, let me reconsider.** The existing implementation does 256 iterations, each with 1 squaring + conditionally 1 multiplication = ~384 scalar_muls on average. The addition chain approach above is ~336. The gain is modest (~12%).

A much bigger win is to **add a dedicated `secp_scalar_sqr`** (Task 5 covers field_sqr, but we also need scalar_sqr). Since scalar_mul does a full 8x8 schoolbook multiplication (64 multiplies), but squaring the same value only needs 36 multiplies (8 squares + 28 cross-terms doubled), this would make EVERY squaring operation ~44% faster.

**Revised approach for Task 4**: Add `secp_scalar_sqr` and use it in `scalar_inv`:

```glsl
void secp_scalar_sqr(inout U256 r, in U256 a) {
    // Squaring: exploit a[i]*a[j] == a[j]*a[i]
    // Same reduction as secp_scalar_mul but with optimized product computation
    uint64_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0ul;

    // Diagonal terms: a[i]^2
    for (int i = 0; i < 8; i++) {
        t[2*i] += uint64_t(a.d[i]) * uint64_t(a.d[i]);
    }

    // Cross terms: 2 * a[i] * a[j] for i < j
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0ul;
        for (int j = i + 1; j < 8; j++) {
            uint64_t prod = uint64_t(a.d[i]) * uint64_t(a.d[j]);
            // Add twice (shift left 1)
            uint64_t lo = (prod << 1) + t[i + j] + carry;
            // Handle overflow from the shift
            carry = (prod >> 63) + (lo < t[i + j] ? 1ul : 0ul); // approximate
            t[i + j] = lo & 0xFFFFFFFFul;
            carry += lo >> 32;
        }
        t[i + 8] += carry;
    }
    // ... then same reduction as scalar_mul
}
```

Hmm, the carry handling in GLSL for squaring is tricky and error-prone. The safer approach is simpler: just use the binary method with `secp_scalar_mul(t, t, t)` for squaring. The real wins come from reducing total operation count, not making individual squarings faster.

**Let's simplify Task 4**: Keep the generic binary method but add the addition-chain optimization for the top bits:

```glsl
void secp_scalar_inv(inout U256 r, in U256 a) {
    // Hybrid: addition chain for top 128 all-1 bits + binary method for lower 128 bits
    U256 x2, x4, x8, x16, x32, x64, x128, t;

    // Build a^(2^k - 1) using repeated square-and-multiply
    secp_scalar_mul(t, a, a);
    secp_scalar_mul(x2, t, a);            // a^3 = a^(2^2-1)

    secp_scalar_mul(t, x2, x2);
    secp_scalar_mul(t, t, t);
    secp_scalar_mul(x4, t, x2);           // a^(2^4-1)

    t = x4;
    for (int i = 0; i < 4; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x8, t, x4);           // a^(2^8-1)

    t = x8;
    for (int i = 0; i < 8; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x16, t, x8);          // a^(2^16-1)

    t = x16;
    for (int i = 0; i < 16; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x32, t, x16);         // a^(2^32-1)

    t = x32;
    for (int i = 0; i < 32; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x64, t, x32);         // a^(2^64-1)

    t = x64;
    for (int i = 0; i < 64; i++) secp_scalar_mul(t, t, t);
    secp_scalar_mul(x128, t, x64);        // a^(2^128-1)

    // Shift up 128 bits
    t = x128;
    for (int i = 0; i < 128; i++) secp_scalar_mul(t, t, t);
    // t = a^((2^128-1)*2^128) = a^(2^256 - 2^128)

    // Binary method for lower 128 bits of n-2: 0xBAAEDCE6_AF48A03B_BFD25E8C_D036413F
    const uint low128[4] = uint[4](0xD036413Fu, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u);

    for (int limb = 3; limb >= 0; limb--) {
        for (int bit = 31; bit >= 0; bit--) {
            secp_scalar_mul(t, t, t);
            if (((low128[limb] >> bit) & 1u) != 0u) {
                secp_scalar_mul(t, t, a);
            }
        }
    }

    r = t;
}
```

Total operations: 2+1+2+1+4+1+8+1+16+1+32+1+64+1+128+128+~80 = ~471 scalar_muls.

That's WORSE than the current 384. The addition chain doesn't help here because we can't avoid the 128 squarings to shift up and the 128 squarings+multiplies for the lower bits.

**Final verdict on Task 4**: The binary method is actually close to optimal for arbitrary exponents. The real optimization is making the individual `secp_scalar_mul` calls faster (Task 5: field_sqr). Skip the addition chain for now. **Replace Task 4 with: dedicated field_sqr.**

**Step 2: Verify existing test still passes**

Run: `cargo test --features vulkan -- --test-threads=1 vulkan_scalar_inv`
Expected: PASS

**Step 3: Commit**

No changes needed for Task 4 — it's been merged into Task 5.

---

### Task 4 (revised): Dedicated field_sqr

The inner loops of EC point operations are dominated by field multiplications. Many of these are squarings (`field_mul(r, a, a)`). A dedicated squaring function saves ~30% by exploiting symmetry.

**Files:**
- Modify: `vulkan/secp256k1.glsl:185-187` — replace `secp_field_sqr` body

**Step 1: Implement optimized field squaring**

Replace:
```glsl
void secp_field_sqr(inout U256 r, in U256 a) {
    secp_field_mul(r, a, a);
}
```

With:
```glsl
void secp_field_sqr(inout U256 r, in U256 a) {
    // Optimized squaring exploiting a[i]*a[j] == a[j]*a[i]
    uint64_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0ul;

    // Cross terms: sum a[i]*a[j] for i < j, accumulated once (will double later)
    for (int i = 0; i < 7; i++) {
        uint64_t carry = 0ul;
        for (int j = i + 1; j < 8; j++) {
            uint64_t prod = uint64_t(a.d[i]) * uint64_t(a.d[j]) + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        t[i + 8] += carry;
    }

    // Double all cross terms
    uint64_t carry = 0ul;
    for (int i = 1; i < 15; i++) {
        uint64_t v = (t[i] << 1) | carry;
        carry = t[i] >> 63;
        t[i] = v & 0xFFFFFFFFul;
        carry += v >> 32;
        // Hmm, this is getting messy with 64-bit intermediates stored as 32-bit
    }
    t[15] = (t[15] << 1) | carry;

    // Add diagonal terms: a[i]^2
    carry = 0ul;
    for (int i = 0; i < 8; i++) {
        uint64_t sq = uint64_t(a.d[i]) * uint64_t(a.d[i]);
        uint64_t lo = (sq & 0xFFFFFFFFul) + t[2*i] + carry;
        t[2*i] = lo & 0xFFFFFFFFul;
        carry = (lo >> 32) + (sq >> 32);
        lo = t[2*i+1] + carry;
        t[2*i+1] = lo & 0xFFFFFFFFul;
        carry = lo >> 32;
    }

    // Same secp256k1 reduction as field_mul
    {
        uint64_t c = 0ul;
        for (int i = 0; i < 8; i++) {
            uint64_t hi = t[i + 8];
            c += t[i] + hi * 977ul;
            t[i] = c & 0xFFFFFFFFul;
            c >>= 32;
            c += hi;
        }
        uint64_t c_lo = c * 977ul;
        uint64_t c_hi = c;
        c_lo += t[0];
        t[0] = c_lo & 0xFFFFFFFFul;
        c_lo >>= 32;
        c_lo += t[1] + c_hi;
        t[1] = c_lo & 0xFFFFFFFFul;
        c_lo >>= 32;
        for (int i = 2; i < 8 && c_lo != 0ul; i++) {
            c_lo += t[i];
            t[i] = c_lo & 0xFFFFFFFFul;
            c_lo >>= 32;
        }
    }

    for (int i = 0; i < 8; i++) r.d[i] = uint(t[i]);
    if (secp_cmp(r, SECP_P) >= 0) secp_sub256(r, r, SECP_P);
}
```

NOTE: The carry propagation in the doubling step above is tricky. Need to be very careful with 64-bit arithmetic in 32-bit limbs. An alternative simpler approach that's still faster than full mul:

```glsl
void secp_field_sqr(inout U256 r, in U256 a) {
    // Compute a^2 using the observation that cross terms appear twice.
    // We compute cross terms once and add them twice, plus diagonal terms.
    uint64_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0ul;

    // Cross terms (i < j): accumulate a[i]*a[j]
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0ul;
        for (int j = i + 1; j < 8; j++) {
            uint64_t prod = uint64_t(a.d[i]) * uint64_t(a.d[j]) + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        t[i + 8] += carry;
    }

    // Double: shift entire array left by 1 bit
    t[15] = t[15] << 1;
    for (int i = 14; i >= 1; i--) {
        t[i] = (t[i] << 1) | (t[i] >> 31); // Note: only 32 bits stored per slot
        // Actually t[] holds 64-bit values but only bottom 32 bits are meaningful
        // So shifting is: t[i] = (t[i] * 2), propagate carry
    }

    // Simpler: just add the cross-term array to itself
    uint64_t dc = 0ul;
    for (int i = 0; i < 16; i++) {
        dc += t[i] + t[i];
        t[i] = dc & 0xFFFFFFFFul;
        dc >>= 32;
    }

    // Add diagonal: a[i]^2
    dc = 0ul;
    for (int i = 0; i < 8; i++) {
        uint64_t sq = uint64_t(a.d[i]) * uint64_t(a.d[i]);
        dc += t[2*i] + (sq & 0xFFFFFFFFul);
        t[2*i] = dc & 0xFFFFFFFFul;
        dc >>= 32;
        dc += t[2*i+1] + (sq >> 32);
        t[2*i+1] = dc & 0xFFFFFFFFul;
        dc >>= 32;
    }

    // Reduce mod p (identical to field_mul)
    {
        uint64_t c = 0ul;
        for (int i = 0; i < 8; i++) {
            uint64_t hi = t[i + 8];
            c += t[i] + hi * 977ul;
            t[i] = c & 0xFFFFFFFFul;
            c >>= 32;
            c += hi;
        }
        uint64_t c_lo = c * 977ul;
        uint64_t c_hi = c;
        c_lo += t[0];
        t[0] = c_lo & 0xFFFFFFFFul;
        c_lo >>= 32;
        c_lo += t[1] + c_hi;
        t[1] = c_lo & 0xFFFFFFFFul;
        c_lo >>= 32;
        for (int i = 2; i < 8 && c_lo != 0ul; i++) {
            c_lo += t[i];
            t[i] = c_lo & 0xFFFFFFFFul;
            c_lo >>= 32;
        }
    }

    for (int i = 0; i < 8; i++) r.d[i] = uint(t[i]);
    if (secp_cmp(r, SECP_P) >= 0) secp_sub256(r, r, SECP_P);
}
```

Cross-term loop: 28 multiplies (vs 64 in full mul) + 8 diagonal squares = 36 total multiplies. That's a 44% reduction in multiplies.

**Step 2: Write a test for correctness**

Add to the existing `vulkan_field_mul_gx_gy` test: also test `field_sqr(GX)` and compare with `field_mul(GX, GX)`.

Create a new test shader `vulkan/test_field_sqr.comp`:
```glsl
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 1) in;
#include "secp256k1.glsl"

layout(std430, set = 0, binding = 0) readonly buffer A { uint a_limbs[8]; };
layout(std430, set = 0, binding = 1) buffer R { uint r_limbs[8]; };

void main() {
    U256 a, r;
    for (int i = 0; i < 8; i++) a.d[i] = a_limbs[i];
    secp_field_sqr(r, a);
    for (int i = 0; i < 8; i++) r_limbs[i] = r.d[i];
}
```

Add to build.rs to compile it. Add test in vulkan.rs that verifies `field_sqr(GX) == field_mul(GX, GX)` and `field_sqr(GX) == GX^2 mod p` (via num_bigint).

**Step 3: Build and test**

Run: `cargo build --release --features vulkan`
Run: `cargo test --features vulkan -- --test-threads=1 vulkan_field_sqr`
Run: `cargo test --features vulkan -- --test-threads=1 vulkan_mine_minimal_diagnostic`
Expected: All pass

**Step 4: Commit**

```bash
git add vulkan/secp256k1.glsl vulkan/test_field_sqr.comp build.rs src/mining/vulkan.rs
git commit -m "perf(vulkan): dedicated field_sqr with 44% fewer multiplies"
```

---

### Task 5: Increase Iterations Per Launch

After Tasks 1-4 reduce per-iteration cost, increase iterations per launch to amortize host-GPU sync overhead.

**Files:**
- Modify: `src/mining/vulkan.rs` — adjust `iterations_per_launch`

**Step 1: Increase iterations_per_launch**

After Tasks 1-3 are done and verified, try progressively larger values:

```rust
let iterations_per_launch: u32 = 64; // start here
```

If dispatch takes > 500ms, reduce. If < 100ms, increase.

**Step 2: Benchmark**

Run: `cargo run --release --features vulkan -- 'a*' --placeholder --backend vulkan`
Record keys/sec at different iteration counts.

**Step 3: Choose final value and commit**

Pick the value that gives good throughput without risking GPU timeout:

```bash
git add src/mining/vulkan.rs
git commit -m "perf(vulkan): increase iterations per launch after optimization"
```

---

### Task 6: Final Benchmark and Documentation

**Step 1: Run benchmark comparison**

Run Vulkan backend and record keys/sec:
```bash
cargo run --release --features vulkan -- 'a*' --placeholder --backend vulkan
```

Run CPU backend for comparison:
```bash
cargo run --release -- 'a*' --placeholder --backend cpu
```

If CUDA is available, also compare:
```bash
cargo run --release --features cuda -- 'a*' --placeholder --backend cuda
```

**Step 2: Update CLAUDE.md if needed**

If any architectural details changed (buffer layouts, shared memory usage, etc.), update the CLAUDE.md documentation.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with Vulkan optimization notes"
```
