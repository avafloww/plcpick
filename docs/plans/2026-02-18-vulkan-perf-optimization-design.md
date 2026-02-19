# Vulkan Backend Performance Optimization

## Problem

The Vulkan backend is functionally correct (all 33 tests pass, GPU output matches CPU re-verification) but runs at ~1 key/sec with 256 threads and 1 iteration per launch. The CUDA backend achieves 10K-100K keys/sec. We need roughly 10,000-100,000x speedup.

## Root Causes

Profiled bottlenecks, ordered by expected impact:

1. **Per-thread g_table copy (960 bytes/thread)**: Each thread copies 15 affine points (240 uints) from SSBO to a local array. This dominates register pressure and kills occupancy. CUDA stores this in `__device__` global memory, accessed directly.

2. **Per-iteration SHA256 buffer copies (1024 bytes/thread)**: Each iteration copies full CBOR templates (306 + 398 bytes) from SSBO into local `uint8_t[512]` arrays, then patches in-place. This wastes bandwidth and register space.

3. **Only 1 workgroup dispatched**: Currently dispatching 1 workgroup = 256 threads = 1 SM utilized. Modern GPUs have 20-80 SMs sitting idle.

4. **Generic binary scalar_inv (~256 scalar_muls)**: Uses bit-by-bit binary method. An addition chain for secp256k1's `n-2` can do it in ~40 operations.

5. **No dedicated field_sqr**: `secp_field_sqr` just calls `secp_field_mul(r, a, a)`. Exploiting the symmetry of squaring saves ~30% of multiplications in the inner loop.

## Approach: Incremental Optimization

Apply optimizations one at a time, measure after each, stop when we reach CUDA-competitive speeds.

### Phase 1: Shared Memory g_table

**What**: Load g_table into `shared` memory once per workgroup instead of per-thread local arrays.

**How**:
- Declare `shared AffinePoint shared_g_table[15]` in mine.comp
- First 15 threads of each workgroup load one point each from the SSBO
- `barrier()` before proceeding
- All threads read from shared_g_table instead of local copy

**Expected impact**: Eliminates 960 bytes/thread of register pressure. Should dramatically improve occupancy (more waves per SM).

### Phase 2: Stream SHA256 from SSBO

**What**: Instead of copying CBOR templates to local buffers, read directly from SSBO during SHA256 hashing, only patching the few bytes that differ.

**How**:
- Add a `sha256_hash_ssbo` variant that reads 64-byte blocks directly from the SSBO
- For patched regions (pubkey at 2 offsets, signature at 1 offset), overlay the patched bytes during block reads
- Eliminates the 512-byte local buffers for both unsigned and signed templates

**Expected impact**: Saves ~1024 bytes/thread of register/local memory. Further improves occupancy.

### Phase 3: Scale Up Workgroups

**What**: Dispatch enough workgroups to saturate all GPU SMs, and increase iterations per launch.

**How**:
- Query physical device properties for compute unit count
- Dispatch `4 * compute_units` workgroups (4x oversubscription for latency hiding)
- Start with 64-256 iterations per launch, tune based on dispatch time
- Add a timeout/watchdog to avoid GPU hangs during tuning

**Expected impact**: Linear scaling with GPU size. A 40-SM GPU should see ~40x throughput increase.

### Phase 4: Addition-Chain scalar_inv

**What**: Replace generic binary modular inverse with an optimized addition chain for secp256k1's group order.

**How**:
- Implement the standard addition chain for computing `x^(n-2) mod n` where n is secp256k1's group order
- Uses ~40 scalar_mul + scalar_sqr operations instead of ~256
- Port the known addition chain from libsecp256k1 or derive one

**Expected impact**: ~6x speedup on the inverse step. This is called once per iteration for ECDSA signing.

### Phase 5: Dedicated field_sqr

**What**: Implement `secp_field_sqr` that exploits the algebraic identity `a*a` having symmetric cross-terms.

**How**:
- In schoolbook multiplication of `a*a`, cross-terms `a[i]*a[j]` appear twice (for i!=j). Compute each once and double.
- Reduces from 64 multiplications to ~40 (8 squares + 28 cross-terms doubled)
- Used heavily in `secp_field_mul_reduce` calls where both operands are the same

**Expected impact**: ~30% speedup on field squaring, which dominates EC point operations. Compounds with all other EC improvements.

### Phase 6: Iterations per Launch

**What**: After phases 1-5 reduce per-iteration cost, increase iterations_per_launch to amortize dispatch overhead.

**How**:
- Start with 256 iterations per launch
- Measure dispatch time, target ~100ms per dispatch
- Auto-tune: if dispatch < 50ms, double iterations; if > 200ms, halve

**Expected impact**: Reduces host-GPU synchronization overhead from dominating to negligible.

## Measurement

After each phase:
1. Run `cargo run --release --features vulkan -- 'a*' --placeholder --backend vulkan`
2. Record keys/sec from the progress display
3. Compare against CPU backend on same machine
4. Document results before moving to next phase

## Non-Goals

- Montgomery arithmetic (Approach 3) — deferred unless Approach 1 doesn't reach target
- Multi-GPU support
- Async/overlapped dispatch (can add later if dispatch overhead is still significant)
