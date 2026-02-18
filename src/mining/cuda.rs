use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;
use k256::ecdsa::SigningKey;
use super::{Match, MiningBackend, MiningConfig};
use crate::pattern::glob_match;
use crate::plc::{build_signed_op, did_suffix, CborTemplate};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

const BLOCK_SIZE: u32 = 256;
const NUM_BLOCKS: u32 = 128;
const TOTAL_THREADS: u32 = BLOCK_SIZE * NUM_BLOCKS;
const ITERATIONS_PER_LAUNCH: u32 = 64;
const MAX_MATCHES: u32 = 64;

/// GPU match result, must match the CUDA GpuMatch struct layout exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuMatch {
    privkey: [u8; 32],
    signature: [u8; 64],
    suffix: [u8; 24],
    found: u32,
}

// SAFETY: GpuMatch is repr(C) with only primitive fields, safe for GPU transfer.
unsafe impl cudarc::driver::DeviceRepr for GpuMatch {}
// SAFETY: All-zeros is valid for GpuMatch (found=0 means no match).
unsafe impl ValidAsZeroBits for GpuMatch {}

/// Kernel parameters struct, must match the CUDA KernelParams layout exactly.
/// All pointer fields are raw CUdeviceptr values (u64).
#[repr(C)]
#[derive(Clone, Copy)]
struct KernelParams {
    unsigned_template: u64,         // const uint8_t *
    unsigned_template_len: u32,
    _pad1: u32,
    signed_template: u64,           // const uint8_t *
    signed_template_len: u32,
    unsigned_pubkey_offsets: [u32; 2],
    signed_pubkey_offsets: [u32; 2],
    signed_sig_offset: u32,
    pattern: u64,                   // const char *
    pattern_len: u32,
    _pad2: u32,
    scalars: u64,                   // uint8_t *
    pubkeys: u64,                   // JacobianPoint *
    stride: [u32; 8],              // U256 (8 × u32)
    stride_g: [u32; 24],           // JacobianPoint (3 × U256 = 24 × u32)
    matches: u64,                   // GpuMatch *
    match_count: u64,               // uint32_t *
    max_matches: u32,
    iterations_per_thread: u32,
    is_first_launch: u32,
    _pad3: u32,
}

// SAFETY: KernelParams is repr(C) with only primitive fields, safe for GPU transfer.
unsafe impl cudarc::driver::DeviceRepr for KernelParams {}

/// Get the raw device pointer value from a CudaSlice.
fn device_ptr<T>(slice: &CudaSlice<T>, stream: &cudarc::driver::CudaStream) -> u64 {
    let (ptr, _guard) = slice.device_ptr(stream);
    ptr
}

/// Get the raw device pointer value from a mutable CudaSlice.
fn device_ptr_mut<T>(slice: &mut CudaSlice<T>, stream: &cudarc::driver::CudaStream) -> u64 {
    use cudarc::driver::DevicePtrMut;
    let (ptr, _guard) = slice.device_ptr_mut(stream);
    ptr
}

pub struct CudaBackend {
    pub device_id: usize,
}

impl MiningBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn run(
        &self,
        config: &MiningConfig,
        stop: &AtomicBool,
        total: &AtomicU64,
        tx: mpsc::Sender<Match>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();

        // 1. Initialize CUDA context and load PTX
        let ctx = CudaContext::new(self.device_id)?;
        let stream = ctx.default_stream();
        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx)?;

        let init_g_table_fn = module.load_function("init_g_table")?;
        let mine_fn = module.load_function("mine_kernel")?;
        let stride_fn = module.load_function("compute_stride_g")?;

        // 1b. Initialize G table (must happen before any secp256k1 operations)
        unsafe {
            stream
                .launch_builder(&init_g_table_fn)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        // 2. Build CBOR template
        let tmpl = CborTemplate::new(&config.handle, &config.pds);

        // 3. Upload templates and pattern to device
        let unsigned_template_dev = stream.clone_htod(&tmpl.unsigned_bytes)?;
        let signed_template_dev = stream.clone_htod(&tmpl.signed_bytes)?;
        let pattern_dev = stream.clone_htod(&config.pattern)?;

        // 4. Compute stride_G on GPU
        let mut stride_dev: CudaSlice<u32> = stream.alloc_zeros(8)?;
        let mut stride_g_dev: CudaSlice<u32> = stream.alloc_zeros(24)?;

        unsafe {
            stream
                .launch_builder(&stride_fn)
                .arg(&TOTAL_THREADS)
                .arg(&mut stride_dev)
                .arg(&mut stride_g_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        let stride_host = stream.clone_dtoh(&stride_dev)?;
        let stride_g_host = stream.clone_dtoh(&stride_g_dev)?;

        let mut stride_arr = [0u32; 8];
        stride_arr.copy_from_slice(&stride_host);
        let mut stride_g_arr = [0u32; 24];
        stride_g_arr.copy_from_slice(&stride_g_host);

        // 5. Generate initial per-thread scalars
        let mut rng = rand::thread_rng();
        let base_key = SigningKey::random(&mut rng);
        let mut scalar_data = vec![0u8; TOTAL_THREADS as usize * 32];
        for tid in 0..TOTAL_THREADS as u64 {
            // scalar = base + tid (mod n, but tid is tiny relative to n so no wrap)
            let offset_scalar = k256::Scalar::from(tid);
            let thread_scalar = base_key.as_nonzero_scalar().as_ref() + &offset_scalar;
            let thread_bytes = thread_scalar.to_bytes();
            let start_idx = tid as usize * 32;
            scalar_data[start_idx..start_idx + 32].copy_from_slice(&thread_bytes);
        }

        let mut scalars_dev = stream.clone_htod(&scalar_data)?;

        // 6. Allocate pubkeys buffer (96 bytes per thread = 24 u32s per thread)
        // Pubkeys are computed by the kernel on first launch (is_first_launch = 1)
        let mut pubkeys_dev: CudaSlice<u32> =
            stream.alloc_zeros(TOTAL_THREADS as usize * 24)?;

        // 7. Allocate match output buffer
        let mut matches_dev: CudaSlice<GpuMatch> =
            stream.alloc_zeros(MAX_MATCHES as usize)?;
        let mut match_count_dev: CudaSlice<u32> = stream.alloc_zeros(1)?;

        // 8. Build launch config
        let launch_cfg = LaunchConfig {
            grid_dim: (NUM_BLOCKS, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut is_first_launch = 1u32;

        // 9. Main mining loop
        loop {
            if stop.load(Ordering::Relaxed) && !config.keep_going {
                break;
            }

            // Reset match count to 0
            stream.memcpy_htod(&[0u32], &mut match_count_dev)?;

            // Build kernel params
            let params = KernelParams {
                unsigned_template: device_ptr(&unsigned_template_dev, &stream),
                unsigned_template_len: tmpl.unsigned_bytes.len() as u32,
                _pad1: 0,
                signed_template: device_ptr(&signed_template_dev, &stream),
                signed_template_len: tmpl.signed_bytes.len() as u32,
                unsigned_pubkey_offsets: [
                    tmpl.unsigned_pubkey_offsets[0] as u32,
                    tmpl.unsigned_pubkey_offsets[1] as u32,
                ],
                signed_pubkey_offsets: [
                    tmpl.signed_pubkey_offsets[0] as u32,
                    tmpl.signed_pubkey_offsets[1] as u32,
                ],
                signed_sig_offset: tmpl.signed_sig_offset as u32,
                pattern: device_ptr(&pattern_dev, &stream),
                pattern_len: config.pattern.len() as u32,
                _pad2: 0,
                scalars: device_ptr_mut(&mut scalars_dev, &stream),
                pubkeys: device_ptr_mut(&mut pubkeys_dev, &stream),
                stride: stride_arr,
                stride_g: stride_g_arr,
                matches: device_ptr_mut(&mut matches_dev, &stream),
                match_count: device_ptr_mut(&mut match_count_dev, &stream),
                max_matches: MAX_MATCHES,
                iterations_per_thread: ITERATIONS_PER_LAUNCH,
                is_first_launch,
                _pad3: 0,
            };

            // Launch kernel
            unsafe {
                stream
                    .launch_builder(&mine_fn)
                    .arg(&params)
                    .launch(launch_cfg)?;
            }
            stream.synchronize()?;

            is_first_launch = 0;

            // Update total count
            let batch_ops = TOTAL_THREADS as u64 * ITERATIONS_PER_LAUNCH as u64;
            total.fetch_add(batch_ops, Ordering::Relaxed);

            // Check for matches
            let match_count_host = stream.clone_dtoh(&match_count_dev)?;
            let num_matches = match_count_host[0].min(MAX_MATCHES);

            if num_matches > 0 {
                let matches_host = stream.clone_dtoh(&matches_dev)?;

                for i in 0..num_matches as usize {
                    let gpu_match = &matches_host[i];
                    if gpu_match.found == 0 {
                        continue;
                    }

                    // CPU-side re-verification: reconstruct the full operation and
                    // verify the DID suffix matches what the GPU reported.
                    let privkey_bytes: [u8; 32] = gpu_match.privkey;
                    let signing_key = match SigningKey::from_bytes((&privkey_bytes).into()) {
                        Ok(k) => k,
                        Err(_) => continue, // invalid key, skip
                    };

                    let op = build_signed_op(&signing_key, &config.handle, &config.pds);
                    let suffix = did_suffix(&op);

                    // Verify the GPU's suffix matches
                    let gpu_suffix = std::str::from_utf8(&gpu_match.suffix)
                        .unwrap_or("");
                    if suffix != gpu_suffix {
                        eprintln!(
                            "warning: GPU match verification failed: GPU={} CPU={}",
                            gpu_suffix, suffix
                        );
                        continue;
                    }

                    // Double-check pattern match on CPU side
                    if !glob_match(&config.pattern, suffix.as_bytes()) {
                        continue;
                    }

                    let m = Match {
                        did: format!("did:plc:{suffix}"),
                        key_hex: data_encoding::HEXLOWER.encode(&privkey_bytes),
                        op,
                        attempts: total.load(Ordering::Relaxed),
                        elapsed: start.elapsed(),
                    };

                    if tx.send(m).is_err() {
                        return Ok(());
                    }

                    if !config.keep_going {
                        stop.store(true, Ordering::Relaxed);
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    use sha2::{Sha256, Digest};

    /// Helper: create CUDA context, load PTX, and initialize G table.
    fn setup() -> (
        std::sync::Arc<cudarc::driver::CudaContext>,
        std::sync::Arc<cudarc::driver::CudaStream>,
        std::sync::Arc<cudarc::driver::CudaModule>,
    ) {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let ptx = Ptx::from_src(PTX_SRC);
        let module = ctx.load_module(ptx).expect("load PTX");

        // Initialize G table before any secp256k1 operations
        let init_fn = module.load_function("init_g_table").expect("init_g_table");
        unsafe {
            stream
                .launch_builder(&init_fn)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        (ctx, stream, module)
    }

    #[test]
    fn gpu_sha256_matches_cpu() {
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_sha256").expect("test_sha256");

        // Test vector: "abc"
        let input = b"abc";
        let expected = Sha256::digest(input);

        let input_dev = stream.clone_htod(input.as_slice()).unwrap();
        let mut output_dev: CudaSlice<u8> = stream.alloc_zeros(32).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&input_dev)
                .arg(&(input.len() as u32))
                .arg(&mut output_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let output = stream.clone_dtoh(&output_dev).unwrap();
        assert_eq!(&output[..], &expected[..], "GPU SHA256 must match CPU");
    }

    #[test]
    fn gpu_sha256_empty_input() {
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_sha256").expect("test_sha256");

        let input: &[u8] = &[];
        let expected = Sha256::digest(input);

        let input_dev = stream.clone_htod(&[0u8; 1]).unwrap(); // need some allocation
        let mut output_dev: CudaSlice<u8> = stream.alloc_zeros(32).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&input_dev)
                .arg(&0u32)
                .arg(&mut output_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let output = stream.clone_dtoh(&output_dev).unwrap();
        assert_eq!(&output[..], &expected[..], "GPU SHA256 of empty must match CPU");
    }

    fn le_limbs_to_biguint(limbs: &[u32; 8]) -> num_bigint::BigUint {
        let mut bytes = vec![0u8; 32];
        for i in 0..8 {
            let b = limbs[i].to_le_bytes();
            bytes[i*4..i*4+4].copy_from_slice(&b);
        }
        num_bigint::BigUint::from_bytes_le(&bytes)
    }

    #[test]
    fn gpu_scalar_mul_mod_n() {
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_scalar_mul").expect("test_scalar_mul");

        // Use known values: a = GX interpreted as scalar, b = GY interpreted as scalar
        let a_limbs: [u32; 8] = [
            0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
            0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E,
        ];
        let b_limbs: [u32; 8] = [
            0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
            0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77,
        ];

        let n = num_bigint::BigUint::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16
        ).unwrap();
        let a_big = le_limbs_to_biguint(&a_limbs);
        let b_big = le_limbs_to_biguint(&b_limbs);
        let expected_big = (&a_big * &b_big) % &n;

        let expected_bytes = expected_big.to_bytes_le();
        let mut expected_limbs = [0u32; 8];
        for (i, chunk) in expected_bytes.chunks(4).enumerate() {
            if i < 8 {
                let mut buf = [0u8; 4];
                buf[..chunk.len()].copy_from_slice(chunk);
                expected_limbs[i] = u32::from_le_bytes(buf);
            }
        }

        let a_dev = stream.clone_htod(&a_limbs).unwrap();
        let b_dev = stream.clone_htod(&b_limbs).unwrap();
        let mut r_dev: CudaSlice<u32> = stream.alloc_zeros(8).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut r_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_result = stream.clone_dtoh(&r_dev).unwrap();
        assert_eq!(
            &gpu_result[..], &expected_limbs[..],
            "GPU scalar_mul(GX, GY) mod n must match.\nGPU: {:08x?}\nExpected: {:08x?}",
            gpu_result, expected_limbs
        );
    }

    #[test]
    fn gpu_scalar_mul_squaring() {
        // Test scalar_mul(a, a) — squaring case used heavily by scalar_inv
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_scalar_mul").expect("test_scalar_mul");

        let n = num_bigint::BigUint::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16
        ).unwrap();

        // Test with a value near n (worst case for reduction)
        let a_big = num_bigint::BigUint::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140", 16
        ).unwrap(); // n-1
        let expected_big = (&a_big * &a_big) % &n;

        let a_bytes = a_big.to_bytes_le();
        let mut a_limbs = [0u32; 8];
        for (i, chunk) in a_bytes.chunks(4).enumerate() {
            if i < 8 {
                let mut buf = [0u8; 4];
                buf[..chunk.len()].copy_from_slice(chunk);
                a_limbs[i] = u32::from_le_bytes(buf);
            }
        }

        let expected_bytes = expected_big.to_bytes_le();
        let mut expected_limbs = [0u32; 8];
        for (i, chunk) in expected_bytes.chunks(4).enumerate() {
            if i < 8 {
                let mut buf = [0u8; 4];
                buf[..chunk.len()].copy_from_slice(chunk);
                expected_limbs[i] = u32::from_le_bytes(buf);
            }
        }

        let a_dev = stream.clone_htod(&a_limbs).unwrap();
        let mut r_dev: CudaSlice<u32> = stream.alloc_zeros(8).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&a_dev)
                .arg(&a_dev) // same input for squaring
                .arg(&mut r_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_result = stream.clone_dtoh(&r_dev).unwrap();
        assert_eq!(
            &gpu_result[..], &expected_limbs[..],
            "GPU scalar_mul(n-1, n-1) mod n must match.\nGPU: {:08x?}\nExpected: {:08x?}",
            gpu_result, expected_limbs
        );
    }

    #[test]
    fn gpu_scalar_mul_chain() {
        // Test a chain of scalar_muls like scalar_inv does: a^2, a^4, a^8...
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_scalar_mul").expect("test_scalar_mul");

        let n = num_bigint::BigUint::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16
        ).unwrap();

        let a_big = num_bigint::BigUint::from(7u32);
        let mut cpu_val = a_big.clone();
        let mut gpu_limbs = [0u32; 8];
        gpu_limbs[0] = 7;

        // Do 10 squarings on CPU and GPU
        for step in 0..10 {
            cpu_val = (&cpu_val * &cpu_val) % &n;

            let a_dev = stream.clone_htod(&gpu_limbs).unwrap();
            let mut r_dev: CudaSlice<u32> = stream.alloc_zeros(8).unwrap();

            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&a_dev)
                    .arg(&a_dev)
                    .arg(&mut r_dev)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            stream.synchronize().unwrap();

            let gpu_result = stream.clone_dtoh(&r_dev).unwrap();
            gpu_limbs = [
                gpu_result[0], gpu_result[1], gpu_result[2], gpu_result[3],
                gpu_result[4], gpu_result[5], gpu_result[6], gpu_result[7],
            ];

            let cpu_bytes = cpu_val.to_bytes_le();
            let mut expected_limbs = [0u32; 8];
            for (i, chunk) in cpu_bytes.chunks(4).enumerate() {
                if i < 8 {
                    let mut buf = [0u8; 4];
                    buf[..chunk.len()].copy_from_slice(chunk);
                    expected_limbs[i] = u32::from_le_bytes(buf);
                }
            }

            assert_eq!(
                &gpu_limbs[..], &expected_limbs[..],
                "Squaring chain diverged at step {}.\nGPU: {:08x?}\nCPU: {:08x?}",
                step, gpu_limbs, expected_limbs
            );
        }
    }

    #[test]
    fn gpu_scalar_inv_mod_n() {
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_scalar_inv").expect("test_scalar_inv");

        // Use a = 7 (small known value)
        let a_limbs: [u32; 8] = [7, 0, 0, 0, 0, 0, 0, 0];

        // Expected: 7^(-1) mod n
        // n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        // 7^(-1) mod n: use extended Euclidean
        let n = num_bigint::BigUint::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16
        ).unwrap();
        let a_big = num_bigint::BigUint::from(7u32);
        // Fermat: a^(n-2) mod n
        let nm2 = &n - num_bigint::BigUint::from(2u32);
        let expected_big = a_big.modpow(&nm2, &n);

        let expected_bytes = expected_big.to_bytes_le();
        let mut expected_limbs = [0u32; 8];
        for (i, chunk) in expected_bytes.chunks(4).enumerate() {
            if i < 8 {
                let mut buf = [0u8; 4];
                buf[..chunk.len()].copy_from_slice(chunk);
                expected_limbs[i] = u32::from_le_bytes(buf);
            }
        }

        let a_dev = stream.clone_htod(&a_limbs).unwrap();
        let mut r_dev: CudaSlice<u32> = stream.alloc_zeros(8).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&a_dev)
                .arg(&mut r_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_result = stream.clone_dtoh(&r_dev).unwrap();
        assert_eq!(
            &gpu_result[..], &expected_limbs[..],
            "GPU scalar_inv(7) mod n must match.\nGPU: {:08x?}\nExpected: {:08x?}",
            gpu_result, expected_limbs
        );
    }

    #[test]
    fn gpu_field_mul_gx_gy() {
        // Test field_mul(GX, GY) against num-bigint reference
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_field_mul").expect("test_field_mul");

        // GX and GY as LE u32 limbs (matching CUDA U256 layout)
        let gx_limbs: [u32; 8] = [
            0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
            0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E,
        ];
        let gy_limbs: [u32; 8] = [
            0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
            0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77,
        ];

        // Compute GX * GY mod p using big integers
        // p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        use std::str::FromStr;
        let p = num_bigint::BigUint::from_str(
            "115792089237316195423570985008687907853269984665640564039457584007908834671663"
        ).unwrap();
        let gx_big = num_bigint::BigUint::parse_bytes(
            b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16
        ).unwrap();
        let gy_big = num_bigint::BigUint::parse_bytes(
            b"483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16
        ).unwrap();
        let expected_big = (&gx_big * &gy_big) % &p;

        // Convert expected to LE u32 limbs
        let expected_bytes = expected_big.to_bytes_le();
        let mut expected_limbs = [0u32; 8];
        for (i, chunk) in expected_bytes.chunks(4).enumerate() {
            if i < 8 {
                let mut buf = [0u8; 4];
                buf[..chunk.len()].copy_from_slice(chunk);
                expected_limbs[i] = u32::from_le_bytes(buf);
            }
        }

        let a_dev = stream.clone_htod(&gx_limbs).unwrap();
        let b_dev = stream.clone_htod(&gy_limbs).unwrap();
        let mut r_dev: CudaSlice<u32> = stream.alloc_zeros(8).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut r_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_result = stream.clone_dtoh(&r_dev).unwrap();
        assert_eq!(
            &gpu_result[..], &expected_limbs[..],
            "GPU field_mul(GX, GY) must match reference.\nGPU: {:08x?}\nExpected: {:08x?}",
            gpu_result, expected_limbs
        );
    }

    #[test]
    fn gpu_point_double_g() {
        // Test that point_double(G) gives correct 2*G
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_point_double_g").expect("test_point_double_g");

        // Known 2*G compressed: 02 c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
        let expected: [u8; 33] = [
            0x02,
            0xc6, 0x04, 0x7f, 0x94, 0x41, 0xed, 0x7d, 0x6d,
            0x30, 0x45, 0x40, 0x6e, 0x95, 0xc0, 0x7c, 0xd8,
            0x5c, 0x77, 0x8e, 0x4b, 0x8c, 0xef, 0x3c, 0xa7,
            0xab, 0xac, 0x09, 0xb9, 0x5c, 0x70, 0x9e, 0xe5,
        ];

        let mut pubkey_dev: CudaSlice<u8> = stream.alloc_zeros(33).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&mut pubkey_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_pubkey = stream.clone_dtoh(&pubkey_dev).unwrap();
        assert_eq!(
            &gpu_pubkey[..], &expected[..],
            "point_double(G) must equal 2*G"
        );
    }

    #[test]
    fn gpu_scalar_mul_g_identity() {
        // scalar = 1 should give generator G
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_scalar_mul_G").expect("test_scalar_mul_G");

        // scalar = 1 in big-endian 32 bytes
        let mut scalar = [0u8; 32];
        scalar[31] = 1;

        // Known compressed G: 02 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        let expected: [u8; 33] = [
            0x02,
            0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
            0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
            0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
            0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
        ];

        let scalar_dev = stream.clone_htod(&scalar).unwrap();
        let mut pubkey_dev: CudaSlice<u8> = stream.alloc_zeros(33).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&scalar_dev)
                .arg(&mut pubkey_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_pubkey = stream.clone_dtoh(&pubkey_dev).unwrap();
        assert_eq!(&gpu_pubkey[..], &expected[..], "1*G must equal G");
    }

    #[test]
    fn gpu_scalar_mul_g_small_scalars() {
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_scalar_mul_G").expect("test_scalar_mul_G");

        // Test scalars 2, 3, 15, 16, 17, 256
        let test_values: &[u64] = &[2, 3, 15, 16, 17, 256];
        for &val in test_values {
            let mut bytes = [0u8; 32];
            bytes[24..].copy_from_slice(&val.to_be_bytes());
            let key = k256::SecretKey::from_slice(&bytes).unwrap();
            let expected = key.public_key().to_encoded_point(true);

            let scalar = bytes;

            let scalar_dev = stream.clone_htod(&scalar).unwrap();
            let mut pubkey_dev: CudaSlice<u8> = stream.alloc_zeros(33).unwrap();

            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(&scalar_dev)
                    .arg(&mut pubkey_dev)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            stream.synchronize().unwrap();

            let gpu_pubkey = stream.clone_dtoh(&pubkey_dev).unwrap();
            assert_eq!(
                &gpu_pubkey[..], expected.as_bytes(),
                "GPU scalar_mul_G failed for scalar={}",
                val
            );
        }
    }

    #[test]
    fn gpu_scalar_mul_g_matches_k256() {
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_scalar_mul_G").expect("test_scalar_mul_G");

        let key = SigningKey::random(&mut rand::thread_rng());
        let scalar_bytes = key.to_bytes(); // 32 bytes big-endian

        // CPU: get compressed pubkey
        let cpu_pubkey = key.verifying_key().to_encoded_point(true);
        let cpu_bytes = cpu_pubkey.as_bytes(); // 33 bytes compressed

        let scalar_dev = stream.clone_htod(scalar_bytes.as_slice()).unwrap();
        let mut pubkey_dev: CudaSlice<u8> = stream.alloc_zeros(33).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&scalar_dev)
                .arg(&mut pubkey_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_pubkey = stream.clone_dtoh(&pubkey_dev).unwrap();
        assert_eq!(&gpu_pubkey[..], cpu_bytes, "GPU scalar_mul_G must match k256");
    }

    #[test]
    fn gpu_ecdsa_sign_matches_k256() {
        use k256::ecdsa::Signature;
        use signature::hazmat::PrehashSigner;

        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_ecdsa_sign").expect("test_ecdsa_sign");

        let key = SigningKey::random(&mut rand::thread_rng());
        let privkey_bytes = key.to_bytes();

        // Create a message hash (just SHA256 of some data)
        let msg_hash = Sha256::digest(b"test message for ecdsa");

        // CPU sign using prehash (GPU takes the hash directly, doesn't hash again)
        let cpu_sig: Signature = key.sign_prehash(&msg_hash[..]).expect("sign_prehash");
        let (r_cpu, s_cpu) = cpu_sig.split_bytes();

        let privkey_dev = stream.clone_htod(privkey_bytes.as_slice()).unwrap();
        let msg_hash_dev = stream.clone_htod(msg_hash.as_slice()).unwrap();
        let mut sig_dev: CudaSlice<u8> = stream.alloc_zeros(64).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&privkey_dev)
                .arg(&msg_hash_dev)
                .arg(&mut sig_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_sig = stream.clone_dtoh(&sig_dev).unwrap();
        let gpu_r = &gpu_sig[..32];
        let gpu_s = &gpu_sig[32..64];

        assert_eq!(gpu_r, &r_cpu[..], "GPU ECDSA r must match k256");
        assert_eq!(gpu_s, &s_cpu[..], "GPU ECDSA s must match k256");
    }

    #[test]
    fn gpu_base58_matches_cpu() {
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_base58").expect("test_base58");

        // Build a 35-byte multicodec-prefixed pubkey (0xe7, 0x01, + 33 compressed pubkey)
        let key = SigningKey::random(&mut rand::thread_rng());
        let pubkey = key.verifying_key().to_encoded_point(true);
        let mut input = Vec::with_capacity(35);
        input.extend_from_slice(&[0xe7, 0x01]);
        input.extend_from_slice(pubkey.as_bytes());
        assert_eq!(input.len(), 35);

        // CPU base58
        let cpu_encoded = bs58::encode(&input).into_string();
        assert_eq!(cpu_encoded.len(), 48);

        let input_dev = stream.clone_htod(&input).unwrap();
        let mut output_dev: CudaSlice<u8> = stream.alloc_zeros(48).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&input_dev)
                .arg(&mut output_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_output = stream.clone_dtoh(&output_dev).unwrap();
        let gpu_str = std::str::from_utf8(&gpu_output).expect("valid utf8");
        assert_eq!(gpu_str, &cpu_encoded, "GPU base58 must match CPU");
    }

    #[test]
    fn gpu_base64url_matches_cpu() {
        use base64::Engine as _;
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_base64url").expect("test_base64url");

        // 64-byte ECDSA signature
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (i * 17 + 42) as u8;
        }

        let cpu_encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&input);
        assert_eq!(cpu_encoded.len(), 86);

        let input_dev = stream.clone_htod(&input).unwrap();
        let mut output_dev: CudaSlice<u8> = stream.alloc_zeros(86).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&input_dev)
                .arg(&mut output_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_output = stream.clone_dtoh(&output_dev).unwrap();
        let gpu_str = std::str::from_utf8(&gpu_output).expect("valid utf8");
        assert_eq!(gpu_str, &cpu_encoded, "GPU base64url must match CPU");
    }

    #[test]
    fn gpu_base32_matches_cpu() {
        let (_ctx, stream, module) = setup();
        let f = module.load_function("test_base32").expect("test_base32");

        // 15-byte SHA256 prefix (simulating DID hash truncation)
        let hash = Sha256::digest(b"test did hash");
        let input: Vec<u8> = hash[..15].to_vec();

        let cpu_encoded = data_encoding::BASE32_NOPAD
            .encode(&input)
            .to_ascii_lowercase();
        assert_eq!(cpu_encoded.len(), 24);

        let input_dev = stream.clone_htod(&input).unwrap();
        let mut output_dev: CudaSlice<u8> = stream.alloc_zeros(24).unwrap();

        unsafe {
            stream
                .launch_builder(&f)
                .arg(&input_dev)
                .arg(&mut output_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();

        let gpu_output = stream.clone_dtoh(&output_dev).unwrap();
        let gpu_str = std::str::from_utf8(&gpu_output).expect("valid utf8");
        assert_eq!(gpu_str, &cpu_encoded, "GPU base32 must match CPU");
    }

    #[test]
    fn gpu_mining_finds_valid_match() {
        // Mine with an easy pattern that should match quickly
        let config = MiningConfig {
            pattern: b"a*".to_vec(),
            handle: "test.bsky.social".into(),
            pds: "https://bsky.social".into(),
            keep_going: false,
        };

        let stop = AtomicBool::new(false);
        let total = AtomicU64::new(0);
        let (tx, rx) = mpsc::channel();

        let backend = CudaBackend { device_id: 0 };
        backend.run(&config, &stop, &total, tx).expect("CUDA mining should succeed");

        // Should have found at least one match (pattern "a*" matches ~1/32 DIDs)
        let m = rx.try_recv().expect("should have found a match");
        assert!(m.did.starts_with("did:plc:a"), "DID should match pattern a*: {}", m.did);

        // Verify the match is valid by re-deriving from the private key
        let key = SigningKey::from_bytes(
            (&data_encoding::HEXLOWER.decode(m.key_hex.as_bytes()).unwrap()[..]).into()
        ).unwrap();
        let op = build_signed_op(&key, "test.bsky.social", "https://bsky.social");
        let suffix = did_suffix(&op);
        assert_eq!(m.did, format!("did:plc:{suffix}"), "GPU match must be valid on CPU re-verification");
    }
}
