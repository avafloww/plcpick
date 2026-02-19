use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use k256::elliptic_curve::ff::PrimeField;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{ProjectivePoint, Scalar};
use wgpu::util::DeviceExt;

use super::{Match, MiningBackend, MiningConfig};
use crate::pattern::glob_match;
use crate::plc::{build_signed_op, did_suffix, CborTemplate};

const WORKGROUP_SIZE: u32 = 256;
const MAX_MATCHES: u32 = 64;
const EC_RESULT_STRIDE: u32 = 142; // u32s per thread in ec_pass output
const MATCH_STRIDE: u32 = 32; // u32s per match slot

pub struct WgpuBackend {
    pub device_index: usize,
}

/// Compose WGSL shader source by concatenating module files.
fn compose_ec_pass_shader() -> String {
    let field = include_str!("../../wgpu/shaders/field.wgsl");
    let curve = include_str!("../../wgpu/shaders/curve.wgsl");
    let scalar = include_str!("../../wgpu/shaders/scalar.wgsl");
    let sha256 = include_str!("../../wgpu/shaders/sha256.wgsl");
    let hmac = include_str!("../../wgpu/shaders/hmac_drbg.wgsl");
    let encoding = include_str!("../../wgpu/shaders/encoding.wgsl");
    let ec_pass = include_str!("../../wgpu/shaders/ec_pass.wgsl");
    format!("{field}\n{curve}\n{scalar}\n{sha256}\n{hmac}\n{encoding}\n{ec_pass}")
}

fn compose_hash_pass_shader() -> String {
    let sha256 = include_str!("../../wgpu/shaders/sha256.wgsl");
    let encoding = include_str!("../../wgpu/shaders/encoding.wgsl");
    let pattern = include_str!("../../wgpu/shaders/pattern.wgsl");
    let hash_pass = include_str!("../../wgpu/shaders/hash_pass.wgsl");
    format!("{sha256}\n{encoding}\n{pattern}\n{hash_pass}")
}

/// Compute the precomputed G table [1*G, 2*G, ..., 15*G] on CPU.
/// Returns (g_table_x, g_table_y) as flattened u32 arrays (15 * 8 = 120 u32s each).
fn compute_g_table() -> (Vec<u32>, Vec<u32>) {
    let g = ProjectivePoint::GENERATOR;
    let mut g_table_x = Vec::with_capacity(120);
    let mut g_table_y = Vec::with_capacity(120);

    for i in 1..=15u64 {
        let point = g * Scalar::from(i);
        let affine = point.to_affine();
        let encoded = affine.to_encoded_point(false);
        let x_bytes = encoded.x().unwrap();
        let y_bytes = encoded.y().unwrap();

        // Convert 32 big-endian bytes to 8 little-endian u32 limbs
        for limb_idx in 0..8 {
            let byte_offset = (7 - limb_idx) * 4;
            let x_limb = u32::from_be_bytes([
                x_bytes[byte_offset],
                x_bytes[byte_offset + 1],
                x_bytes[byte_offset + 2],
                x_bytes[byte_offset + 3],
            ]);
            let y_limb = u32::from_be_bytes([
                y_bytes[byte_offset],
                y_bytes[byte_offset + 1],
                y_bytes[byte_offset + 2],
                y_bytes[byte_offset + 3],
            ]);
            g_table_x.push(x_limb);
            g_table_y.push(y_limb);
        }
    }

    (g_table_x, g_table_y)
}

/// Convert a k256::Scalar to 8 little-endian u32 limbs.
fn scalar_to_limbs(s: &Scalar) -> [u32; 8] {
    let bytes = s.to_bytes(); // 32 bytes, big-endian
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        let offset = (7 - i) * 4;
        limbs[i] = u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
    }
    limbs
}

/// Convert 8 little-endian u32 limbs back to 32 big-endian bytes.
fn limbs_to_bytes(limbs: &[u32; 8]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..8 {
        let offset = (7 - i) * 4;
        let be = limbs[i].to_be_bytes();
        bytes[offset] = be[0];
        bytes[offset + 1] = be[1];
        bytes[offset + 2] = be[2];
        bytes[offset + 3] = be[3];
    }
    bytes
}

/// Prepare unsigned CBOR template as u32 array (one byte per u32).
fn template_to_u32_array(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|&b| b as u32).collect()
}

impl MiningBackend for WgpuBackend {
    fn name(&self) -> &str {
        "wgpu"
    }

    fn run(
        &self,
        config: &MiningConfig,
        stop: &AtomicBool,
        total: &AtomicU64,
        tx: mpsc::Sender<Match>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mining_start = Instant::now();

        // Dispatch parameters — conservative to avoid TDR
        let num_workgroups: u32 = 1;
        let total_threads: u32 = WORKGROUP_SIZE * num_workgroups;

        // 1. Init wgpu
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        ?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("plcpick"),
                ..Default::default()
            },
        ))?;

        // 2. Build CBOR template
        let tmpl = CborTemplate::new(&config.handle, &config.pds);

        // Unsigned template as u32 array + pubkey offsets
        let unsigned_tmpl_u32 = template_to_u32_array(&tmpl.unsigned_bytes);
        let unsigned_pubkey_offsets: [u32; 4] = [
            tmpl.unsigned_pubkey_offsets[0] as u32,
            tmpl.unsigned_pubkey_offsets[1] as u32,
            48, // length of base58 pubkey
            48,
        ];

        // Signed template as u32 array
        let signed_tmpl_u32 = template_to_u32_array(&tmpl.signed_bytes);

        // Pattern as u32 array (one char per u32)
        let pattern_u32: Vec<u32> = config.pattern.iter().map(|&b| b as u32).collect();

        // 3. Compute G table on CPU
        let (g_table_x, g_table_y) = compute_g_table();

        // 4. Compose and create shader modules
        let ec_shader_src = compose_ec_pass_shader();
        let hash_shader_src = compose_hash_pass_shader();

        let ec_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ec_pass"),
            source: wgpu::ShaderSource::Wgsl(ec_shader_src.into()),
        });
        let hash_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hash_pass"),
            source: wgpu::ShaderSource::Wgsl(hash_shader_src.into()),
        });

        // 5. Create compute pipelines
        let ec_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ec_pass_pipeline"),
            layout: None, // auto-layout
            module: &ec_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let hash_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hash_pass_pipeline"),
            layout: None,
            module: &hash_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // 6. Generate initial scalars
        let mut rng = rand::thread_rng();
        let base_key = k256::ecdsa::SigningKey::random(&mut rng);
        let mut scalar_data = vec![0u32; total_threads as usize * 8];
        for tid in 0..total_threads {
            let offset_scalar = Scalar::from(tid as u64);
            let thread_scalar = *base_key.as_nonzero_scalar().as_ref() + &offset_scalar;
            let limbs = scalar_to_limbs(&thread_scalar);
            let start = tid as usize * 8;
            scalar_data[start..start + 8].copy_from_slice(&limbs);
        }

        // 7. Create GPU buffers
        let scalars_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scalars"),
            contents: bytemuck::cast_slice(&scalar_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let g_table_x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("g_table_x"),
            contents: bytemuck::cast_slice(&g_table_x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let g_table_y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("g_table_y"),
            contents: bytemuck::cast_slice(&g_table_y),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Stride G (computed on CPU): stride * G where stride = total_threads
        let stride_g_point = ProjectivePoint::GENERATOR * Scalar::from(total_threads as u64);
        let stride_g_affine = stride_g_point.to_affine();
        let stride_g_encoded = stride_g_affine.to_encoded_point(false);
        let mut stride_g_data = vec![0u32; 16]; // x[8] + y[8]
        for limb_idx in 0..8 {
            let byte_offset = (7 - limb_idx) * 4;
            stride_g_data[limb_idx] = u32::from_be_bytes([
                stride_g_encoded.x().unwrap()[byte_offset],
                stride_g_encoded.x().unwrap()[byte_offset + 1],
                stride_g_encoded.x().unwrap()[byte_offset + 2],
                stride_g_encoded.x().unwrap()[byte_offset + 3],
            ]);
            stride_g_data[8 + limb_idx] = u32::from_be_bytes([
                stride_g_encoded.y().unwrap()[byte_offset],
                stride_g_encoded.y().unwrap()[byte_offset + 1],
                stride_g_encoded.y().unwrap()[byte_offset + 2],
                stride_g_encoded.y().unwrap()[byte_offset + 3],
            ]);
        }
        let stride_g_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stride_g"),
            contents: bytemuck::cast_slice(&stride_g_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let unsigned_tmpl_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unsigned_template"),
            contents: bytemuck::cast_slice(&unsigned_tmpl_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let unsigned_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unsigned_pubkey_offsets"),
            contents: bytemuck::cast_slice(&unsigned_pubkey_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // EC pass results buffer
        let ec_results_size = (total_threads * EC_RESULT_STRIDE * 4) as u64;
        let ec_results_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ec_results"),
            size: ec_results_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // EC pass uniform params
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct EcPassParams {
            is_first_launch: u32,
            num_threads: u32,
            unsigned_template_byte_len: u32,
            _pad: u32,
        }
        let ec_params = EcPassParams {
            is_first_launch: 1,
            num_threads: total_threads,
            unsigned_template_byte_len: tmpl.unsigned_bytes.len() as u32,
            _pad: 0,
        };
        let ec_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ec_params"),
            contents: bytemuck::bytes_of(&ec_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Hash pass buffers
        let signed_tmpl_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("signed_template"),
            contents: bytemuck::cast_slice(&signed_tmpl_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pattern_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pattern"),
            contents: bytemuck::cast_slice(&pattern_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let match_buf_size = (MAX_MATCHES * MATCH_STRIDE * 4) as u64;
        let match_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matches"),
            size: match_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let match_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("match_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct HashPassParams {
            num_threads: u32,
            pattern_len: u32,
            signed_template_byte_len: u32,
            max_matches: u32,
            pubkey_offset_1: u32,
            pubkey_offset_2: u32,
            sig_offset: u32,
            _pad: u32,
        }
        let hash_params = HashPassParams {
            num_threads: total_threads,
            pattern_len: config.pattern.len() as u32,
            signed_template_byte_len: tmpl.signed_bytes.len() as u32,
            max_matches: MAX_MATCHES,
            pubkey_offset_1: tmpl.signed_pubkey_offsets[0] as u32,
            pubkey_offset_2: tmpl.signed_pubkey_offsets[1] as u32,
            sig_offset: tmpl.signed_sig_offset as u32,
            _pad: 0,
        };
        let hash_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hash_params"),
            contents: bytemuck::bytes_of(&hash_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Staging buffers for readback
        let match_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("match_count_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let match_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("match_staging"),
            size: match_buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 8. Create bind groups
        let ec_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ec_bind_group"),
            layout: &ec_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scalars_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: g_table_x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: g_table_y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: stride_g_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: unsigned_tmpl_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: unsigned_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: ec_results_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: ec_params_buf.as_entire_binding() },
            ],
        });

        let hash_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hash_bind_group"),
            layout: &hash_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ec_results_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: signed_tmpl_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pattern_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: match_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: match_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: hash_params_buf.as_entire_binding() },
            ],
        });

        // 9. Main mining loop
        let mut is_first = true;
        let mut launch_count = 0u64;

        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }

            // Update ec_params for is_first_launch
            if !is_first {
                let params = EcPassParams {
                    is_first_launch: 0,
                    num_threads: total_threads,
                    unsigned_template_byte_len: tmpl.unsigned_bytes.len() as u32,
                    _pad: 0,
                };
                queue.write_buffer(&ec_params_buf, 0, bytemuck::bytes_of(&params));
            }

            // Reset match count
            queue.write_buffer(&match_count_buf, 0, &0u32.to_le_bytes());

            // Encode and submit both passes
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mining_encoder"),
            });

            // EC pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ec_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&ec_pipeline);
                pass.set_bind_group(0, &ec_bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Hash pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hash_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&hash_pipeline);
                pass.set_bind_group(0, &hash_bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Copy match count + matches to staging for readback
            encoder.copy_buffer_to_buffer(&match_count_buf, 0, &match_count_staging, 0, 4);
            encoder.copy_buffer_to_buffer(&match_buf, 0, &match_staging, 0, match_buf_size);

            queue.submit(Some(encoder.finish()));

            // Read back match count
            let match_count_slice = match_count_staging.slice(..);
            match_count_slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

            let count = {
                let data = match_count_slice.get_mapped_range();
                u32::from_le_bytes([data[0], data[1], data[2], data[3]])
            };
            match_count_staging.unmap();

            // Process matches
            if count > 0 {
                let matches_slice = match_staging.slice(..);
                matches_slice.map_async(wgpu::MapMode::Read, |_| {});
                let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

                let match_data = {
                    let data = matches_slice.get_mapped_range();
                    let u32_data: &[u32] =
                        bytemuck::cast_slice(&data[..count.min(MAX_MATCHES) as usize * MATCH_STRIDE as usize * 4]);
                    u32_data.to_vec()
                };
                match_staging.unmap();

                for slot in 0..count.min(MAX_MATCHES) {
                    let offset = slot as usize * MATCH_STRIDE as usize;
                    let mut privkey_limbs = [0u32; 8];
                    privkey_limbs.copy_from_slice(&match_data[offset..offset + 8]);
                    let privkey_bytes = limbs_to_bytes(&privkey_limbs);

                    let mut suffix_chars = String::new();
                    for i in 0..24 {
                        suffix_chars.push(match_data[offset + 8 + i] as u8 as char);
                    }

                    // CPU-side verification
                    let key_hex = data_encoding::HEXLOWER.encode(&privkey_bytes);
                    let signing_key =
                        k256::ecdsa::SigningKey::from_bytes((&privkey_bytes).into())
                            .map_err(|e| format!("invalid key from GPU: {e}"))?;

                    let op = build_signed_op(&signing_key, &config.handle, &config.pds);
                    let cpu_did = did_suffix(&op);
                    let cpu_suffix = cpu_did.strip_prefix("did:plc:").unwrap_or(&cpu_did);

                    if cpu_suffix == suffix_chars && glob_match(&config.pattern, cpu_suffix.as_bytes()) {
                        let _ = tx.send(Match {
                            did: cpu_did.clone(),
                            key_hex,
                            op,
                            attempts: total.load(Ordering::Relaxed),
                            elapsed: mining_start.elapsed(),
                        });
                    }
                }
            }

            is_first = false;
            launch_count += 1;
            total.fetch_add(total_threads as u64, Ordering::Relaxed);

            // Advance scalars for next launch (CPU-side for now)
            let stride = Scalar::from(total_threads as u64);
            for tid in 0..total_threads {
                let offset = tid as usize * 8;
                let mut limbs = [0u32; 8];
                limbs.copy_from_slice(&scalar_data[offset..offset + 8]);
                let scalar_bytes = limbs_to_bytes(&limbs);
                let current = Scalar::from_repr(scalar_bytes.into())
                    .expect("invalid scalar");
                let next = current + stride;
                let next_limbs = scalar_to_limbs(&next);
                scalar_data[offset..offset + 8].copy_from_slice(&next_limbs);
            }
            queue.write_buffer(&scalars_buf, 0, bytemuck::cast_slice(&scalar_data));
        }

        Ok(())
    }
}
