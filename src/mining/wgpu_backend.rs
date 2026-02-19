use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{ProjectivePoint, Scalar};
use wgpu::util::DeviceExt;

use super::{Match, MiningBackend, MiningConfig};
use crate::pattern::glob_match;
use crate::plc::{build_signed_op, did_suffix, CborTemplate};

const WORKGROUP_SIZE: u32 = 256;
const NUM_WORKGROUPS: u32 = 128;
const TOTAL_THREADS: u32 = WORKGROUP_SIZE * NUM_WORKGROUPS;
const ITERATIONS_PER_THREAD: u32 = 64;
const MAX_MATCHES: u32 = 64;
const MATCH_STRIDE: u32 = 32; // u32s per match slot

/// Compile WGSL source to SPIR-V words using naga directly.
/// This avoids the mysterious hang in wgpu's create_compute_pipeline
/// when processing large WGSL shaders.
fn compile_wgsl_to_spirv(wgsl_source: &str, label: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let module = naga::front::wgsl::parse_str(wgsl_source)
        .map_err(|e| format!("WGSL parse error in {label}: {e}"))?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("WGSL validation error in {label}: {e}"))?;

    let mut spv = Vec::new();
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    let mut writer = naga::back::spv::Writer::new(&options).map_err(|e| format!("SPIR-V writer init error: {e}"))?;
    writer
        .write(&module, &info, Some(&pipeline_options), &None, &mut spv)
        .map_err(|e| format!("SPIR-V codegen error in {label}: {e}"))?;

    Ok(spv)
}

pub struct WgpuBackend;

/// Compose unified mining shader by concatenating all module files.
fn compose_mine_pass_shader() -> String {
    let field = include_str!("../../wgpu/shaders/field.wgsl");
    let curve = include_str!("../../wgpu/shaders/curve.wgsl");
    let scalar = include_str!("../../wgpu/shaders/scalar.wgsl");
    let sha256 = include_str!("../../wgpu/shaders/sha256.wgsl");
    let hmac = include_str!("../../wgpu/shaders/hmac_drbg.wgsl");
    let encoding = include_str!("../../wgpu/shaders/encoding.wgsl");
    let pattern = include_str!("../../wgpu/shaders/pattern.wgsl");
    let mine_pass = include_str!("../../wgpu/shaders/mine_pass.wgsl");
    format!("{field}\n{curve}\n{scalar}\n{sha256}\n{hmac}\n{encoding}\n{pattern}\n{mine_pass}")
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
        let total_threads = TOTAL_THREADS;
        let num_workgroups = NUM_WORKGROUPS;

        // 1. Init wgpu
        eprintln!("[wgpu] initializing...");
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("plcpick"),
                required_features: wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            },
        ))?;

        // 2. Build CBOR template
        let tmpl = CborTemplate::new(&config.handle, &config.pds);
        let unsigned_tmpl_u32 = template_to_u32_array(&tmpl.unsigned_bytes);
        let signed_tmpl_u32 = template_to_u32_array(&tmpl.signed_bytes);
        let pattern_u32: Vec<u32> = config.pattern.iter().map(|&b| b as u32).collect();

        // Pack all_templates: [unsigned... | signed... | pattern...]
        let unsigned_tmpl_offset: u32 = 0;
        let signed_tmpl_offset: u32 = unsigned_tmpl_u32.len() as u32;
        let pattern_offset: u32 = signed_tmpl_offset + signed_tmpl_u32.len() as u32;
        let mut all_templates: Vec<u32> = Vec::with_capacity(
            unsigned_tmpl_u32.len() + signed_tmpl_u32.len() + pattern_u32.len(),
        );
        all_templates.extend_from_slice(&unsigned_tmpl_u32);
        all_templates.extend_from_slice(&signed_tmpl_u32);
        all_templates.extend_from_slice(&pattern_u32);

        // 3. Compute G table on CPU
        let (g_table_x, g_table_y) = compute_g_table();

        // 4. Compile unified shader
        eprintln!("[wgpu] compiling shader...");
        let shader_src = compose_mine_pass_shader();
        let spv = compile_wgsl_to_spirv(&shader_src, "mine_pass")?;
        eprintln!("[wgpu] SPIR-V compiled ({} words)", spv.len());

        eprintln!("[wgpu] creating shader module via passthrough...");
        let t = Instant::now();
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("mine_pass"),
                spirv: Some(std::borrow::Cow::Borrowed(&spv)),
                ..Default::default()
            })
        };
        eprintln!("[wgpu] module created in {:?}", t.elapsed());

        // 5. Bind group layout: 8 storage + 1 uniform
        use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};

        let storage_ro = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // mine_pass bindings: 0=rw(scalars), 1=rw(pubkeys), 2=ro(g_table_x), 3=ro(g_table_y),
        //   4=ro(stride_g_xy), 5=ro(all_templates), 6=rw(matches), 7=rw(match_count), 8=uniform(params)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mine_bind_group_layout"),
            entries: &[
                storage_rw(0), storage_rw(1), storage_ro(2), storage_ro(3),
                storage_ro(4), storage_ro(5), storage_rw(6), storage_rw(7),
                uniform_entry(8),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mine_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        // 6. Create compute pipeline
        eprintln!("[wgpu] creating pipeline...");
        let t = Instant::now();
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mine_pass_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });
        eprintln!("[wgpu] pipeline created in {:?}", t.elapsed());

        // 7. Generate initial scalars
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

        // 8. Create GPU buffers
        let scalars_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scalars"),
            contents: bytemuck::cast_slice(&scalar_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Pubkeys buffer: 24 u32 per thread (JacobianPoint: x[8] + y[8] + z[8])
        let pubkeys_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pubkeys"),
            size: (total_threads as u64) * 24 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
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

        // Stride G: stride * G where stride = total_threads
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

        let all_templates_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_templates"),
            contents: bytemuck::cast_slice(&all_templates),
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

        // MineParams uniform (matches WGSL MineParams struct exactly)
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct MineParams {
            num_threads: u32,
            iterations_per_thread: u32,
            is_first_launch: u32,
            pattern_len: u32,
            unsigned_tmpl_offset: u32,
            unsigned_tmpl_len: u32,
            signed_tmpl_offset: u32,
            signed_tmpl_len: u32,
            pattern_offset: u32,
            max_matches: u32,
            unsigned_pk_off1: u32,
            unsigned_pk_off2: u32,
            signed_pk_off1: u32,
            signed_pk_off2: u32,
            signed_sig_offset: u32,
            _pad: u32,
        }

        let mine_params = MineParams {
            num_threads: total_threads,
            iterations_per_thread: ITERATIONS_PER_THREAD,
            is_first_launch: 1,
            pattern_len: config.pattern.len() as u32,
            unsigned_tmpl_offset: unsigned_tmpl_offset,
            unsigned_tmpl_len: tmpl.unsigned_bytes.len() as u32,
            signed_tmpl_offset: signed_tmpl_offset,
            signed_tmpl_len: tmpl.signed_bytes.len() as u32,
            pattern_offset: pattern_offset,
            max_matches: MAX_MATCHES,
            unsigned_pk_off1: tmpl.unsigned_pubkey_offsets[0] as u32,
            unsigned_pk_off2: tmpl.unsigned_pubkey_offsets[1] as u32,
            signed_pk_off1: tmpl.signed_pubkey_offsets[0] as u32,
            signed_pk_off2: tmpl.signed_pubkey_offsets[1] as u32,
            signed_sig_offset: tmpl.signed_sig_offset as u32,
            _pad: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mine_params"),
            contents: bytemuck::bytes_of(&mine_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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

        // 9. Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mine_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scalars_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pubkeys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: g_table_x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: g_table_y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: stride_g_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: all_templates_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: match_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: match_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: params_buf.as_entire_binding() },
            ],
        });

        let keys_per_dispatch = total_threads as u64 * ITERATIONS_PER_THREAD as u64;
        eprintln!(
            "[wgpu] setup complete, entering mining loop ({} threads x {} iter = {} keys/dispatch)...",
            total_threads, ITERATIONS_PER_THREAD, keys_per_dispatch
        );

        // 10. Main mining loop
        let mut is_first = true;

        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }

            // Update is_first_launch flag after first dispatch
            if !is_first {
                let params = MineParams {
                    is_first_launch: 0,
                    ..mine_params
                };
                queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
            }

            // Reset match count
            queue.write_buffer(&match_count_buf, 0, &0u32.to_le_bytes());

            // Single dispatch — full pipeline in one pass
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mining_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mine_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Copy results to staging
            encoder.copy_buffer_to_buffer(&match_count_buf, 0, &match_count_staging, 0, 4);
            encoder.copy_buffer_to_buffer(&match_buf, 0, &match_staging, 0, match_buf_size);

            let submit_t = Instant::now();
            queue.submit(Some(encoder.finish()));

            // Read back match count
            let match_count_slice = match_count_staging.slice(..);
            match_count_slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            if is_first {
                eprintln!("[wgpu] first dispatch completed in {:?}", submit_t.elapsed());
            }

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
            total.fetch_add(keys_per_dispatch, Ordering::Relaxed);
            // No CPU scalar advancement needed — GPU handles it internally
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a wgpu device for testing. Returns None if no GPU available.
    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty()
                | wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            ..Default::default()
        });
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("request_adapter failed: {e}");
                return None;
            }
        };
        eprintln!("Adapter: {:?}", adapter.get_info().name);

        let features = wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS;
        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test"),
                required_features: features,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            },
        )) {
            Ok(dq) => dq,
            Err(e) => {
                eprintln!("request_device failed: {e}");
                return None;
            }
        };

        Some((device, queue))
    }

    /// Run a single-workgroup compute shader with an output buffer, return the output.
    fn run_gpu_shader(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str, output_size: u64) -> Vec<u32> {
        let spv = compile_wgsl_to_spirv(wgsl, "test_shader").expect("WGSL compile failed");

        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("test_shader"),
                spirv: Some(std::borrow::Cow::Borrowed(&spv)),
                ..Default::default()
            })
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        result
    }

    #[test]
    fn test_gpu_sha256() {
        let (device, queue) = match create_test_device() {
            Some(dq) => dq,
            None => { eprintln!("No GPU available, skipping"); return; }
        };

        let sha256_src = include_str!("../../wgpu/shaders/sha256.wgsl");
        let test_shader = format!(r#"
{sha256_src}

@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // Test 1: SHA256("abc") = ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad
    var data: array<u32, 512>;
    data[0] = 0x61u; // 'a'
    data[1] = 0x62u; // 'b'
    data[2] = 0x63u; // 'c'
    let hash = sha256_hash(&data, 3u);
    for (var i = 0u; i < 8u; i++) {{
        output[i] = hash[i];
    }}

    // Test 2: SHA256("") = e3b0c442 98fc1c14 9afbf4c8 996fb924 27ae41e4 649b934c a495991b 7852b855
    var empty: array<u32, 512>;
    let hash2 = sha256_hash(&empty, 0u);
    for (var i = 0u; i < 8u; i++) {{
        output[8u + i] = hash2[i];
    }}

    // Test 3: SHA256 of 64 bytes (exactly one block + padding block)
    // SHA256("aaaa...a" x 64) = ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb
    var data3: array<u32, 512>;
    for (var j = 0u; j < 64u; j++) {{
        data3[j] = 0x61u; // 'a'
    }}
    let hash3 = sha256_hash(&data3, 64u);
    for (var i = 0u; i < 8u; i++) {{
        output[16u + i] = hash3[i];
    }}
}}
"#);

        let result = run_gpu_shader(&device, &queue, &test_shader, 24 * 4);

        // SHA256("abc")
        let expected_abc: [u32; 8] = [
            0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223,
            0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad,
        ];
        let got_abc = &result[0..8];
        eprintln!("SHA256(\"abc\"):");
        eprintln!("  expected: {:08x?}", expected_abc);
        eprintln!("  got:      {:08x?}", got_abc);
        assert_eq!(got_abc, &expected_abc, "SHA256(\"abc\") mismatch");

        // SHA256("")
        let expected_empty: [u32; 8] = [
            0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924,
            0x27ae41e4, 0x649b934c, 0xa495991b, 0x7852b855,
        ];
        let got_empty = &result[8..16];
        eprintln!("SHA256(\"\"):");
        eprintln!("  expected: {:08x?}", expected_empty);
        eprintln!("  got:      {:08x?}", got_empty);
        assert_eq!(got_empty, &expected_empty, "SHA256(\"\") mismatch");

        // SHA256("a" * 64)
        let expected_64a: [u32; 8] = [
            0xffe054fe, 0x7ae0cb6d, 0xc65c3af9, 0xb61d5209,
            0xf439851d, 0xb43d0ba5, 0x997337df, 0x154668eb,
        ];
        let got_64a = &result[16..24];
        eprintln!("SHA256(\"a\"*64):");
        eprintln!("  expected: {:08x?}", expected_64a);
        eprintln!("  got:      {:08x?}", got_64a);
        assert_eq!(got_64a, &expected_64a, "SHA256(\"a\"*64) mismatch");
    }

    #[test]
    fn test_gpu_fe_mul() {
        let (device, queue) = match create_test_device() {
            Some(dq) => dq,
            None => { eprintln!("No GPU available, skipping"); return; }
        };

        let field_src = include_str!("../../wgpu/shaders/field.wgsl");
        let test_shader = format!(r#"
{field_src}

@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // Test 1: fe_mul(3, 7) = 21 (small values, no reduction needed)
    var a: array<u32, 8>;
    a[0] = 3u;
    var b: array<u32, 8>;
    b[0] = 7u;
    let c = fe_mul(a, b);
    for (var i = 0u; i < 8u; i++) {{
        output[i] = c[i];
    }}

    // Test 2: fe_square(2) = 4
    var two: array<u32, 8>;
    two[0] = 2u;
    let sq = fe_square(two);
    for (var i = 0u; i < 8u; i++) {{
        output[8u + i] = sq[i];
    }}

    // Test 3: fe_mul with large values
    // secp256k1 G.x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    // G.x in LE limbs:
    var gx: array<u32, 8>;
    gx[0] = 0x16F81798u; gx[1] = 0x59F2815Bu; gx[2] = 0x2DCE28D9u; gx[3] = 0x029BFCDBu;
    gx[4] = 0xCE870B07u; gx[5] = 0x55A06295u; gx[6] = 0xF9DCBBACu; gx[7] = 0x79BE667Eu;

    // fe_square(G.x) — compute on GPU, verify on CPU side
    let gx_sq = fe_square(gx);
    for (var i = 0u; i < 8u; i++) {{
        output[16u + i] = gx_sq[i];
    }}

    // Test 4: fe_mul(G.x, G.x) — should equal fe_square(G.x)
    let gx_mul = fe_mul(gx, gx);
    for (var i = 0u; i < 8u; i++) {{
        output[24u + i] = gx_mul[i];
    }}
}}
"#);

        let result = run_gpu_shader(&device, &queue, &test_shader, 32 * 4);

        // fe_mul(3, 7) = 21
        let got_mul = &result[0..8];
        eprintln!("fe_mul(3, 7):");
        eprintln!("  got: {:08x?}", got_mul);
        assert_eq!(got_mul[0], 21, "fe_mul(3,7) should be 21");
        for i in 1..8 { assert_eq!(got_mul[i], 0, "fe_mul(3,7) limb {} should be 0", i); }

        // fe_square(2) = 4
        let got_sq2 = &result[8..16];
        eprintln!("fe_square(2):");
        eprintln!("  got: {:08x?}", got_sq2);
        assert_eq!(got_sq2[0], 4, "fe_square(2) should be 4");
        for i in 1..8 { assert_eq!(got_sq2[i], 0, "fe_square(2) limb {} should be 0", i); }

        // fe_square(G.x) should equal fe_mul(G.x, G.x)
        let got_gx_sq = &result[16..24];
        let got_gx_mul = &result[24..32];
        eprintln!("fe_square(G.x):");
        eprintln!("  square: {:08x?}", got_gx_sq);
        eprintln!("  mul:    {:08x?}", got_gx_mul);
        assert_eq!(got_gx_sq, got_gx_mul, "fe_square(G.x) should equal fe_mul(G.x, G.x)");

        // Verify G.x² mod p against known value
        // G.x² mod p = python: pow(0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798, 2, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F)
        // = 0x34b6ab3ddb040b15...  (we'll verify on the Rust side)
        eprintln!("fe_square(G.x) result for manual verification: {:08x?}", got_gx_sq);
    }

    #[test]
    fn test_gpu_hmac_sha256() {
        let (device, queue) = match create_test_device() {
            Some(dq) => dq,
            None => { eprintln!("No GPU available, skipping"); return; }
        };

        let sha256_src = include_str!("../../wgpu/shaders/sha256.wgsl");
        let field_src = include_str!("../../wgpu/shaders/field.wgsl");
        let scalar_src = include_str!("../../wgpu/shaders/scalar.wgsl");
        let hmac_src = include_str!("../../wgpu/shaders/hmac_drbg.wgsl");
        let test_shader = format!(r#"
{field_src}
{scalar_src}
{sha256_src}
{hmac_src}

@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // RFC 4231 Test Case 2: HMAC-SHA256
    // Key = "Jefe" (4 bytes, zero-padded to 32)
    // Data = "what do ya want for nothing?" (28 bytes)
    // Expected: 5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843

    // Key as 8 u32 words (big-endian packed). "Jefe" = 0x4a656665, rest zero
    let key = array<u32, 8>(
        0x4a656665u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
    );

    // Data: "what do ya want for nothing?" = 28 bytes
    var msg: array<u32, 97>;
    let data_str = array<u32, 28>(
        0x77u, 0x68u, 0x61u, 0x74u, 0x20u, 0x64u, 0x6fu, 0x20u,
        0x79u, 0x61u, 0x20u, 0x77u, 0x61u, 0x6eu, 0x74u, 0x20u,
        0x66u, 0x6fu, 0x72u, 0x20u, 0x6eu, 0x6fu, 0x74u, 0x68u,
        0x69u, 0x6eu, 0x67u, 0x3fu
    );
    for (var i = 0u; i < 28u; i++) {{
        msg[i] = data_str[i];
    }}
    for (var i = 28u; i < 97u; i++) {{
        msg[i] = 0u;
    }}

    let result = hmac_sha256(key, &msg, 28u);
    for (var i = 0u; i < 8u; i++) {{
        output[i] = result[i];
    }}

    // Test 2: HMAC with all-zero key and short message (like RFC 6979 step d)
    // Key = 0x00 * 32, Message = 0x01*32 || 0x00 || <32 bytes priv> || <32 bytes hash>
    // Use simpler version: HMAC(zero_key, V=0x01*32) — msg_len = 32
    let zero_key = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var msg2: array<u32, 97>;
    for (var i = 0u; i < 32u; i++) {{
        msg2[i] = 0x01u;
    }}
    for (var i = 32u; i < 97u; i++) {{
        msg2[i] = 0u;
    }}
    let result2 = hmac_sha256(zero_key, &msg2, 32u);
    for (var i = 0u; i < 8u; i++) {{
        output[8u + i] = result2[i];
    }}
}}
"#);

        let result = run_gpu_shader(&device, &queue, &test_shader, 16 * 4);

        // RFC 4231 Test Case 2
        let expected_hmac: [u32; 8] = [
            0x5bdcc146, 0xbf60754e, 0x6a042426, 0x089575c7,
            0x5a003f08, 0x9d273983, 0x9dec58b9, 0x64ec3843,
        ];
        let got_hmac = &result[0..8];
        eprintln!("HMAC-SHA256 (RFC 4231 TC2):");
        eprintln!("  expected: {:08x?}", expected_hmac);
        eprintln!("  got:      {:08x?}", got_hmac);
        assert_eq!(got_hmac, &expected_hmac, "HMAC-SHA256 RFC 4231 TC2 mismatch");

        // HMAC(zero_key, 0x01*32)
        let got_hmac2 = &result[8..16];
        eprintln!("HMAC-SHA256 (zero_key, 0x01*32):");
        eprintln!("  got: {:08x?}", got_hmac2);
        // Verify it's not all zeros
        let all_zero = got_hmac2.iter().all(|&v| v == 0);
        assert!(!all_zero, "HMAC should not return all-zero");
    }

    #[test]
    fn test_gpu_scalar_mul_g() {
        let (device, queue) = match create_test_device() {
            Some(dq) => dq,
            None => { eprintln!("No GPU available, skipping"); return; }
        };

        // Compute G table on CPU
        let (g_table_x_data, g_table_y_data) = compute_g_table();

        let field_src = include_str!("../../wgpu/shaders/field.wgsl");
        let curve_src = include_str!("../../wgpu/shaders/curve.wgsl");
        let test_shader = format!(r#"
{field_src}
{curve_src}

@group(0) @binding(0) var<storage, read> g_table_x: array<u32>;
@group(0) @binding(1) var<storage, read> g_table_y: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

// scalar_mul_g_windowed — same as mine_pass.wgsl
fn scalar_mul_g_windowed(scalar: array<u32, 8>) -> JacobianPoint {{
    var result = jac_infinity();
    for (var i = 63i; i >= 0; i--) {{
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);
        let limb_idx = u32(i) / 8u;
        let nibble_idx = u32(i) % 8u;
        let nibble = (scalar[limb_idx] >> (nibble_idx * 4u)) & 0xFu;
        if (nibble != 0u) {{
            let table_offset = (nibble - 1u) * 8u;
            var qx: array<u32, 8>;
            var qy: array<u32, 8>;
            for (var j = 0u; j < 8u; j++) {{
                qx[j] = g_table_x[table_offset + j];
                qy[j] = g_table_y[table_offset + j];
            }}
            result = jac_add_affine(result, qx, qy);
        }}
    }}
    return result;
}}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // Test 1: scalar = 1 → should give G
    var s1: array<u32, 8>;
    s1[0] = 1u;
    let p1 = scalar_mul_g_windowed(s1);
    let a1 = jac_to_affine(p1);
    for (var i = 0u; i < 8u; i++) {{
        output[i] = a1.x[i];
        output[8u + i] = a1.y[i];
    }}

    // Test 2: scalar = 2 → should give 2G
    var s2: array<u32, 8>;
    s2[0] = 2u;
    let p2 = scalar_mul_g_windowed(s2);
    let a2 = jac_to_affine(p2);
    for (var i = 0u; i < 8u; i++) {{
        output[16u + i] = a2.x[i];
        output[24u + i] = a2.y[i];
    }}

    // Test 3: a specific scalar (e.g. 0xDEADBEEF)
    var s3: array<u32, 8>;
    s3[0] = 0xDEADBEEFu;
    let p3 = scalar_mul_g_windowed(s3);
    let a3 = jac_to_affine(p3);
    for (var i = 0u; i < 8u; i++) {{
        output[32u + i] = a3.x[i];
        output[40u + i] = a3.y[i];
    }}
}}
"#);

        let spv = compile_wgsl_to_spirv(&test_shader, "test_scalarmul").expect("compile");

        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("test_scalarmul"),
                spirv: Some(std::borrow::Cow::Borrowed(&spv)),
                ..Default::default()
            })
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gtx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&g_table_x_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let gty_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&g_table_y_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size: u64 = 48 * 4; // 6 arrays of 8 u32
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gtx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gty_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        queue.submit(Some(enc.finish()));

        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let result: Vec<u32> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        staging_buf.unmap();

        // Compute expected values on CPU using k256
        // 1*G
        let g = ProjectivePoint::GENERATOR;
        let g_aff = g.to_affine();
        let g_enc = g_aff.to_encoded_point(false);
        let expected_gx = point_x_to_limbs(g_enc.x().unwrap().as_slice());
        let expected_gy = point_x_to_limbs(g_enc.y().unwrap().as_slice());

        let got_gx = &result[0..8];
        let got_gy = &result[8..16];
        eprintln!("1*G x:");
        eprintln!("  expected: {:08x?}", expected_gx);
        eprintln!("  got:      {:08x?}", got_gx);
        eprintln!("1*G y:");
        eprintln!("  expected: {:08x?}", expected_gy);
        eprintln!("  got:      {:08x?}", got_gy);
        assert_eq!(got_gx, &expected_gx, "1*G x mismatch");
        assert_eq!(got_gy, &expected_gy, "1*G y mismatch");

        // 2*G
        let g2 = (g + g).to_affine();
        let g2_enc = g2.to_encoded_point(false);
        let expected_2gx = point_x_to_limbs(g2_enc.x().unwrap().as_slice());
        let expected_2gy = point_x_to_limbs(g2_enc.y().unwrap().as_slice());

        let got_2gx = &result[16..24];
        let got_2gy = &result[24..32];
        eprintln!("2*G x:");
        eprintln!("  expected: {:08x?}", expected_2gx);
        eprintln!("  got:      {:08x?}", got_2gx);
        assert_eq!(got_2gx, &expected_2gx, "2*G x mismatch");
        assert_eq!(got_2gy, &expected_2gy, "2*G y mismatch");

        // 0xDEADBEEF * G
        let s3 = Scalar::from(0xDEADBEEFu64);
        let g3 = (g * s3).to_affine();
        let g3_enc = g3.to_encoded_point(false);
        let expected_3x = point_x_to_limbs(g3_enc.x().unwrap().as_slice());
        let expected_3y = point_x_to_limbs(g3_enc.y().unwrap().as_slice());

        let got_3x = &result[32..40];
        let got_3y = &result[40..48];
        eprintln!("0xDEADBEEF*G x:");
        eprintln!("  expected: {:08x?}", expected_3x);
        eprintln!("  got:      {:08x?}", got_3x);
        assert_eq!(got_3x, &expected_3x, "0xDEADBEEF*G x mismatch");
        assert_eq!(got_3y, &expected_3y, "0xDEADBEEF*G y mismatch");
    }

    fn point_x_to_limbs(bytes: &[u8]) -> [u32; 8] {
        let mut limbs = [0u32; 8];
        for i in 0..8 {
            let offset = (7 - i) * 4;
            limbs[i] = u32::from_be_bytes([
                bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
            ]);
        }
        limbs
    }

    /// Convert 32 big-endian bytes to 8 LE u32 limbs (same as WGSL hmac_load_scalar_bytes)
    fn be_bytes_to_le_limbs(bytes: &[u8; 32]) -> [u32; 8] {
        let mut limbs = [0u32; 8];
        for i in 0..8 {
            let offset = (7 - i) * 4;
            limbs[i] = u32::from_be_bytes([
                bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
            ]);
        }
        limbs
    }

    #[test]
    fn test_gpu_ecdsa_sign() {
        use k256::ecdsa::{SigningKey, signature::hazmat::PrehashSigner};

        let (device, queue) = match create_test_device() {
            Some(dq) => dq,
            None => { eprintln!("No GPU available, skipping"); return; }
        };

        // Compute G table on CPU
        let (g_table_x_data, g_table_y_data) = compute_g_table();

        // Known private key (32 bytes, big-endian)
        let privkey_be: [u8; 32] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
        ];

        // Known message hash (32 bytes, big-endian) - just use SHA256("test")
        let msg_hash_be: [u8; 32] = {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(b"test");
            let result = hasher.finalize();
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&result);
            arr
        };

        // Convert to LE limbs for the shader
        let privkey_limbs = be_bytes_to_le_limbs(&privkey_be);
        let hash_limbs = be_bytes_to_le_limbs(&msg_hash_be);

        // CPU reference: sign with k256
        let signing_key = SigningKey::from_bytes((&privkey_be).into()).expect("valid key");
        let (cpu_sig, _recid) = signing_key.sign_prehash(&msg_hash_be).expect("sign");
        let cpu_sig_bytes = cpu_sig.to_bytes();
        let cpu_r_be: [u8; 32] = cpu_sig_bytes[0..32].try_into().unwrap();
        let cpu_s_be: [u8; 32] = cpu_sig_bytes[32..64].try_into().unwrap();
        let cpu_r_limbs = be_bytes_to_le_limbs(&cpu_r_be);
        let cpu_s_limbs = be_bytes_to_le_limbs(&cpu_s_be);

        eprintln!("CPU r: {:08x?}", cpu_r_limbs);
        eprintln!("CPU s: {:08x?}", cpu_s_limbs);

        // Build the test shader
        let field_src = include_str!("../../wgpu/shaders/field.wgsl");
        let curve_src = include_str!("../../wgpu/shaders/curve.wgsl");
        let scalar_src = include_str!("../../wgpu/shaders/scalar.wgsl");
        let sha256_src = include_str!("../../wgpu/shaders/sha256.wgsl");
        let hmac_src = include_str!("../../wgpu/shaders/hmac_drbg.wgsl");

        // Format limbs as WGSL array literal
        let fmt_limbs = |l: &[u32; 8]| -> String {
            format!("array<u32, 8>({}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u)",
                l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7])
        };

        let test_shader = format!(r#"
{field_src}
{curve_src}
{scalar_src}
{sha256_src}
{hmac_src}

@group(0) @binding(0) var<storage, read> g_table_x: array<u32>;
@group(0) @binding(1) var<storage, read> g_table_y: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn scalar_mul_g_windowed(scalar: array<u32, 8>) -> JacobianPoint {{
    var result = jac_infinity();
    for (var i = 63i; i >= 0; i--) {{
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);
        let limb_idx = u32(i) / 8u;
        let nibble_idx = u32(i) % 8u;
        let nibble = (scalar[limb_idx] >> (nibble_idx * 4u)) & 0xFu;
        if (nibble != 0u) {{
            let table_offset = (nibble - 1u) * 8u;
            var qx: array<u32, 8>;
            var qy: array<u32, 8>;
            for (var j = 0u; j < 8u; j++) {{
                qx[j] = g_table_x[table_offset + j];
                qy[j] = g_table_y[table_offset + j];
            }}
            result = jac_add_affine(result, qx, qy);
        }}
    }}
    return result;
}}

const SECP_N_HALF = array<u32, 8>(
    0x681B20A0u, 0xDFE92F46u, 0x57A4501Du, 0x5D576E73u,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu
);

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let privkey = {pk};
    let hash_scalar = {hs};

    // Output[0..8]: nonce_k
    let nonce_k = hmac_rfc6979_nonce(privkey, hash_scalar);
    for (var i = 0u; i < 8u; i++) {{ output[i] = nonce_k[i]; }}

    // Output[8..16]: R.x (affine)
    let R = scalar_mul_g_windowed(nonce_k);
    let R_aff = jac_to_affine(R);
    for (var i = 0u; i < 8u; i++) {{ output[8u + i] = R_aff.x[i]; }}

    // r = R.x mod n
    var r_val = R_aff.x;
    if (scalar_cmp(r_val, SECP_N) >= 0) {{
        r_val = scalar_sub_internal(r_val, SECP_N);
    }}

    // s = k_inv * (hash + r * privkey) mod n
    let r_times_priv = scalar_mod_n_mul(r_val, privkey);
    let hash_plus_rpriv = scalar_mod_n_add(hash_scalar, r_times_priv);
    let k_inv = scalar_mod_n_inv(nonce_k);
    var s_val = scalar_mod_n_mul(k_inv, hash_plus_rpriv);

    // Low-s normalization (BIP-62)
    if (scalar_cmp(s_val, SECP_N_HALF) > 0) {{
        s_val = scalar_mod_n_sub(SECP_N, s_val);
    }}

    // Output[16..24]: r
    for (var i = 0u; i < 8u; i++) {{ output[16u + i] = r_val[i]; }}
    // Output[24..32]: s
    for (var i = 0u; i < 8u; i++) {{ output[24u + i] = s_val[i]; }}

    // Output[32..40]: r_times_priv (debug)
    for (var i = 0u; i < 8u; i++) {{ output[32u + i] = r_times_priv[i]; }}
    // Output[40..48]: hash_plus_rpriv (debug)
    for (var i = 0u; i < 8u; i++) {{ output[40u + i] = hash_plus_rpriv[i]; }}
    // Output[48..56]: k_inv (debug)
    for (var i = 0u; i < 8u; i++) {{ output[48u + i] = k_inv[i]; }}
}}
"#,
            pk = fmt_limbs(&privkey_limbs),
            hs = fmt_limbs(&hash_limbs),
        );

        let spv = compile_wgsl_to_spirv(&test_shader, "test_ecdsa").expect("compile");

        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("test_ecdsa"),
                spirv: Some(std::borrow::Cow::Borrowed(&spv)),
                ..Default::default()
            })
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gtx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&g_table_x_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let gty_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&g_table_y_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size: u64 = 56 * 4; // 7 arrays of 8 u32
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gtx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gty_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        queue.submit(Some(enc.finish()));

        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let result: Vec<u32> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        staging_buf.unmap();

        let gpu_nonce = &result[0..8];
        let gpu_rx = &result[8..16];
        let gpu_r = &result[16..24];
        let gpu_s = &result[24..32];
        let gpu_rpriv = &result[32..40];
        let gpu_hash_rpriv = &result[40..48];
        let gpu_kinv = &result[48..56];

        eprintln!("GPU nonce_k:       {:08x?}", gpu_nonce);
        eprintln!("GPU R.x:           {:08x?}", gpu_rx);
        eprintln!("GPU r:             {:08x?}", gpu_r);
        eprintln!("GPU s:             {:08x?}", gpu_s);
        eprintln!("GPU r*priv:        {:08x?}", gpu_rpriv);
        eprintln!("GPU hash+r*priv:   {:08x?}", gpu_hash_rpriv);
        eprintln!("GPU k_inv:         {:08x?}", gpu_kinv);
        eprintln!("CPU r:             {:08x?}", cpu_r_limbs);
        eprintln!("CPU s:             {:08x?}", cpu_s_limbs);

        assert_eq!(gpu_r, &cpu_r_limbs, "r mismatch");
        assert_eq!(gpu_s, &cpu_s_limbs, "s mismatch");
    }

    #[test]
    fn test_gpu_scalar_mod_n_mul() {
        let (device, queue) = match create_test_device() {
            Some(dq) => dq,
            None => { eprintln!("No GPU available, skipping"); return; }
        };

        let field_src = include_str!("../../wgpu/shaders/field.wgsl");
        let scalar_src = include_str!("../../wgpu/shaders/scalar.wgsl");
        let test_shader = format!(r#"
{field_src}
{scalar_src}

@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // Test 1: 3 * 7 mod n = 21
    var a1 = array<u32, 8>(3u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var b1 = array<u32, 8>(7u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    let r1 = scalar_mod_n_mul(a1, b1);
    for (var i = 0u; i < 8u; i++) {{ output[i] = r1[i]; }}

    // Test 2: (n-1) * (n-1) mod n = 1
    var nm1 = array<u32, 8>(0xD0364140u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    let r2 = scalar_mod_n_mul(nm1, nm1);
    for (var i = 0u; i < 8u; i++) {{ output[8u + i] = r2[i]; }}

    // Test 3: scalar_mod_n_inv(k) * k mod n = 1
    // Use the known nonce_k from ECDSA test
    var k = array<u32, 8>(0xb29d7774u, 0x3e479ca6u, 0xb54519f0u, 0x9f53738au, 0x516c5f21u, 0xafa0a612u, 0x4298bf55u, 0xb8f5683au);
    let kinv = scalar_mod_n_inv(k);
    let r3 = scalar_mod_n_mul(kinv, k);
    for (var i = 0u; i < 8u; i++) {{ output[16u + i] = kinv[i]; }}
    for (var i = 0u; i < 8u; i++) {{ output[24u + i] = r3[i]; }}

    // Test 4: scalar_mod_n_add and scalar_mod_n_mul individually
    // hash_scalar and r * priv separately
    let hash_s = array<u32, 8>(0xb0f00a08u, 0xd15d6c15u, 0x2b0b822cu, 0xa3bf4f1bu, 0xc55ad015u, 0x9a2feaa0u, 0x884c7d65u, 0x9f86d081u);
    let r_val = array<u32, 8>(0x193e6d54u, 0x21aac2a2u, 0x27662ae9u, 0x61039685u, 0xe5eee683u, 0xd9dd4aa5u, 0x34d12dd1u, 0x47d866f4u);
    let priv_key = array<u32, 8>(0x1C1D1E1Fu, 0x18191A1Bu, 0x14151617u, 0x10111213u, 0x0C0D0E0Fu, 0x08090A0Bu, 0x04050607u, 0x00010203u);
    let rpriv = scalar_mod_n_mul(r_val, priv_key);
    let hpr = scalar_mod_n_add(hash_s, rpriv);
    for (var i = 0u; i < 8u; i++) {{ output[32u + i] = rpriv[i]; }}
    for (var i = 0u; i < 8u; i++) {{ output[40u + i] = hpr[i]; }}
}}
"#);

        let result = run_gpu_shader(&device, &queue, &test_shader, 48 * 4);

        // Test 1: 3 * 7 = 21
        let r1 = &result[0..8];
        eprintln!("3*7 mod n: {:08x?}", r1);
        assert_eq!(r1[0], 21, "3*7 should be 21");
        assert!(r1[1..8].iter().all(|&x| x == 0), "upper limbs should be 0");

        // Test 2: (n-1)*(n-1) mod n = 1
        let r2 = &result[8..16];
        eprintln!("(n-1)^2 mod n: {:08x?}", r2);
        assert_eq!(r2[0], 1, "(n-1)^2 mod n should be 1");
        assert!(r2[1..8].iter().all(|&x| x == 0), "upper limbs should be 0");

        // Test 3: kinv * k mod n = 1
        let kinv = &result[16..24];
        let kinv_times_k = &result[24..32];
        eprintln!("k_inv: {:08x?}", kinv);
        eprintln!("k_inv*k mod n: {:08x?}", kinv_times_k);
        assert_eq!(kinv_times_k[0], 1, "k_inv * k mod n should be 1");
        assert!(kinv_times_k[1..8].iter().all(|&x| x == 0), "upper limbs should be 0");

        // Test 4: r*priv and hash+r*priv
        let rpriv = &result[32..40];
        let hpr = &result[40..48];
        eprintln!("r*priv mod n: {:08x?}", rpriv);
        eprintln!("hash+r*priv mod n: {:08x?}", hpr);
    }
}
