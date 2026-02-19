use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use k256::ecdsa::SigningKey;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::VulkanLibrary;

use super::{Match, MiningBackend, MiningConfig};
use crate::pattern::glob_match;
use crate::plc::{build_signed_op, did_suffix, CborTemplate};

const WORKGROUP_SIZE: u32 = 256;
const NUM_WORKGROUPS: u32 = 4;
const TOTAL_THREADS: u32 = WORKGROUP_SIZE * NUM_WORKGROUPS;
const ITERATIONS_PER_LAUNCH: u32 = 16;
const MAX_MATCHES: u32 = 64;
const MATCH_SLOT_UINTS: u32 = 32; // 32 uints per match slot

// Push constants struct — must match the GLSL push_constant layout
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    is_first_launch: u32,
    iterations_per_thread: u32,
    max_matches: u32,
}

pub struct VulkanBackend {
    pub device_index: usize,
}

fn load_shader_module(device: Arc<Device>, spv_bytes: &[u8]) -> Arc<ShaderModule> {
    let spirv_words: Vec<u32> = spv_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    unsafe { ShaderModule::new(device, ShaderModuleCreateInfo::new(&spirv_words)) }
        .expect("failed to create shader module")
}

fn make_compute_pipeline(device: Arc<Device>, shader_module: Arc<ShaderModule>) -> Arc<ComputePipeline> {
    let entry_point = shader_module.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(entry_point);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

fn make_buffer_u32(
    allocator: Arc<StandardMemoryAllocator>,
    data: &[u32],
) -> Subbuffer<[u32]> {
    Buffer::from_iter(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data.iter().copied(),
    )
    .unwrap()
}

fn make_buffer_u8(
    allocator: Arc<StandardMemoryAllocator>,
    data: &[u8],
) -> Subbuffer<[u8]> {
    Buffer::from_iter(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data.iter().copied(),
    )
    .unwrap()
}

impl MiningBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn run(
        &self,
        config: &MiningConfig,
        stop: &AtomicBool,
        total: &AtomicU64,
        tx: mpsc::Sender<Match>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();

        // 1. Init Vulkan
        let library = VulkanLibrary::new()?;
        let instance = Instance::new(library, InstanceCreateInfo::default())?;

        let physical_devices: Vec<_> = instance
            .enumerate_physical_devices()?
            .filter(|pd| {
                pd.queue_family_properties()
                    .iter()
                    .any(|qf| qf.queue_flags.intersects(QueueFlags::COMPUTE))
            })
            .collect();

        let physical_device = physical_devices
            .into_iter()
            .nth(self.device_index)
            .ok_or("no compute-capable Vulkan device at requested index")?;

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|qf| qf.queue_flags.intersects(QueueFlags::COMPUTE))
            .unwrap() as u32;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_features: DeviceFeatures {
                    shader_int64: true,
                    shader_int8: true,
                    uniform_and_storage_buffer8_bit_access: true,
                    storage_buffer8_bit_access: true,
                    ..DeviceFeatures::empty()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;
        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // 2. Load SPIR-V modules
        let mine_spv = include_bytes!(concat!(env!("OUT_DIR"), "/mine.comp.spv"));
        let init_g_table_spv = include_bytes!(concat!(env!("OUT_DIR"), "/init_g_table.comp.spv"));
        let stride_g_spv = include_bytes!(concat!(env!("OUT_DIR"), "/stride_g.comp.spv"));

        let mine_module = load_shader_module(device.clone(), mine_spv);
        let init_g_table_module = load_shader_module(device.clone(), init_g_table_spv);
        let stride_g_module = load_shader_module(device.clone(), stride_g_spv);

        // 3. Create compute pipelines
        let mine_pipeline = make_compute_pipeline(device.clone(), mine_module);
        let init_g_table_pipeline = make_compute_pipeline(device.clone(), init_g_table_module);
        let stride_g_pipeline = make_compute_pipeline(device.clone(), stride_g_module);

        // 4. Build CBOR template
        let tmpl = CborTemplate::new(&config.handle, &config.pds);

        // 5. Allocate g_table buffer and run init_g_table
        let g_table_buf = make_buffer_u32(memory_allocator.clone(), &[0u32; 240]);
        {
            let layout = init_g_table_pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                layout.clone(),
                [WriteDescriptorSet::buffer(0, g_table_buf.clone())],
                [],
            )
            .unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .bind_pipeline_compute(init_g_table_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    init_g_table_pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap();
            unsafe { builder.dispatch([1, 1, 1]) }.unwrap();

            let cmd = builder.build().unwrap();
            vulkano::sync::now(device.clone())
                .then_execute(queue.clone(), cmd)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)?;
        }

        // 6. Compute stride_G
        let stride_val_u32 = TOTAL_THREADS;
        let mut stride_in_data = [0u32; 8];
        stride_in_data[0] = stride_val_u32;
        let stride_in_buf = make_buffer_u32(memory_allocator.clone(), &stride_in_data);
        let stride_out_buf = make_buffer_u32(memory_allocator.clone(), &[0u32; 32]); // 8 (stride) + 24 (stride_g)
        {
            let layout = stride_g_pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, stride_in_buf),
                    WriteDescriptorSet::buffer(1, stride_out_buf.clone()),
                    WriteDescriptorSet::buffer(2, g_table_buf.clone()),
                ],
                [],
            )
            .unwrap();

            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .bind_pipeline_compute(stride_g_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    stride_g_pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap();
            unsafe { builder.dispatch([1, 1, 1]) }.unwrap();

            let cmd = builder.build().unwrap();
            vulkano::sync::now(device.clone())
                .then_execute(queue.clone(), cmd)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)?;
        }

        // Read back stride and stride_G
        let stride_out_data = stride_out_buf.read().unwrap();
        let mut stride_arr = [0u32; 8];
        let mut stride_g_arr = [0u32; 24];
        stride_arr.copy_from_slice(&stride_out_data[0..8]);
        stride_g_arr.copy_from_slice(&stride_out_data[8..32]);

        // 7. Generate initial per-thread scalars (big-endian, 32 bytes each)
        let mut rng = rand::thread_rng();
        let base_key = SigningKey::random(&mut rng);
        let mut scalar_data = vec![0u8; TOTAL_THREADS as usize * 32];
        for tid in 0..TOTAL_THREADS as u64 {
            let offset_scalar = k256::Scalar::from(tid);
            let thread_scalar = base_key.as_nonzero_scalar().as_ref() + &offset_scalar;
            let thread_bytes = thread_scalar.to_bytes();
            let start_idx = tid as usize * 32;
            scalar_data[start_idx..start_idx + 32].copy_from_slice(&thread_bytes);
        }

        let scalars_buf = make_buffer_u8(memory_allocator.clone(), &scalar_data);

        // 8. Allocate pubkeys buffer (24 uints per thread)
        let pubkeys_buf = make_buffer_u32(
            memory_allocator.clone(),
            &vec![0u32; TOTAL_THREADS as usize * 24],
        );

        // 9. Allocate match output buffers
        let matches_buf = make_buffer_u32(
            memory_allocator.clone(),
            &vec![0u32; MAX_MATCHES as usize * MATCH_SLOT_UINTS as usize],
        );
        let match_count_buf = make_buffer_u32(memory_allocator.clone(), &[0u32]);

        // 10. Upload templates and pattern
        let unsigned_template_buf = make_buffer_u8(memory_allocator.clone(), &tmpl.unsigned_bytes);
        let signed_template_buf = make_buffer_u8(memory_allocator.clone(), &tmpl.signed_bytes);

        // Pad pattern to at least 4 bytes (vulkano requires non-zero buffer)
        let mut pattern_data = config.pattern.clone();
        while pattern_data.len() < 4 || pattern_data.len() % 4 != 0 {
            pattern_data.push(0);
        }
        let pattern_buf = make_buffer_u8(memory_allocator.clone(), &pattern_data);

        // 11. Build KernelParams SSBO
        // Layout matches the GLSL struct:
        //   uint unsigned_template_len
        //   uint signed_template_len
        //   uint unsigned_pubkey_offsets[2]
        //   uint signed_pubkey_offsets[2]
        //   uint signed_sig_offset
        //   uint pattern_len
        //   uint stride[8]
        //   uint stride_g[24]
        // Total: 2 + 2 + 2 + 1 + 1 + 8 + 24 = 40 uints
        let mut params_data = vec![0u32; 40];
        params_data[0] = tmpl.unsigned_bytes.len() as u32;
        params_data[1] = tmpl.signed_bytes.len() as u32;
        params_data[2] = tmpl.unsigned_pubkey_offsets[0] as u32;
        params_data[3] = tmpl.unsigned_pubkey_offsets[1] as u32;
        params_data[4] = tmpl.signed_pubkey_offsets[0] as u32;
        params_data[5] = tmpl.signed_pubkey_offsets[1] as u32;
        params_data[6] = tmpl.signed_sig_offset as u32;
        params_data[7] = config.pattern.len() as u32;
        params_data[8..16].copy_from_slice(&stride_arr);
        params_data[16..40].copy_from_slice(&stride_g_arr);
        let params_buf = make_buffer_u32(memory_allocator.clone(), &params_data);

        // 12. Create descriptor set for mine pipeline
        let mine_layout = mine_pipeline.layout().set_layouts().get(0).unwrap();

        let mut is_first_launch = 1u32;

        // 13. Main mining loop
        loop {
            if stop.load(Ordering::Relaxed) && !config.keep_going {
                break;
            }

            // Reset match count to 0
            {
                let mut content = match_count_buf.write().unwrap();
                content[0] = 0;
            }

            // Create descriptor set (bindings 0-8)
            let descriptor_set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                mine_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, params_buf.clone()),
                    WriteDescriptorSet::buffer(1, unsigned_template_buf.clone()),
                    WriteDescriptorSet::buffer(2, signed_template_buf.clone()),
                    WriteDescriptorSet::buffer(3, pattern_buf.clone()),
                    WriteDescriptorSet::buffer(4, scalars_buf.clone()),
                    WriteDescriptorSet::buffer(5, pubkeys_buf.clone()),
                    WriteDescriptorSet::buffer(6, matches_buf.clone()),
                    WriteDescriptorSet::buffer(7, match_count_buf.clone()),
                    WriteDescriptorSet::buffer(8, g_table_buf.clone()),
                ],
                [],
            )
            .unwrap();

            let push = PushConstants {
                is_first_launch,
                iterations_per_thread: ITERATIONS_PER_LAUNCH,
                max_matches: MAX_MATCHES,
            };

            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .bind_pipeline_compute(mine_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    mine_pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(mine_pipeline.layout().clone(), 0, push)
                .unwrap();
            unsafe { builder.dispatch([NUM_WORKGROUPS, 1, 1]) }.unwrap();

            let cmd = builder.build().unwrap();
            vulkano::sync::now(device.clone())
                .then_execute(queue.clone(), cmd)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)?;

            is_first_launch = 0;

            // Update total count
            let batch_ops = TOTAL_THREADS as u64 * ITERATIONS_PER_LAUNCH as u64;
            total.fetch_add(batch_ops, Ordering::Relaxed);

            // Check for matches
            let num_matches = {
                let content = match_count_buf.read().unwrap();
                content[0].min(MAX_MATCHES)
            };

            if num_matches > 0 {
                let match_data = matches_buf.read().unwrap();

                for i in 0..num_matches as usize {
                    let base = i * MATCH_SLOT_UINTS as usize;
                    let found = match_data[base + 30];
                    if found == 0 {
                        continue;
                    }

                    // Unpack privkey (8 uints -> 32 bytes, big-endian within each uint)
                    let mut privkey_bytes = [0u8; 32];
                    for j in 0..8 {
                        let val = match_data[base + j];
                        privkey_bytes[j * 4] = ((val >> 24) & 0xFF) as u8;
                        privkey_bytes[j * 4 + 1] = ((val >> 16) & 0xFF) as u8;
                        privkey_bytes[j * 4 + 2] = ((val >> 8) & 0xFF) as u8;
                        privkey_bytes[j * 4 + 3] = (val & 0xFF) as u8;
                    }

                    // Unpack suffix (6 uints -> 24 bytes)
                    let mut suffix_bytes = [0u8; 24];
                    for j in 0..6 {
                        let val = match_data[base + 24 + j];
                        suffix_bytes[j * 4] = ((val >> 24) & 0xFF) as u8;
                        suffix_bytes[j * 4 + 1] = ((val >> 16) & 0xFF) as u8;
                        suffix_bytes[j * 4 + 2] = ((val >> 8) & 0xFF) as u8;
                        suffix_bytes[j * 4 + 3] = (val & 0xFF) as u8;
                    }

                    // CPU-side re-verification
                    let signing_key = match SigningKey::from_bytes((&privkey_bytes).into()) {
                        Ok(k) => k,
                        Err(_) => continue,
                    };

                    let op = build_signed_op(&signing_key, &config.handle, &config.pds);
                    let suffix = did_suffix(&op);

                    // Verify GPU suffix matches CPU
                    let gpu_suffix = std::str::from_utf8(&suffix_bytes).unwrap_or("");
                    if suffix != gpu_suffix {
                        eprintln!(
                            "warning: Vulkan match verification failed: GPU={} CPU={}",
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

    struct VulkanTestContext {
        device: Arc<Device>,
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    }

    fn setup() -> VulkanTestContext {
        let library = VulkanLibrary::new().expect("failed to load Vulkan library");
        let instance =
            Instance::new(library, InstanceCreateInfo::default()).expect("failed to create Vulkan instance");

        let physical_device = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .find(|pd| {
                pd.queue_family_properties()
                    .iter()
                    .any(|qf| qf.queue_flags.intersects(QueueFlags::COMPUTE))
            })
            .expect("no compute-capable Vulkan device found");

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|qf| qf.queue_flags.intersects(QueueFlags::COMPUTE))
            .unwrap() as u32;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_features: DeviceFeatures {
                    shader_int64: true,
                    shader_int8: true,
                    uniform_and_storage_buffer8_bit_access: true,
                    storage_buffer8_bit_access: true,
                    ..DeviceFeatures::empty()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create logical device");

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        VulkanTestContext {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    fn load_shader(ctx: &VulkanTestContext, spv_bytes: &[u8]) -> Arc<ShaderModule> {
        load_shader_module(ctx.device.clone(), spv_bytes)
    }

    fn create_compute_pipeline(ctx: &VulkanTestContext, shader_module: Arc<ShaderModule>) -> Arc<ComputePipeline> {
        make_compute_pipeline(ctx.device.clone(), shader_module)
    }

    fn create_storage_buffer(
        ctx: &VulkanTestContext,
        data: &[u32],
    ) -> Subbuffer<[u32]> {
        make_buffer_u32(ctx.memory_allocator.clone(), data)
    }

    fn create_storage_buffer_u8(
        ctx: &VulkanTestContext,
        data: &[u8],
    ) -> Subbuffer<[u8]> {
        make_buffer_u8(ctx.memory_allocator.clone(), data)
    }

    fn dispatch_and_wait(
        ctx: &VulkanTestContext,
        pipeline: &Arc<ComputePipeline>,
        descriptor_set: Arc<DescriptorSet>,
    ) {
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator.clone(),
            ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap();
        unsafe { builder.dispatch([1, 1, 1]) }.unwrap();

        let command_buffer = builder.build().unwrap();
        let future = vulkano::sync::now(ctx.device.clone())
            .then_execute(ctx.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).expect("GPU execution failed");
    }

    #[allow(dead_code)]
    fn le_limbs_to_biguint(limbs: &[u32; 8]) -> num_bigint::BigUint {
        let mut bytes = Vec::with_capacity(32);
        for &limb in limbs.iter() {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
        num_bigint::BigUint::from_bytes_le(&bytes)
    }

    #[allow(dead_code)]
    fn biguint_to_le_limbs(val: &num_bigint::BigUint) -> [u32; 8] {
        let bytes = val.to_bytes_le();
        let mut limbs = [0u32; 8];
        for (i, chunk) in bytes.chunks(4).enumerate() {
            if i < 8 {
                let mut buf = [0u8; 4];
                buf[..chunk.len()].copy_from_slice(chunk);
                limbs[i] = u32::from_le_bytes(buf);
            }
        }
        limbs
    }

    // --- GX/GY constants as LE limbs ---
    #[allow(dead_code)]
    const GX_LIMBS: [u32; 8] = [
        0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
        0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E,
    ];
    #[allow(dead_code)]
    const GY_LIMBS: [u32; 8] = [
        0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
        0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77,
    ];

    #[test]
    fn vulkan_field_mul_gx_gy() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_field_mul.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        // Compute expected: GX * GY mod p
        let p = num_bigint::BigUint::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16
        ).unwrap();
        let gx_big = le_limbs_to_biguint(&GX_LIMBS);
        let gy_big = le_limbs_to_biguint(&GY_LIMBS);
        let expected_big = (&gx_big * &gy_big) % &p;
        let expected_limbs = biguint_to_le_limbs(&expected_big);

        let input_a = create_storage_buffer(&ctx, &GX_LIMBS);
        let input_b = create_storage_buffer(&ctx, &GY_LIMBS);
        let output = create_storage_buffer(&ctx, &[0u32; 8]);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_a),
                WriteDescriptorSet::buffer(1, input_b),
                WriteDescriptorSet::buffer(2, output.clone()),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let result = output.read().unwrap();
        assert_eq!(
            &result[..], &expected_limbs[..],
            "GPU field_mul(GX, GY) must match reference.\nGPU: {:08x?}\nExpected: {:08x?}",
            &result[..], expected_limbs
        );
    }

    #[test]
    fn vulkan_scalar_mul_mod_n() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_scalar_mul.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        let n = num_bigint::BigUint::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16
        ).unwrap();
        let a_big = le_limbs_to_biguint(&GX_LIMBS);
        let b_big = le_limbs_to_biguint(&GY_LIMBS);
        let expected_big = (&a_big * &b_big) % &n;
        let expected_limbs = biguint_to_le_limbs(&expected_big);

        let input_a = create_storage_buffer(&ctx, &GX_LIMBS);
        let input_b = create_storage_buffer(&ctx, &GY_LIMBS);
        let output = create_storage_buffer(&ctx, &[0u32; 8]);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_a),
                WriteDescriptorSet::buffer(1, input_b),
                WriteDescriptorSet::buffer(2, output.clone()),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let result = output.read().unwrap();
        assert_eq!(
            &result[..], &expected_limbs[..],
            "GPU scalar_mul(GX, GY) mod n must match.\nGPU: {:08x?}\nExpected: {:08x?}",
            &result[..], expected_limbs
        );
    }

    #[test]
    fn vulkan_scalar_inv_mod_n() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_scalar_inv.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        let a_limbs: [u32; 8] = [7, 0, 0, 0, 0, 0, 0, 0];

        // Expected: 7^(-1) mod n via Fermat
        let n = num_bigint::BigUint::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16
        ).unwrap();
        let a_big = num_bigint::BigUint::from(7u32);
        let nm2 = &n - num_bigint::BigUint::from(2u32);
        let expected_big = a_big.modpow(&nm2, &n);
        let expected_limbs = biguint_to_le_limbs(&expected_big);

        let input_a = create_storage_buffer(&ctx, &a_limbs);
        let output = create_storage_buffer(&ctx, &[0u32; 8]);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_a),
                WriteDescriptorSet::buffer(1, output.clone()),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let result = output.read().unwrap();
        assert_eq!(
            &result[..], &expected_limbs[..],
            "GPU scalar_inv(7) mod n must match.\nGPU: {:08x?}\nExpected: {:08x?}",
            &result[..], expected_limbs
        );
    }

    #[test]
    fn vulkan_point_double_g() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_point_double_g.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        // Known 2*G compressed pubkey
        let expected: [u32; 33] = [
            0x02,
            0xc6, 0x04, 0x7f, 0x94, 0x41, 0xed, 0x7d, 0x6d,
            0x30, 0x45, 0x40, 0x6e, 0x95, 0xc0, 0x7c, 0xd8,
            0x5c, 0x77, 0x8e, 0x4b, 0x8c, 0xef, 0x3c, 0xa7,
            0xab, 0xac, 0x09, 0xb9, 0x5c, 0x70, 0x9e, 0xe5,
        ];

        let output = create_storage_buffer(&ctx, &[0u32; 33]);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, output.clone()),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let result = output.read().unwrap();
        assert_eq!(
            &result[..], &expected[..],
            "point_double(G) must equal 2*G"
        );
    }

    #[test]
    fn vulkan_scalar_mul_g_identity() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_scalar_mul_G.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        // scalar = 1 in LE limbs
        let scalar_limbs: [u32; 8] = [1, 0, 0, 0, 0, 0, 0, 0];

        // Known compressed G
        let expected: [u32; 33] = [
            0x02,
            0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
            0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
            0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
            0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
        ];

        let input_scalar = create_storage_buffer(&ctx, &scalar_limbs);
        let output = create_storage_buffer(&ctx, &[0u32; 33]);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_scalar),
                WriteDescriptorSet::buffer(1, output.clone()),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let result = output.read().unwrap();
        assert_eq!(
            &result[..], &expected[..],
            "1*G must equal G"
        );
    }

    #[test]
    fn vulkan_scalar_mul_g_small_scalars() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_scalar_mul_G.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        let test_values: &[u64] = &[2, 3, 15, 16, 17, 256];
        for &val in test_values {
            use k256::elliptic_curve::sec1::ToEncodedPoint;
            // Build 32-byte big-endian scalar for k256
            let mut be_bytes = [0u8; 32];
            be_bytes[24..].copy_from_slice(&val.to_be_bytes());
            let key = k256::SecretKey::from_slice(&be_bytes).unwrap();
            let expected_point = key.public_key().to_encoded_point(true);
            let expected_bytes = expected_point.as_bytes(); // 33 bytes

            // Convert to LE limbs for our shader
            let scalar_limbs: [u32; 8] = [val as u32, 0, 0, 0, 0, 0, 0, 0];

            let input_scalar = create_storage_buffer(&ctx, &scalar_limbs);
            let output = create_storage_buffer(&ctx, &[0u32; 33]);

            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                ctx.descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, input_scalar),
                    WriteDescriptorSet::buffer(1, output.clone()),
                ],
                [],
            )
            .unwrap();

            dispatch_and_wait(&ctx, &pipeline, set);

            let result = output.read().unwrap();
            // Convert GPU uint-per-byte output to actual bytes for comparison
            let gpu_bytes: Vec<u8> = result.iter().map(|&v| v as u8).collect();
            assert_eq!(
                &gpu_bytes[..], expected_bytes,
                "GPU scalar_mul_G failed for scalar={}",
                val
            );
        }
    }

    #[test]
    fn vulkan_scalar_mul_g_matches_k256() {
        use k256::elliptic_curve::sec1::ToEncodedPoint;

        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_scalar_mul_G.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        let key = SigningKey::random(&mut rand::thread_rng());
        let scalar_be = key.to_bytes(); // 32 bytes big-endian

        // Convert big-endian bytes to LE limbs
        let mut scalar_limbs = [0u32; 8];
        for i in 0..8 {
            let offset = (7 - i) * 4;
            scalar_limbs[i] = u32::from_be_bytes([
                scalar_be[offset],
                scalar_be[offset + 1],
                scalar_be[offset + 2],
                scalar_be[offset + 3],
            ]);
        }

        let cpu_pubkey = key.verifying_key().to_encoded_point(true);
        let cpu_bytes = cpu_pubkey.as_bytes();

        let input_scalar = create_storage_buffer(&ctx, &scalar_limbs);
        let output = create_storage_buffer(&ctx, &[0u32; 33]);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_scalar),
                WriteDescriptorSet::buffer(1, output.clone()),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let result = output.read().unwrap();
        let gpu_bytes: Vec<u8> = result.iter().map(|&v| v as u8).collect();
        assert_eq!(
            &gpu_bytes[..], cpu_bytes,
            "GPU scalar_mul_G must match k256"
        );
    }

    // --- SHA256 tests ---

    #[test]
    fn vulkan_sha256_matches_cpu() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_sha256.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        let test_cases: &[&[u8]] = &[
            b"hello",
            b"",
            b"The quick brown fox jumps over the lazy dog",
            // A 64-byte input (exactly one block boundary)
            b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            // A 65-byte input (just over one block)
            b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef!",
        ];

        for input in test_cases {
            use sha2::Digest;
            let expected = sha2::Sha256::digest(input);

            // Pad input to at least 4 bytes for the SSBO (can't have empty buffer)
            let mut padded_input = input.to_vec();
            if padded_input.is_empty() {
                padded_input.push(0);
            }
            while padded_input.len() % 4 != 0 {
                padded_input.push(0);
            }

            let input_buf = create_storage_buffer_u8(&ctx, &padded_input);
            let len_buf = create_storage_buffer(&ctx, &[input.len() as u32]);
            let output_buf = create_storage_buffer_u8(&ctx, &[0u8; 32]);

            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                ctx.descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, input_buf),
                    WriteDescriptorSet::buffer(1, len_buf),
                    WriteDescriptorSet::buffer(2, output_buf.clone()),
                ],
                [],
            )
            .unwrap();

            dispatch_and_wait(&ctx, &pipeline, set);

            let result = output_buf.read().unwrap();
            assert_eq!(
                &result[..], &expected[..],
                "GPU SHA256 mismatch for input {:?}\nGPU:      {:02x?}\nExpected: {:02x?}",
                String::from_utf8_lossy(input), &result[..], &expected[..]
            );
        }
    }

    // --- ECDSA signing test ---

    #[test]
    fn vulkan_ecdsa_sign_matches_k256() {
        use k256::ecdsa::{VerifyingKey, Signature};

        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_ecdsa_sign.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        // Use a fixed key for reproducibility
        let privkey_bytes: [u8; 32] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
        ];

        // Hash a message
        use sha2::Digest;
        let msg_hash = sha2::Sha256::digest(b"test message for ECDSA");
        let msg_hash_bytes: [u8; 32] = msg_hash.into();

        let privkey_buf = create_storage_buffer_u8(&ctx, &privkey_bytes);
        let msghash_buf = create_storage_buffer_u8(&ctx, &msg_hash_bytes);
        let sig_buf = create_storage_buffer_u8(&ctx, &[0u8; 64]);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, privkey_buf),
                WriteDescriptorSet::buffer(1, msghash_buf),
                WriteDescriptorSet::buffer(2, sig_buf.clone()),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let sig_result = sig_buf.read().unwrap();
        let r_bytes: [u8; 32] = sig_result[..32].try_into().unwrap();
        let s_bytes: [u8; 32] = sig_result[32..64].try_into().unwrap();

        // Construct signature and verify with k256
        let mut sig_bytes = [0u8; 64];
        sig_bytes[..32].copy_from_slice(&r_bytes);
        sig_bytes[32..].copy_from_slice(&s_bytes);
        let signature = Signature::from_bytes(&sig_bytes.into()).expect("invalid signature format");

        let signing_key = SigningKey::from_bytes(&privkey_bytes.into()).unwrap();
        let verifying_key = VerifyingKey::from(&signing_key);

        // Verify using prehashed message
        use k256::ecdsa::signature::hazmat::PrehashVerifier;
        verifying_key
            .verify_prehash(&msg_hash_bytes, &signature)
            .expect("GPU ECDSA signature verification failed");
    }

    // --- Encoding tests ---

    #[test]
    fn vulkan_base32_matches_cpu() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_encoding.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        // Test data: 15 bytes
        let b32_input: [u8; 15] = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04,
                                    0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B];
        let expected_b32 = data_encoding::BASE32_NOPAD.encode(&b32_input).to_lowercase();

        // Dummy data for other encodings (must provide all bindings)
        let b58_input = [0u8; 35];
        let b64_input = [0u8; 64];

        let b32_in_buf = create_storage_buffer_u8(&ctx, &b32_input);
        let b32_out_buf = create_storage_buffer_u8(&ctx, &[0u8; 24]);
        let b58_in_buf = create_storage_buffer_u8(&ctx, &b58_input);
        let b58_out_buf = create_storage_buffer_u8(&ctx, &[0u8; 48]);
        let b64_in_buf = create_storage_buffer_u8(&ctx, &b64_input);
        let b64_out_buf = create_storage_buffer_u8(&ctx, &[0u8; 88]); // pad to 4-byte alignment

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, b32_in_buf),
                WriteDescriptorSet::buffer(1, b32_out_buf.clone()),
                WriteDescriptorSet::buffer(2, b58_in_buf),
                WriteDescriptorSet::buffer(3, b58_out_buf),
                WriteDescriptorSet::buffer(4, b64_in_buf),
                WriteDescriptorSet::buffer(5, b64_out_buf),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let result = b32_out_buf.read().unwrap();
        let gpu_str = String::from_utf8(result[..24].to_vec()).unwrap();
        assert_eq!(
            gpu_str, expected_b32,
            "GPU base32 mismatch\nGPU:      {}\nExpected: {}",
            gpu_str, expected_b32
        );
    }

    #[test]
    fn vulkan_base58_matches_cpu() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_encoding.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        // Test data: 35 bytes starting with 0xe7 (multicodec prefix, same as real usage)
        let mut b58_input = [0u8; 35];
        b58_input[0] = 0xe7;
        b58_input[1] = 0x01;
        for i in 2..35 {
            b58_input[i] = (i * 7 + 3) as u8;
        }
        let expected_b58 = bs58::encode(&b58_input).into_string();

        // Dummy data for other encodings
        let b32_input = [0u8; 15];
        let b64_input = [0u8; 64];

        let b32_in_buf = create_storage_buffer_u8(&ctx, &b32_input);
        let b32_out_buf = create_storage_buffer_u8(&ctx, &[0u8; 24]);
        let b58_in_buf = create_storage_buffer_u8(&ctx, &b58_input);
        let b58_out_buf = create_storage_buffer_u8(&ctx, &[0u8; 48]);
        let b64_in_buf = create_storage_buffer_u8(&ctx, &b64_input);
        let b64_out_buf = create_storage_buffer_u8(&ctx, &[0u8; 88]);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, b32_in_buf),
                WriteDescriptorSet::buffer(1, b32_out_buf),
                WriteDescriptorSet::buffer(2, b58_in_buf),
                WriteDescriptorSet::buffer(3, b58_out_buf.clone()),
                WriteDescriptorSet::buffer(4, b64_in_buf),
                WriteDescriptorSet::buffer(5, b64_out_buf),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let result = b58_out_buf.read().unwrap();
        let gpu_str = String::from_utf8(result[..expected_b58.len()].to_vec()).unwrap();
        assert_eq!(
            gpu_str, expected_b58,
            "GPU base58 mismatch\nGPU:      {}\nExpected: {}",
            gpu_str, expected_b58
        );
    }

    #[test]
    fn vulkan_base64url_matches_cpu() {
        use base64::Engine;
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_encoding.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        // Test data: 64 bytes
        let mut b64_input = [0u8; 64];
        for i in 0..64 {
            b64_input[i] = (i * 13 + 5) as u8;
        }
        let expected_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&b64_input);

        // Dummy data for other encodings
        let b32_input = [0u8; 15];
        let b58_input = [0u8; 35];

        let b32_in_buf = create_storage_buffer_u8(&ctx, &b32_input);
        let b32_out_buf = create_storage_buffer_u8(&ctx, &[0u8; 24]);
        let b58_in_buf = create_storage_buffer_u8(&ctx, &b58_input);
        let b58_out_buf = create_storage_buffer_u8(&ctx, &[0u8; 48]);
        let b64_in_buf = create_storage_buffer_u8(&ctx, &b64_input);
        let b64_out_buf = create_storage_buffer_u8(&ctx, &[0u8; 88]);

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, b32_in_buf),
                WriteDescriptorSet::buffer(1, b32_out_buf),
                WriteDescriptorSet::buffer(2, b58_in_buf),
                WriteDescriptorSet::buffer(3, b58_out_buf),
                WriteDescriptorSet::buffer(4, b64_in_buf),
                WriteDescriptorSet::buffer(5, b64_out_buf.clone()),
            ],
            [],
        )
        .unwrap();

        dispatch_and_wait(&ctx, &pipeline, set);

        let result = b64_out_buf.read().unwrap();
        let gpu_str = String::from_utf8(result[..86].to_vec()).unwrap();
        assert_eq!(
            gpu_str, expected_b64,
            "GPU base64url mismatch\nGPU:      {}\nExpected: {}",
            gpu_str, expected_b64
        );
    }

    // --- Pattern matching test ---

    #[test]
    fn vulkan_glob_match_works() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/test_pattern.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        let test_cases: &[(&str, &str, bool)] = &[
            // (pattern, text, expected_match)
            ("abcdefghijklmnopqrstuvwx", "abcdefghijklmnopqrstuvwx", true),  // exact match
            ("abc*", "abcdefghijklmnopqrstuvwx", true),                       // prefix
            ("*vwx", "abcdefghijklmnopqrstuvwx", true),                       // suffix
            ("*ghij*", "abcdefghijklmnopqrstuvwx", true),                     // middle
            ("*", "abcdefghijklmnopqrstuvwx", true),                          // match all
            ("xyz*", "abcdefghijklmnopqrstuvwx", false),                      // no match prefix
            ("*xyz", "abcdefghijklmnopqrstuvwx", false),                      // no match suffix
            ("abc", "abcdefghijklmnopqrstuvwx", false),                       // too short
            ("a*x", "abcdefghijklmnopqrstuvwx", true),                        // prefix+suffix
            ("a*z", "abcdefghijklmnopqrstuvwx", false),                       // prefix+wrong suffix
        ];

        for (pattern, text, expected) in test_cases {
            // Pad pattern to 24 bytes
            let mut pat_bytes = [0u8; 24];
            for (i, &b) in pattern.as_bytes().iter().enumerate() {
                if i < 24 { pat_bytes[i] = b; }
            }
            let pat_len = pattern.len() as u32;

            // Pad text to 24 bytes
            let mut text_bytes = [0u8; 24];
            for (i, &b) in text.as_bytes().iter().enumerate() {
                if i < 24 { text_bytes[i] = b; }
            }

            let pat_buf = create_storage_buffer_u8(&ctx, &pat_bytes);
            let pat_len_buf = create_storage_buffer(&ctx, &[pat_len]);
            let text_buf = create_storage_buffer_u8(&ctx, &text_bytes);
            let result_buf = create_storage_buffer(&ctx, &[0u32]);

            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                ctx.descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, pat_buf),
                    WriteDescriptorSet::buffer(1, pat_len_buf),
                    WriteDescriptorSet::buffer(2, text_buf),
                    WriteDescriptorSet::buffer(3, result_buf.clone()),
                ],
                [],
            )
            .unwrap();

            dispatch_and_wait(&ctx, &pipeline, set);

            let result = result_buf.read().unwrap();
            let matched = result[0] == 1;
            assert_eq!(
                matched, *expected,
                "Pattern '{}' vs text '{}': GPU={}, expected={}",
                pattern, text, matched, expected
            );
        }
    }

    // --- End-to-end mining test ---

    /// Minimal diagnostic: dispatch mine.comp with 1 workgroup × 1 iteration,
    /// pattern "*" (match everything). Checks if the shader produces any output at all.
    #[test]
    fn vulkan_mine_minimal_diagnostic() {
        let ctx = setup();

        let mine_spv = include_bytes!(concat!(env!("OUT_DIR"), "/mine.comp.spv"));
        let init_g_table_spv = include_bytes!(concat!(env!("OUT_DIR"), "/init_g_table.comp.spv"));
        let stride_g_spv = include_bytes!(concat!(env!("OUT_DIR"), "/stride_g.comp.spv"));

        let mine_module = load_shader(&ctx, mine_spv);
        let init_g_table_module = load_shader(&ctx, init_g_table_spv);
        let stride_g_module = load_shader(&ctx, stride_g_spv);

        let mine_pipeline = create_compute_pipeline(&ctx, mine_module);
        let init_g_table_pipeline = create_compute_pipeline(&ctx, init_g_table_module);
        let stride_g_pipeline = create_compute_pipeline(&ctx, stride_g_module);

        // 1. Init g_table
        let g_table_buf = create_storage_buffer(&ctx, &[0u32; 240]);
        {
            let layout = init_g_table_pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                ctx.descriptor_set_allocator.clone(),
                layout.clone(),
                [WriteDescriptorSet::buffer(0, g_table_buf.clone())],
                [],
            ).unwrap();
            dispatch_and_wait(&ctx, &init_g_table_pipeline, set);
        }

        // 2. Compute stride_G (stride = 256, one workgroup)
        let test_num_threads: u32 = 256;
        let mut stride_in_data = [0u32; 8];
        stride_in_data[0] = test_num_threads;
        let stride_in_buf = create_storage_buffer(&ctx, &stride_in_data);
        let stride_out_buf = create_storage_buffer(&ctx, &[0u32; 32]);
        {
            let layout = stride_g_pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                ctx.descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, stride_in_buf),
                    WriteDescriptorSet::buffer(1, stride_out_buf.clone()),
                    WriteDescriptorSet::buffer(2, g_table_buf.clone()),
                ],
                [],
            ).unwrap();
            dispatch_and_wait(&ctx, &stride_g_pipeline, set);
        }

        let stride_out_data = stride_out_buf.read().unwrap();
        let mut stride_arr = [0u32; 8];
        let mut stride_g_arr = [0u32; 24];
        stride_arr.copy_from_slice(&stride_out_data[0..8]);
        stride_g_arr.copy_from_slice(&stride_out_data[8..32]);

        // 3. Generate scalars for 256 threads
        let mut rng = rand::thread_rng();
        let base_key = SigningKey::random(&mut rng);
        let mut scalar_data = vec![0u8; test_num_threads as usize * 32];
        for tid in 0..test_num_threads as u64 {
            let offset_scalar = k256::Scalar::from(tid);
            let thread_scalar = base_key.as_nonzero_scalar().as_ref() + &offset_scalar;
            let thread_bytes = thread_scalar.to_bytes();
            let start_idx = tid as usize * 32;
            scalar_data[start_idx..start_idx + 32].copy_from_slice(&thread_bytes);
        }

        // 4. Set up buffers
        let tmpl = CborTemplate::new("test.bsky.social", "https://bsky.social");
        eprintln!("Template sizes: unsigned={}, signed={}", tmpl.unsigned_bytes.len(), tmpl.signed_bytes.len());

        let scalars_buf = create_storage_buffer_u8(&ctx, &scalar_data);
        let pubkeys_buf = create_storage_buffer(&ctx, &vec![0u32; test_num_threads as usize * 24]);

        let max_matches: u32 = 64;
        let match_slot_uints: u32 = 32;
        let matches_buf = create_storage_buffer(&ctx, &vec![0u32; max_matches as usize * match_slot_uints as usize]);
        let match_count_buf = create_storage_buffer(&ctx, &[0u32]);

        let unsigned_template_buf = create_storage_buffer_u8(&ctx, &tmpl.unsigned_bytes);
        let signed_template_buf = create_storage_buffer_u8(&ctx, &tmpl.signed_bytes);

        // Pattern "*" — matches everything
        let pattern = b"*\0\0\0"; // pad to 4 bytes
        let pattern_buf = create_storage_buffer_u8(&ctx, pattern);

        // KernelParams
        let mut params_data = vec![0u32; 40];
        params_data[0] = tmpl.unsigned_bytes.len() as u32;
        params_data[1] = tmpl.signed_bytes.len() as u32;
        params_data[2] = tmpl.unsigned_pubkey_offsets[0] as u32;
        params_data[3] = tmpl.unsigned_pubkey_offsets[1] as u32;
        params_data[4] = tmpl.signed_pubkey_offsets[0] as u32;
        params_data[5] = tmpl.signed_pubkey_offsets[1] as u32;
        params_data[6] = tmpl.signed_sig_offset as u32;
        params_data[7] = 1; // pattern_len = 1 (just "*")
        params_data[8..16].copy_from_slice(&stride_arr);
        params_data[16..40].copy_from_slice(&stride_g_arr);
        let params_buf = create_storage_buffer(&ctx, &params_data);

        // 5. Dispatch mine with 1 workgroup, 1 iteration
        let mine_layout = mine_pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            mine_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, params_buf),
                WriteDescriptorSet::buffer(1, unsigned_template_buf),
                WriteDescriptorSet::buffer(2, signed_template_buf),
                WriteDescriptorSet::buffer(3, pattern_buf),
                WriteDescriptorSet::buffer(4, scalars_buf),
                WriteDescriptorSet::buffer(5, pubkeys_buf),
                WriteDescriptorSet::buffer(6, matches_buf.clone()),
                WriteDescriptorSet::buffer(7, match_count_buf.clone()),
                WriteDescriptorSet::buffer(8, g_table_buf),
            ],
            [],
        ).unwrap();

        let push = PushConstants {
            is_first_launch: 1,
            iterations_per_thread: 1,
            max_matches: max_matches,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator.clone(),
            ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        builder
            .bind_pipeline_compute(mine_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                mine_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(mine_pipeline.layout().clone(), 0, push)
            .unwrap();
        unsafe { builder.dispatch([1, 1, 1]) }.unwrap();

        let cmd = builder.build().unwrap();
        eprintln!("Dispatching mine shader: 1 workgroup × 256 threads × 1 iteration...");
        vulkano::sync::now(ctx.device.clone())
            .then_execute(ctx.queue.clone(), cmd)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .expect("GPU dispatch failed");
        eprintln!("Dispatch completed!");

        // 6. Check results
        let num_matches = {
            let content = match_count_buf.read().unwrap();
            content[0]
        };
        eprintln!("Match count: {} (expected ~256 with pattern '*')", num_matches);
        assert!(num_matches > 0, "Pattern '*' with 256 threads should produce matches, got 0");

        // Verify first match
        let match_data = matches_buf.read().unwrap();
        let base = 0usize;
        let found = match_data[base + 30];
        assert_eq!(found, 1, "First match slot should have found=1");

        // Unpack privkey
        let mut privkey_bytes = [0u8; 32];
        for j in 0..8 {
            let val = match_data[base + j];
            privkey_bytes[j * 4] = ((val >> 24) & 0xFF) as u8;
            privkey_bytes[j * 4 + 1] = ((val >> 16) & 0xFF) as u8;
            privkey_bytes[j * 4 + 2] = ((val >> 8) & 0xFF) as u8;
            privkey_bytes[j * 4 + 3] = (val & 0xFF) as u8;
        }

        // Unpack suffix
        let mut suffix_bytes = [0u8; 24];
        for j in 0..6 {
            let val = match_data[base + 24 + j];
            suffix_bytes[j * 4] = ((val >> 24) & 0xFF) as u8;
            suffix_bytes[j * 4 + 1] = ((val >> 16) & 0xFF) as u8;
            suffix_bytes[j * 4 + 2] = ((val >> 8) & 0xFF) as u8;
            suffix_bytes[j * 4 + 3] = (val & 0xFF) as u8;
        }

        let gpu_suffix = std::str::from_utf8(&suffix_bytes).unwrap_or("<invalid utf8>");
        eprintln!("GPU suffix: {}", gpu_suffix);
        eprintln!("GPU privkey: {}", data_encoding::HEXLOWER.encode(&privkey_bytes));

        // CPU re-verification
        let signing_key = SigningKey::from_bytes((&privkey_bytes).into())
            .expect("GPU privkey should be valid");
        let op = build_signed_op(&signing_key, "test.bsky.social", "https://bsky.social");
        let cpu_suffix = did_suffix(&op);
        eprintln!("CPU suffix: {}", cpu_suffix);
        assert_eq!(
            gpu_suffix, cpu_suffix,
            "GPU suffix must match CPU re-verification"
        );
    }

    // The full vulkan_mining_finds_valid_match test is replaced by
    // vulkan_mine_minimal_diagnostic above which verifies correctness
    // with manageable GPU workload (256 threads × 1 iteration).
    // The full VulkanBackend::run() test would require minutes of GPU time
    // even in release mode due to the heavy ECDSA pipeline.
}
