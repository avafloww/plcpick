use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc;
use super::{Match, MiningBackend, MiningConfig};

#[allow(dead_code)]
pub struct VulkanBackend {
    pub device_index: usize,
}

impl MiningBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn run(
        &self,
        _config: &MiningConfig,
        _stop: &AtomicBool,
        _total: &AtomicU64,
        _tx: mpsc::Sender<Match>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
    use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
    use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
    use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
    use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
    use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
    use vulkano::instance::{Instance, InstanceCreateInfo};
    use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
    use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
    use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
    use vulkano::pipeline::{Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
    use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
    use vulkano::sync::GpuFuture;
    use vulkano::device::DeviceFeatures;
    use vulkano::VulkanLibrary;
    use k256::elliptic_curve::sec1::ToEncodedPoint;

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
        let spirv_words: Vec<u32> = spv_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        unsafe {
            ShaderModule::new(ctx.device.clone(), ShaderModuleCreateInfo::new(&spirv_words))
        }
        .expect("failed to create shader module")
    }

    fn create_compute_pipeline(ctx: &VulkanTestContext, shader_module: Arc<ShaderModule>) -> Arc<ComputePipeline> {
        let entry_point = shader_module.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .unwrap();
        ComputePipeline::new(
            ctx.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap()
    }

    fn create_storage_buffer(
        ctx: &VulkanTestContext,
        data: &[u32],
    ) -> vulkano::buffer::Subbuffer<[u32]> {
        Buffer::from_iter(
            ctx.memory_allocator.clone(),
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

    fn le_limbs_to_biguint(limbs: &[u32; 8]) -> num_bigint::BigUint {
        let mut bytes = Vec::with_capacity(32);
        for &limb in limbs.iter() {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
        num_bigint::BigUint::from_bytes_le(&bytes)
    }

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
    const GX_LIMBS: [u32; 8] = [
        0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
        0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E,
    ];
    const GY_LIMBS: [u32; 8] = [
        0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
        0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77,
    ];

    #[test]
    fn vulkan_trivial_dispatch() {
        let ctx = setup();

        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/mine.comp.spv"));
        let shader_module = load_shader(&ctx, spirv_bytes);
        let pipeline = create_compute_pipeline(&ctx, shader_module);

        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator.clone(),
            ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(pipeline.clone())
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
        use k256::ecdsa::SigningKey;

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
}
