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

    use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
    use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
    use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
    use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
    use vulkano::instance::{Instance, InstanceCreateInfo};
    use vulkano::memory::allocator::StandardMemoryAllocator;
    use vulkano::pipeline::compute::ComputePipelineCreateInfo;
    use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
    use vulkano::pipeline::{PipelineLayout, PipelineShaderStageCreateInfo};
    use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
    use vulkano::sync::GpuFuture;
    use vulkano::VulkanLibrary;

    #[allow(dead_code)]
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

    #[test]
    fn vulkan_trivial_dispatch() {
        use vulkano::pipeline::compute::ComputePipeline;

        let ctx = setup();

        // Load SPIR-V bytecode compiled from vulkan/mine.comp
        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/mine.comp.spv"));
        let spirv_words: Vec<u32> = spirv_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let shader_module = unsafe {
            ShaderModule::new(ctx.device.clone(), ShaderModuleCreateInfo::new(&spirv_words))
        }
        .expect("failed to create shader module");

        let entry_point = shader_module
            .entry_point("main")
            .expect("shader has no 'main' entry point");

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .expect("failed to create pipeline layout");

        let pipeline = ComputePipeline::new(
            ctx.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

        // Build and submit command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            ctx.command_buffer_allocator.clone(),
            ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create command buffer builder");

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
}
