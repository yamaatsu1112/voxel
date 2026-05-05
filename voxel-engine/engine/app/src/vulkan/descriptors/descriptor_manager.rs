use crate::vulkan::resource_config::{DescriptorPoolId, DescriptorSetId};
use ash::vk;
use std::marker::PhantomData;
use std::ptr;

/// Type-safe identifier for a descriptor set layout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorSetLayoutId(usize);

/// Manages Vulkan descriptor set layouts, pools, and sets
pub struct DescriptorManager {
    device: ash::Device,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_pools: Vec<Option<vk::DescriptorPool>>,
    free_pool_indices: Vec<usize>,
    descriptor_sets: Vec<Option<vk::DescriptorSet>>,
    free_descriptor_set_indices: Vec<usize>,
}

impl DescriptorManager {
    /// Create a new DescriptorManager
    pub fn new(device: ash::Device) -> Self {
        Self {
            device,
            descriptor_set_layouts: Vec::new(),
            descriptor_pools: Vec::new(),
            free_pool_indices: Vec::new(),
            descriptor_sets: Vec::new(),
            free_descriptor_set_indices: Vec::new(),
        }
    }

    /// Create a descriptor set layout and return its ID
    pub fn create_descriptor_set_layout(
        &mut self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> DescriptorSetLayoutId {
        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            _marker: PhantomData,
        };

        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("Failed to create descriptor set layout")
        };

        let layout_index = self.descriptor_set_layouts.len();
        self.descriptor_set_layouts.push(descriptor_set_layout);
        DescriptorSetLayoutId(layout_index)
    }

    /// Get descriptor set layout by ID
    pub fn get_descriptor_set_layout(
        &self,
        layout_id: DescriptorSetLayoutId,
    ) -> Option<vk::DescriptorSetLayout> {
        self.descriptor_set_layouts.get(layout_id.0).copied()
    }

    /// Create a descriptor pool and return its ID
    pub fn create_descriptor_pool(
        &mut self,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
        flags: vk::DescriptorPoolCreateFlags,
    ) -> DescriptorPoolId {
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags,
            max_sets,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            _marker: PhantomData,
        };

        let descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create descriptor pool")
        };

        if let Some(index) = self.free_pool_indices.pop() {
            self.descriptor_pools[index] = Some(descriptor_pool);
            DescriptorPoolId(index)
        } else {
            let index = self.descriptor_pools.len();
            self.descriptor_pools.push(Some(descriptor_pool));
            DescriptorPoolId(index)
        }
    }

    /// Create a single descriptor set and return its ID
    pub fn create_descriptor_set(
        &mut self,
        pool_id: DescriptorPoolId,
        layout_id: DescriptorSetLayoutId,
    ) -> DescriptorSetId {
        assert!(
            pool_id != DescriptorPoolId::INVALID,
            "Descriptor pool not initialized"
        );
        let descriptor_pool = self
            .descriptor_pools
            .get(pool_id.0)
            .and_then(|opt| *opt)
            .expect("Descriptor pool missing");

        let layout = self.descriptor_set_layouts[layout_id.0];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &layout,
            _marker: PhantomData,
        };

        let descriptor_set = unsafe {
            self.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor set")
        };

        self.store_descriptor_set(descriptor_set[0])
    }

    fn store_descriptor_set(&mut self, descriptor_set: vk::DescriptorSet) -> DescriptorSetId {
        if let Some(index) = self.free_descriptor_set_indices.pop() {
            self.descriptor_sets[index] = Some(descriptor_set);
            DescriptorSetId(index)
        } else {
            let index = self.descriptor_sets.len();
            self.descriptor_sets.push(Some(descriptor_set));
            DescriptorSetId(index)
        }
    }

    /// Get descriptor set by ID.
    pub fn get_descriptor_set(&self, id: DescriptorSetId) -> Option<vk::DescriptorSet> {
        if id == DescriptorSetId::INVALID {
            return None;
        }
        self.descriptor_sets.get(id.0).and_then(|opt| *opt)
    }

    pub fn update_buffer(
        &self,
        set_id: DescriptorSetId,
        binding: u32,
        array_index: u32,
        buffer_infos: &[vk::DescriptorBufferInfo],
        descriptor_type: vk::DescriptorType,
    ) {
        let dst_set = self
            .get_descriptor_set(set_id)
            .expect("Descriptor set not found");

        let descriptor_write = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set,
            dst_binding: binding,
            dst_array_element: array_index,
            descriptor_count: buffer_infos.len() as u32,
            descriptor_type,
            p_image_info: ptr::null(),
            p_buffer_info: buffer_infos.as_ptr(),
            p_texel_buffer_view: ptr::null(),
            _marker: PhantomData,
        };

        unsafe {
            self.device.update_descriptor_sets(&[descriptor_write], &[]);
        }
    }

    pub fn update_image(
        &self,
        set_id: DescriptorSetId,
        binding: u32,
        array_index: u32,
        image_infos: &[vk::DescriptorImageInfo],
        descriptor_type: vk::DescriptorType,
    ) {
        let dst_set = self
            .get_descriptor_set(set_id)
            .expect("Descriptor set not found");

        let descriptor_write = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set,
            dst_binding: binding,
            dst_array_element: array_index,
            descriptor_count: image_infos.len() as u32,
            descriptor_type,
            p_image_info: image_infos.as_ptr(),
            p_buffer_info: ptr::null(),
            p_texel_buffer_view: ptr::null(),
            _marker: PhantomData,
        };

        unsafe {
            self.device.update_descriptor_sets(&[descriptor_write], &[]);
        }
    }

    /// Explicitly clean up all descriptor resources
    pub fn cleanup(&mut self) {
        unsafe {
            for pool in self.descriptor_pools.drain(..).flatten() {
                self.device.destroy_descriptor_pool(pool, None);
            }
            self.free_pool_indices.clear();
            self.free_descriptor_set_indices.clear();
            self.descriptor_sets.clear();

            for &layout in &self.descriptor_set_layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.descriptor_set_layouts.clear();
        }
    }
}
