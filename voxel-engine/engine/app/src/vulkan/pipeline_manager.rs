use crate::vulkan::pipeline_config::{ComputePipelineConfig, GraphicsPipelineConfig};
use crate::vulkan::resource_config::{ComputePipelineId, GraphicsPipelineId};
use ash::vk;
use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::path::Path;
use std::ptr;

/// Manages Vulkan graphics pipelines and their layouts
pub struct PipelineManager {
    device: ash::Device,
    /// Graphics pipelines stored in flat Vec with free list
    graphics_pipelines: Vec<Option<(vk::Pipeline, vk::PipelineLayout)>>,
    free_graphics_pipeline_indices: Vec<usize>,
    /// Compute pipelines stored in flat Vec with free list
    compute_pipelines: Vec<Option<(vk::Pipeline, vk::PipelineLayout)>>,
    free_compute_pipeline_indices: Vec<usize>,
    // Graphics pipeline libraries
    vertex_input_interface_libraries: Vec<vk::Pipeline>,
    pre_rasterization_shaders_libraries: Vec<vk::Pipeline>,
    fragment_shader_libraries: Vec<vk::Pipeline>,
    fragment_output_interface_libraries: Vec<vk::Pipeline>,
}

impl PipelineManager {
    /// Create a new PipelineManager
    pub fn new(device: ash::Device) -> Self {
        Self {
            device,
            graphics_pipelines: Vec::new(),
            free_graphics_pipeline_indices: Vec::new(),
            compute_pipelines: Vec::new(),
            free_compute_pipeline_indices: Vec::new(),
            vertex_input_interface_libraries: Vec::new(),
            pre_rasterization_shaders_libraries: Vec::new(),
            fragment_shader_libraries: Vec::new(),
            fragment_output_interface_libraries: Vec::new(),
        }
    }

    pub fn create_graphics_pipeline(
        &mut self,
        config: &GraphicsPipelineConfig,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        swapchain_format: vk::Format,
    ) -> GraphicsPipelineId {
        let pipeline_layout = self
            .create_pipeline_layout_internal(descriptor_set_layouts, config.push_constant_ranges);

        let color_formats: Vec<vk::Format> = config
            .color_attachment_formats
            .iter()
            .map(|format| format.resolve(swapchain_format))
            .collect();

        let vertex_shader_path = Path::new(config.vert_shader_path);
        let fragment_shader_path = Path::new(config.frag_shader_path);

        let pipeline = self.create_graphics_pipeline_internal(
            pipeline_layout,
            &color_formats,
            vertex_shader_path,
            fragment_shader_path,
            config.binding_descriptions,
            config.attribute_descriptions,
            config.blend_attachments,
            config.logic_op_enable,
            config.logic_op,
            config.blend_constants,
        );

        let idx = if let Some(idx) = self.free_graphics_pipeline_indices.pop() {
            self.graphics_pipelines[idx] = Some((pipeline, pipeline_layout));
            idx
        } else {
            let idx = self.graphics_pipelines.len();
            self.graphics_pipelines
                .push(Some((pipeline, pipeline_layout)));
            idx
        };
        GraphicsPipelineId(idx)
    }

    pub fn destroy_graphics_pipeline(&mut self, id: GraphicsPipelineId) {
        if id == GraphicsPipelineId::INVALID {
            return;
        }
        if let Some(slot) = self.graphics_pipelines.get_mut(id.0)
            && let Some((pipeline, layout)) = slot.take()
        {
            unsafe {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_pipeline_layout(layout, None);
            }
            self.free_graphics_pipeline_indices.push(id.0);
        }
    }

    pub fn get_graphics_pipeline(
        &self,
        id: GraphicsPipelineId,
    ) -> Option<(vk::Pipeline, vk::PipelineLayout)> {
        if id == GraphicsPipelineId::INVALID {
            return None;
        }
        self.graphics_pipelines.get(id.0).and_then(|opt| *opt)
    }

    pub fn create_compute_pipeline(
        &mut self,
        config: &ComputePipelineConfig,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> ComputePipelineId {
        let pipeline_layout = self
            .create_pipeline_layout_internal(descriptor_set_layouts, config.push_constant_ranges);

        let compute_shader_path = Path::new(config.shader_path);

        let pipeline = self.create_compute_pipeline_internal(pipeline_layout, compute_shader_path);

        let idx = if let Some(idx) = self.free_compute_pipeline_indices.pop() {
            self.compute_pipelines[idx] = Some((pipeline, pipeline_layout));
            idx
        } else {
            let idx = self.compute_pipelines.len();
            self.compute_pipelines
                .push(Some((pipeline, pipeline_layout)));
            idx
        };
        ComputePipelineId(idx)
    }

    pub fn destroy_compute_pipeline(&mut self, id: ComputePipelineId) {
        if id == ComputePipelineId::INVALID {
            return;
        }
        if let Some(slot) = self.compute_pipelines.get_mut(id.0)
            && let Some((pipeline, layout)) = slot.take()
        {
            unsafe {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_pipeline_layout(layout, None);
            }
            self.free_compute_pipeline_indices.push(id.0);
        }
    }

    pub fn get_compute_pipeline(
        &self,
        id: ComputePipelineId,
    ) -> Option<(vk::Pipeline, vk::PipelineLayout)> {
        if id == ComputePipelineId::INVALID {
            return None;
        }
        self.compute_pipelines.get(id.0).and_then(|opt| *opt)
    }

    fn create_pipeline_layout_internal(
        &self,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> vk::PipelineLayout {
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: descriptor_set_layouts.len() as u32,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as u32,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
            _marker: PhantomData,
        };

        unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout")
        }
    }

    fn create_graphics_pipeline_internal(
        &mut self,
        pipeline_layout: vk::PipelineLayout,
        color_attachment_formats: &[vk::Format],
        vertex_shader_path: &Path,
        fragment_shader_path: &Path,
        binding_descriptions: &[vk::VertexInputBindingDescription],
        attribute_descriptions: &[vk::VertexInputAttributeDescription],
        blend_attachments: &[vk::PipelineColorBlendAttachmentState],
        logic_op_enable: bool,
        logic_op: vk::LogicOp,
        blend_constants: [f32; 4],
    ) -> vk::Pipeline {
        let vertex_input_interface = self
            .create_vertex_input_interface_library(binding_descriptions, attribute_descriptions);

        let pre_rasterization_shaders =
            self.create_pre_rasterization_shaders_library(pipeline_layout, vertex_shader_path);

        let fragment_shader_library =
            self.create_fragment_shader_library(pipeline_layout, fragment_shader_path);

        let fragment_output_interface = self.create_fragment_output_interface_library(
            color_attachment_formats,
            blend_attachments,
            logic_op_enable,
            logic_op,
            blend_constants,
        );

        let libraries = [
            vertex_input_interface,
            pre_rasterization_shaders,
            fragment_shader_library,
            fragment_output_interface,
        ];
        let linking_info = vk::PipelineLibraryCreateInfoKHR {
            p_next: ptr::null(),
            library_count: libraries.len() as u32,
            p_libraries: libraries.as_ptr(),
            ..Default::default()
        };
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: &linking_info as *const _ as *const c_void,
            flags: vk::PipelineCreateFlags::empty(),
            layout: pipeline_layout,
            ..Default::default()
        };

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .expect("Failed to create graphics pipeline")
        };

        pipelines[0]
    }

    /// Create vertex input interface library
    fn create_vertex_input_interface_library(
        &mut self,
        binding_descriptions: &[vk::VertexInputBindingDescription],
        attribute_descriptions: &[vk::VertexInputAttributeDescription],
    ) -> vk::Pipeline {
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_binding_description_count: binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
            vertex_attribute_description_count: attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
            _marker: PhantomData,
        };
        let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            _marker: PhantomData,
        };
        let library_create_info = vk::GraphicsPipelineLibraryCreateInfoEXT {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
            p_next: ptr::null(),
            flags: vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE,
            _marker: PhantomData,
        };
        let vertex_input_interface_info = vk::GraphicsPipelineCreateInfo {
            p_next: &library_create_info as *const _ as *const c_void,
            flags: vk::PipelineCreateFlags::LIBRARY_KHR
                | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT,
            p_vertex_input_state: &vertex_input_state_create_info,
            p_input_assembly_state: &input_assembly_state_create_info,
            ..Default::default()
        };
        let vertex_input_interface = unsafe {
            self.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vertex_input_interface_info],
                    None,
                )
                .expect("Failed to create graphics pipeline")
        };

        let library = vertex_input_interface[0];
        self.vertex_input_interface_libraries.push(library);
        library
    }

    /// Create pre-rasterization shaders library
    fn create_pre_rasterization_shaders_library(
        &mut self,
        pipeline_layout: vk::PipelineLayout,
        vertex_shader_path: &Path,
    ) -> vk::Pipeline {
        let library_create_info = vk::GraphicsPipelineLibraryCreateInfoEXT {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
            p_next: ptr::null(),
            flags: vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS,
            _marker: PhantomData,
        };
        let vertex_shader_code = Self::read_shader_code(vertex_shader_path);
        let vertex_shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: vertex_shader_code.len(),
            p_code: vertex_shader_code.as_ptr() as *const u32,
            ..Default::default()
        };
        let main_function_name = CString::new("main").unwrap();
        let vertex_shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            p_next: &vertex_shader_module_create_info as *const _ as *const c_void,
            stage: vk::ShaderStageFlags::VERTEX,
            p_name: main_function_name.as_ptr(),
            ..Default::default()
        };
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineDynamicStateCreateFlags::empty(),
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            _marker: PhantomData,
        };

        let viewport_create_info = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            viewport_count: 1,
            p_viewports: ptr::null(),
            scissor_count: 1,
            p_scissors: ptr::null(),
            _marker: PhantomData,
        };
        let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
            _marker: PhantomData,
        };
        let pre_rasterization_shaders_info = vk::GraphicsPipelineCreateInfo {
            p_next: &library_create_info as *const _ as *const c_void,
            flags: vk::PipelineCreateFlags::LIBRARY_KHR
                | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT,
            render_pass: vk::RenderPass::null(),
            stage_count: 1,
            p_stages: &vertex_shader_stage_create_info,
            layout: pipeline_layout,
            p_dynamic_state: &dynamic_state_create_info,
            p_viewport_state: &viewport_create_info,
            p_rasterization_state: &rasterizer_create_info,
            ..Default::default()
        };
        let pre_rasterization_shaders = unsafe {
            self.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pre_rasterization_shaders_info],
                    None,
                )
                .expect("Failed to create graphics pipeline")
        };

        let library = pre_rasterization_shaders[0];
        self.pre_rasterization_shaders_libraries.push(library);
        library
    }

    /// Create fragment shader library
    fn create_fragment_shader_library(
        &mut self,
        pipeline_layout: vk::PipelineLayout,
        fragment_shader_path: &Path,
    ) -> vk::Pipeline {
        let library_create_info = vk::GraphicsPipelineLibraryCreateInfoEXT {
            flags: vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER,
            ..Default::default()
        };
        let fragment_shader_code = Self::read_shader_code(fragment_shader_path);
        let fragment_shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: fragment_shader_code.len(),
            p_code: fragment_shader_code.as_ptr() as *const u32,
            ..Default::default()
        };
        let main_function_name = CString::new("main").unwrap();
        let fragment_shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            p_next: &fragment_shader_module_create_info as *const _ as *const c_void,
            stage: vk::ShaderStageFlags::FRAGMENT,
            p_name: main_function_name.as_ptr(),
            ..Default::default()
        };
        let depth_stencil_create_info = vk::PipelineDepthStencilStateCreateInfo {
            ..Default::default()
        };
        let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
            _marker: PhantomData,
        };
        let fragment_shader_library_info = vk::GraphicsPipelineCreateInfo {
            p_next: &library_create_info as *const _ as *const c_void,
            flags: vk::PipelineCreateFlags::LIBRARY_KHR
                | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT,
            stage_count: 1,
            p_stages: &fragment_shader_stage_create_info,
            layout: pipeline_layout,
            render_pass: vk::RenderPass::null(),
            p_depth_stencil_state: &depth_stencil_create_info,
            p_multisample_state: &multisampling_create_info,
            ..Default::default()
        };
        let fragment_shader_library = unsafe {
            self.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[fragment_shader_library_info],
                    None,
                )
                .expect("Failed to create graphics pipeline")
        };

        let library = fragment_shader_library[0];
        self.fragment_shader_libraries.push(library);
        library
    }

    /// Create fragment output interface library with specified color attachment formats
    fn create_fragment_output_interface_library(
        &mut self,
        color_attachment_formats: &[vk::Format],
        blend_attachments: &[vk::PipelineColorBlendAttachmentState],
        logic_op_enable: bool,
        logic_op: vk::LogicOp,
        blend_constants: [f32; 4],
    ) -> vk::Pipeline {
        let pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo {
            s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
            p_next: ptr::null(),
            view_mask: 0,
            color_attachment_count: color_attachment_formats.len() as u32,
            p_color_attachment_formats: color_attachment_formats.as_ptr(),
            depth_attachment_format: vk::Format::UNDEFINED,
            stencil_attachment_format: vk::Format::UNDEFINED,
            _marker: PhantomData,
        };
        let library_create_info = vk::GraphicsPipelineLibraryCreateInfoEXT {
            flags: vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE,
            p_next: &pipeline_rendering_create_info as *const _ as *const c_void,
            ..Default::default()
        };

        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: if logic_op_enable { vk::TRUE } else { vk::FALSE },
            logic_op,
            attachment_count: blend_attachments.len() as u32,
            p_attachments: blend_attachments.as_ptr(),
            blend_constants,
            _marker: PhantomData,
        };
        let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
            _marker: PhantomData,
        };
        let fragment_output_interface_info = vk::GraphicsPipelineCreateInfo {
            p_next: &library_create_info as *const _ as *const c_void,
            flags: vk::PipelineCreateFlags::LIBRARY_KHR
                | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT,
            render_pass: vk::RenderPass::null(),
            p_color_blend_state: &color_blend_state_create_info,
            p_multisample_state: &multisampling_create_info,
            ..Default::default()
        };
        let fragment_output_interface = unsafe {
            self.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[fragment_output_interface_info],
                    None,
                )
                .expect("Failed to create graphics pipeline")
        };

        let library = fragment_output_interface[0];
        self.fragment_output_interface_libraries.push(library);
        library
    }

    fn create_compute_pipeline_internal(
        &self,
        pipeline_layout: vk::PipelineLayout,
        compute_shader_path: &Path,
    ) -> vk::Pipeline {
        let compute_shader_code = Self::read_shader_code(compute_shader_path);
        let compute_shader_module = Self::create_shader_module(&self.device, &compute_shader_code);

        let main_function_name = CString::new("main").unwrap();

        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: compute_shader_module,
            p_name: main_function_name.as_ptr(),
            p_specialization_info: ptr::null(),
            _marker: PhantomData,
        };

        let pipeline_create_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage: shader_stage_create_info,
            layout: pipeline_layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            _marker: PhantomData,
        };

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .expect("Failed to create compute pipeline")
        };

        unsafe {
            self.device
                .destroy_shader_module(compute_shader_module, None);
        }

        pipelines[0]
    }

    fn read_shader_code(shader_path: &Path) -> Vec<u8> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        let spv_file = File::open(shader_path)
            .unwrap_or_else(|_| panic!("Failed to open shader file. {:?}", shader_path));
        let mut reader = BufReader::new(spv_file);
        let mut bytes_code = Vec::new();
        reader
            .read_to_end(&mut bytes_code)
            .unwrap_or_else(|_| panic!("Failed to read shader file. {:?}", shader_path));

        bytes_code
    }

    fn create_shader_module(device: &ash::Device, shader_code: &[u8]) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: shader_code.len(),
            p_code: shader_code.as_ptr() as *const u32,
            _marker: PhantomData,
        };

        unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create shader module")
        }
    }

    /// Explicitly clean up all pipelines and layouts
    /// This should be called before destroying the device
    pub fn cleanup(&mut self) {
        unsafe {
            // Clean up graphics pipelines
            for (pipeline, layout) in self.graphics_pipelines.drain(..).flatten() {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_pipeline_layout(layout, None);
            }
            self.free_graphics_pipeline_indices.clear();

            // Clean up compute pipelines
            for (pipeline, layout) in self.compute_pipelines.drain(..).flatten() {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_pipeline_layout(layout, None);
            }
            self.free_compute_pipeline_indices.clear();

            for &library in &self.vertex_input_interface_libraries {
                self.device.destroy_pipeline(library, None);
            }
            self.vertex_input_interface_libraries.clear();

            for &library in &self.pre_rasterization_shaders_libraries {
                self.device.destroy_pipeline(library, None);
            }
            self.pre_rasterization_shaders_libraries.clear();

            for &library in &self.fragment_shader_libraries {
                self.device.destroy_pipeline(library, None);
            }
            self.fragment_shader_libraries.clear();

            for &library in &self.fragment_output_interface_libraries {
                self.device.destroy_pipeline(library, None);
            }
            self.fragment_output_interface_libraries.clear();
        }
    }
}
