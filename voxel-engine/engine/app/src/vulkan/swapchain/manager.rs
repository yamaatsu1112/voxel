use std::marker::PhantomData;
use std::ptr;

use ash::vk;
use winit::window::Window;

use crate::vulkan::context::VulkanContext;

use super::support::{
    choose_swap_extent, choose_swap_present_mode, choose_swap_surface_format,
    query_swap_chain_support,
};

pub struct SwapchainManager {
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
}

impl SwapchainManager {
    pub fn new(context: &VulkanContext, window: &Window) -> Self {
        let swap_chain_support = query_swap_chain_support(
            &context.surface_loader,
            context.surface,
            context.physical_device,
        );

        let surface_format = choose_swap_surface_format(&swap_chain_support.formats);
        let present_mode = choose_swap_present_mode(&swap_chain_support.present_modes);
        let extent = choose_swap_extent(window, &swap_chain_support.capabilities);

        let mut image_count = swap_chain_support.capabilities.min_image_count + 1;
        if swap_chain_support.capabilities.max_image_count > 0
            && image_count > swap_chain_support.capabilities.max_image_count
        {
            image_count = swap_chain_support.capabilities.max_image_count;
        }

        let (image_sharing_mode, queue_family_index_count, queue_family_indices) =
            if context.family_indices.graphics_family != context.family_indices.present_family {
                (
                    vk::SharingMode::CONCURRENT,
                    2,
                    vec![
                        context.family_indices.graphics_family.unwrap(),
                        context.family_indices.present_family.unwrap(),
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, vec![])
            };

        let create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: context.surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            queue_family_index_count,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            pre_transform: swap_chain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            _marker: PhantomData,
        };

        let swapchain_loader = ash::khr::swapchain::Device::new(&context.instance, &context.device);
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("Failed to create swapchain")
        };

        let images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images")
        };

        let image_views =
            Self::create_swapchain_image_views(&context.device, &images, surface_format.format);

        Self {
            swapchain_loader,
            swapchain,
            images,
            format: surface_format.format,
            extent,
            image_views,
        }
    }

    pub fn recreate(&mut self, context: &VulkanContext, window: &Window) {
        let old_swapchain = self.swapchain;
        let old_image_views = std::mem::take(&mut self.image_views);
        self.images.clear();

        let swap_chain_support = query_swap_chain_support(
            &context.surface_loader,
            context.surface,
            context.physical_device,
        );

        let surface_format = choose_swap_surface_format(&swap_chain_support.formats);
        let present_mode = choose_swap_present_mode(&swap_chain_support.present_modes);
        let extent = choose_swap_extent(window, &swap_chain_support.capabilities);

        let mut image_count = swap_chain_support.capabilities.min_image_count + 1;
        if swap_chain_support.capabilities.max_image_count > 0
            && image_count > swap_chain_support.capabilities.max_image_count
        {
            image_count = swap_chain_support.capabilities.max_image_count;
        }

        let (image_sharing_mode, queue_family_index_count, queue_family_indices) =
            if context.family_indices.graphics_family != context.family_indices.present_family {
                (
                    vk::SharingMode::CONCURRENT,
                    2,
                    vec![
                        context.family_indices.graphics_family.unwrap(),
                        context.family_indices.present_family.unwrap(),
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, vec![])
            };

        let create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: context.surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            queue_family_index_count,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            pre_transform: swap_chain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain,
            _marker: PhantomData,
        };

        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("Failed to create swapchain")
        };

        let images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images")
        };

        let image_views =
            Self::create_swapchain_image_views(&context.device, &images, surface_format.format);

        self.swapchain = swapchain;
        self.images = images;
        self.format = surface_format.format;
        self.extent = extent;
        self.image_views = image_views;

        unsafe {
            // Graphics submissions are already synchronized by the per-frame fence before the next
            // frame slot is reused, so only the present queue still needs an explicit drain here.
            // This path did not produce errors without the wait in local testing, but keeping the
            // present queue idle before destroying the old swapchain makes that dependency explicit.
            if context.queues.present != context.queues.graphics {
                let _ = context.device.queue_wait_idle(context.queues.present);
            }
            for image_view in old_image_views {
                context.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }
    }

    pub fn get_format(&self) -> vk::Format {
        self.format
    }

    pub fn get_extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn get_image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    pub fn get_swapchain(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn get_swapchain_loader(&self) -> &ash::khr::swapchain::Device {
        &self.swapchain_loader
    }

    pub fn get_images(&self) -> &[vk::Image] {
        &self.images
    }

    pub fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            for &image_view in self.image_views.iter() {
                device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
        self.image_views.clear();
        self.images.clear();
    }

    fn create_swapchain_image_views(
        device: &ash::Device,
        swapchain_images: &[vk::Image],
        format: vk::Format,
    ) -> Vec<vk::ImageView> {
        let mut swapchain_image_views = vec![];

        for &image in swapchain_images.iter() {
            let image_view_create_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                _marker: PhantomData,
            };
            let image_view = unsafe {
                device
                    .create_image_view(&image_view_create_info, None)
                    .expect("Failed to create image view")
            };
            swapchain_image_views.push(image_view);
        }

        swapchain_image_views
    }
}
