use ash::vk;
use winit::window::Window;

pub struct SwapChainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub fn query_swap_chain_support(
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> SwapChainSupportDetails {
    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    };
    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) };
    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
    };

    SwapChainSupportDetails {
        capabilities: capabilities.unwrap(),
        formats: formats.unwrap(),
        present_modes: present_modes.unwrap(),
    }
}

pub fn choose_swap_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    for format in formats.iter() {
        if format.format == vk::Format::B8G8R8A8_SRGB
            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return *format;
        }
    }

    formats[0]
}

pub fn choose_swap_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    for mode in present_modes.iter() {
        if *mode == vk::PresentModeKHR::MAILBOX {
            return *mode;
        }
    }

    vk::PresentModeKHR::FIFO
}

pub fn choose_swap_extent(
    window: &Window,
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }

    let size = window.inner_size();
    vk::Extent2D {
        width: size.width,
        height: size.height,
    }
}
