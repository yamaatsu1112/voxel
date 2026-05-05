use ash::vk;

// required extension ------------------------------------------------------
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        vk::KHR_SURFACE_NAME.as_ptr(),
        vk::KHR_XLIB_SURFACE_NAME.as_ptr(),
        vk::KHR_WAYLAND_SURFACE_NAME.as_ptr(),
        vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
    ]
}
// ------------------------------------------------------------------------

// create surface ---------------------------------------------------------
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::marker::PhantomData;
    use std::os::raw::c_void;
    use std::ptr;
    use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
    use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

    let display_handle = window.display_handle().unwrap().as_raw();
    match display_handle {
        RawDisplayHandle::Xlib(display_handle) => {
            let window_handle = window.window_handle().unwrap().as_raw();
            let window = match window_handle {
                RawWindowHandle::Xlib(x11_window) => x11_window.window,
                _ => panic!("Unsupported window handle"),
            };

            let xlib_create_info = vk::XlibSurfaceCreateInfoKHR {
                s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
                p_next: ptr::null(),
                flags: Default::default(),
                window: window as vk::Window,
                dpy: display_handle.display.unwrap().as_ptr() as *mut vk::Display,
                _marker: PhantomData,
            };
            let xlib_surface_loader = ash::khr::xlib_surface::Instance::new(entry, instance);
            unsafe { xlib_surface_loader.create_xlib_surface(&xlib_create_info, None) }
        }
        RawDisplayHandle::Wayland(display_handle) => {
            let window_handle = window.window_handle().unwrap().as_raw();
            let surface = match window_handle {
                RawWindowHandle::Wayland(wayland_window) => wayland_window.surface,
                _ => panic!("Unsupported window handle"),
            };

            let create_info = vk::WaylandSurfaceCreateInfoKHR {
                s_type: vk::StructureType::WAYLAND_SURFACE_CREATE_INFO_KHR,
                p_next: ptr::null(),
                flags: Default::default(),
                display: display_handle.display.as_ptr() as *mut vk::Display,
                surface: surface.as_ptr() as *mut c_void,
                _marker: PhantomData,
            };
            let wayland_surface_loader = ash::khr::wayland_surface::Instance::new(entry, instance);
            unsafe { wayland_surface_loader.create_wayland_surface(&create_info, None) }
        }
        _ => panic!("Unsupported display handle"),
    }
}
// ------------------------------------------------------------------------
