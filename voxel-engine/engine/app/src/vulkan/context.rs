use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char};
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;

use ash::vk;
use winit::window::Window;

use crate::vulkan::debug::{VALIDATION, ValidationInfo, populate_debug_messenger_create_info};
use crate::vulkan::platforms;
use crate::vulkan::swapchain::query_swap_chain_support;
use crate::vulkan::utils;

pub struct VulkanContext {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub debug_utils_loader: ash::ext::debug_utils::Instance,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub family_indices: QueueFamilyIndices,
    pub queues: Queues,
}

pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub present_family: Option<u32>,
    pub transfer_family: Option<u32>,
    pub compute_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
            && self.present_family.is_some()
            && self.transfer_family.is_some()
            && self.compute_family.is_some()
    }
}

struct QueueSelection {
    family_indices: QueueFamilyIndices,
    graphics_queue_index: u32,
    present_queue_index: u32,
    transfer_queue_index: u32,
    compute_queue_index: u32,
}

pub struct Queues {
    pub graphics: vk::Queue,
    pub compute: vk::Queue,
    pub present: vk::Queue,
    pub transfer: vk::Queue,
}

const DEVICE_EXTENSIONS: [&CStr; 4] = [
    vk::KHR_SWAPCHAIN_NAME,
    vk::KHR_DYNAMIC_RENDERING_NAME,
    vk::KHR_PIPELINE_LIBRARY_NAME,
    vk::EXT_GRAPHICS_PIPELINE_LIBRARY_NAME,
];

const ENABLE_EXTENSION_NAMES: [*const i8; 3] = [
    vk::KHR_SWAPCHAIN_NAME.as_ptr(),
    vk::KHR_PIPELINE_LIBRARY_NAME.as_ptr(),
    vk::EXT_GRAPHICS_PIPELINE_LIBRARY_NAME.as_ptr(),
];

impl VulkanContext {
    /// Create a new VulkanContext from a window.
    pub fn new(window: &Arc<Window>) -> Self {
        // Instance and surface
        let entry = unsafe { ash::Entry::load().unwrap() };
        let instance = create_instance(&entry);
        let (debug_utils_loader, debug_messenger) = setup_debug_utils(&entry, &instance);
        let (surface, surface_loader) = create_surface(&entry, &instance, window);

        // Physical device and logical device
        let physical_device = pick_physical_device(&instance, &surface_loader, surface);
        let (device, queue_selection) = create_logical_device(
            &instance,
            &surface_loader,
            surface,
            physical_device,
            &VALIDATION,
        );

        // Queues
        let queues = Queues {
            graphics: unsafe {
                device.get_device_queue(
                    queue_selection.family_indices.graphics_family.unwrap(),
                    queue_selection.graphics_queue_index,
                )
            },
            compute: unsafe {
                device.get_device_queue(
                    queue_selection.family_indices.compute_family.unwrap(),
                    queue_selection.compute_queue_index,
                )
            },
            present: unsafe {
                device.get_device_queue(
                    queue_selection.family_indices.present_family.unwrap(),
                    queue_selection.present_queue_index,
                )
            },
            transfer: unsafe {
                device.get_device_queue(
                    queue_selection.family_indices.transfer_family.unwrap(),
                    queue_selection.transfer_queue_index,
                )
            },
        };

        Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            debug_utils_loader,
            debug_messenger,
            physical_device,
            device,
            family_indices: queue_selection.family_indices,
            queues,
        }
    }
}

fn create_instance(entry: &ash::Entry) -> ash::Instance {
    if VALIDATION.is_enable && !check_validation_layer_support(entry) {
        panic!("Validation layers requested, but no available.");
    }

    let app_name = CString::new("Vulkan Test").unwrap();
    let engine_name = CString::new("Vulkan Engine").unwrap();
    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        p_application_name: app_name.as_ptr(),
        application_version: vk::make_api_version(0, 1, 0, 0),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_api_version(0, 1, 0, 0),
        api_version: vk::make_api_version(0, 1, 3, 0),
        _marker: PhantomData,
    };

    let debug_utils_create_info = populate_debug_messenger_create_info();

    let extension_names = platforms::required_extension_names();

    let required_validation_layer_raw_names: Vec<CString> = VALIDATION
        .required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let enable_layer_names: Vec<*const i8> = required_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: if VALIDATION.is_enable {
            &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void
        } else {
            ptr::null()
        },
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        pp_enabled_layer_names: if VALIDATION.is_enable {
            enable_layer_names.as_ptr()
        } else {
            ptr::null()
        },
        enabled_layer_count: if VALIDATION.is_enable {
            enable_layer_names.len()
        } else {
            0
        } as u32,
        pp_enabled_extension_names: extension_names.as_ptr(),
        enabled_extension_count: extension_names.len() as u32,
        _marker: PhantomData,
    };

    let instance: ash::Instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Failed to create instance")
    };

    instance
}

fn check_validation_layer_support(entry: &ash::Entry) -> bool {
    // if support validation layer, then return true

    let layer_properties = unsafe {
        entry
            .enumerate_instance_layer_properties()
            .expect("failed to enumerate instance Layers Properties")
    };

    if layer_properties.is_empty() {
        eprintln!("No avaliable layers.");
        return false;
    }

    for required_layer_name in VALIDATION.required_validation_layers.iter() {
        let mut is_layer_found = false;

        for layer_property in layer_properties.iter() {
            let test_layer_name = utils::vk_to_string(&layer_property.layer_name);
            if (*required_layer_name) == test_layer_name {
                is_layer_found = true;
                break;
            }
        }

        if !is_layer_found {
            return false;
        }
    }

    true
}

fn setup_debug_utils(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> (ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT) {
    let debug_utils_loader = ash::ext::debug_utils::Instance::new(entry, instance);

    if !VALIDATION.is_enable {
        (debug_utils_loader, ash::vk::DebugUtilsMessengerEXT::null())
    } else {
        let messenger_ci = populate_debug_messenger_create_info();

        let utils_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&messenger_ci, None)
                .expect("Debug Utils Callback")
        };

        (debug_utils_loader, utils_messenger)
    }
}

fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> (vk::SurfaceKHR, ash::khr::surface::Instance) {
    let surface = unsafe {
        platforms::create_surface(entry, instance, window).expect("Failed to create surface")
    };
    let surface_loader = ash::khr::surface::Instance::new(entry, instance);
    (surface, surface_loader)
}

fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> vk::PhysicalDevice {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate Physical Devices.")
    };

    println!(
        "{} devices found with vulkan support.",
        physical_devices.len()
    );

    let mut result = None;
    for &physical_device in physical_devices.iter() {
        if is_physical_device_suitable(instance, surface_loader, surface, physical_device)
            && result.is_none()
        {
            result = Some(physical_device)
        }
    }

    match result {
        None => panic!("Failed to find a suitable GPU."),
        Some(physical_device) => {
            let device_properties =
                unsafe { instance.get_physical_device_properties(physical_device) };
            let device_name = utils::vk_to_string(&device_properties.device_name);
            println!("\tPicked Device Name: {}", device_name);
            physical_device
        }
    }
}

fn is_physical_device_suitable(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let extension_support = check_device_extension_support(instance, physical_device);

    let mut swap_chain_adequate = false;
    if extension_support {
        let swap_chain_support = query_swap_chain_support(surface_loader, surface, physical_device);

        swap_chain_adequate =
            !swap_chain_support.formats.is_empty() && !swap_chain_support.present_modes.is_empty();
    }

    let indices = find_queue_family(instance, surface_loader, surface, physical_device);

    extension_support && swap_chain_adequate && indices.is_complete()
}

fn find_queue_family(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> QueueFamilyIndices {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let present_support: Vec<bool> = queue_families
        .iter()
        .enumerate()
        .map(|(index, _)| {
            unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface,
                )
            }
            .unwrap()
        })
        .collect();

    select_queue_family_indices(&queue_families, &present_support)
}

fn select_queue_family_indices(
    queue_families: &[vk::QueueFamilyProperties],
    present_support: &[bool],
) -> QueueFamilyIndices {
    let mut queue_family_indices = QueueFamilyIndices {
        graphics_family: None,
        present_family: None,
        transfer_family: None,
        compute_family: None,
    };

    for (index, queue_family) in queue_families.iter().enumerate() {
        let family_index = index as u32;

        if queue_family_indices.graphics_family.is_none()
            && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        {
            queue_family_indices.graphics_family = Some(family_index);
        }

        if queue_family_indices.present_family.is_none() && present_support[index] {
            queue_family_indices.present_family = Some(family_index);
        }
    }

    if let Some(graphics_family) = queue_family_indices.graphics_family {
        queue_family_indices.transfer_family = Some(graphics_family);
        let graphics_queue_family = &queue_families[graphics_family as usize];
        if graphics_queue_family
            .queue_flags
            .contains(vk::QueueFlags::COMPUTE)
            && graphics_queue_family.queue_count >= 2
        {
            queue_family_indices.compute_family = Some(graphics_family);
        }
    }

    queue_family_indices
}

fn build_queue_selection(
    queue_families: &[vk::QueueFamilyProperties],
    family_indices: QueueFamilyIndices,
) -> Option<QueueSelection> {
    let graphics_family = family_indices.graphics_family?;
    let present_family = family_indices.present_family?;
    let transfer_family = family_indices.transfer_family?;
    let compute_family = family_indices.compute_family?;

    let graphics_queue_index = 0;
    let present_queue_index = 0;
    let transfer_queue_index = 0;
    let compute_queue_index = if compute_family == graphics_family {
        let graphics_queue_family = &queue_families[graphics_family as usize];
        if graphics_queue_family
            .queue_flags
            .contains(vk::QueueFlags::COMPUTE)
            && graphics_queue_family.queue_count >= 2
        {
            1
        } else {
            return None;
        }
    } else {
        0
    };

    Some(QueueSelection {
        family_indices: QueueFamilyIndices {
            graphics_family: Some(graphics_family),
            present_family: Some(present_family),
            transfer_family: Some(transfer_family),
            compute_family: Some(compute_family),
        },
        graphics_queue_index,
        present_queue_index,
        transfer_queue_index,
        compute_queue_index,
    })
}

fn check_device_extension_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let available_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .expect("Failed to enumerate device extension properties")
    };

    let mut available_extension_names = Vec::new();
    for extension in available_extensions.iter() {
        let extension_name = utils::vk_to_string(&extension.extension_name);
        available_extension_names.push(extension_name);
    }

    let mut required_extensions = HashSet::new();
    for extension in DEVICE_EXTENSIONS.iter() {
        required_extensions.insert(utils::cstr_to_string(extension));
    }

    for extension in available_extension_names.iter() {
        required_extensions.remove(extension);
    }

    required_extensions.is_empty()
}

fn create_logical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    validation: &ValidationInfo,
) -> (ash::Device, QueueSelection) {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let family_indices = find_queue_family(instance, surface_loader, surface, physical_device);
    let queue_selection = build_queue_selection(&queue_families, family_indices)
        .expect("Failed to build queue selection");

    let mut queue_counts: HashMap<u32, u32> = HashMap::new();
    for (family, queue_index) in [
        (
            queue_selection.family_indices.graphics_family.unwrap(),
            queue_selection.graphics_queue_index,
        ),
        (
            queue_selection.family_indices.present_family.unwrap(),
            queue_selection.present_queue_index,
        ),
        (
            queue_selection.family_indices.transfer_family.unwrap(),
            queue_selection.transfer_queue_index,
        ),
        (
            queue_selection.family_indices.compute_family.unwrap(),
            queue_selection.compute_queue_index,
        ),
    ] {
        queue_counts
            .entry(family)
            .and_modify(|count| *count = (*count).max(queue_index + 1))
            .or_insert(queue_index + 1);
    }

    let queue_priorities = [1.0_f32, 1.0_f32];
    let mut queue_create_infos = Vec::new();
    for (&queue_family, &queue_count) in &queue_counts {
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_family,
            p_queue_priorities: queue_priorities.as_ptr(),
            queue_count,
            _marker: PhantomData,
        };
        queue_create_infos.push(queue_create_info);
    }

    let physical_device_features = vk::PhysicalDeviceFeatures {
        sampler_anisotropy: vk::TRUE,
        logic_op: vk::TRUE,
        ..Default::default()
    };

    let required_validation_layer_raw_names: Vec<CString> = validation
        .required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let enable_layer_names: Vec<*const c_char> = required_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let enable_extension_names = ENABLE_EXTENSION_NAMES.as_slice();

    let mut pipeline_library_feature = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT {
        s_type: vk::StructureType::PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
        p_next: ptr::null_mut(),
        graphics_pipeline_library: vk::TRUE,
        _marker: PhantomData,
    };
    let mut vulkan11_features = vk::PhysicalDeviceVulkan11Features {
        s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        p_next: &mut pipeline_library_feature as *mut _ as *mut c_void,
        shader_draw_parameters: vk::TRUE,
        ..Default::default()
    };
    let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features {
        s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        p_next: &mut vulkan11_features as *mut _ as *mut c_void,
        timeline_semaphore: vk::TRUE,
        ..Default::default()
    };
    let dynamic_rendering_feature = vk::PhysicalDeviceDynamicRenderingFeatures {
        s_type: vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
        p_next: &mut vulkan12_features as *mut _ as *mut c_void,
        dynamic_rendering: vk::TRUE,
        _marker: PhantomData,
    };

    #[allow(deprecated)]
    let device_create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next: &dynamic_rendering_feature as *const _ as *const c_void,
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_layer_count: if validation.is_enable {
            enable_layer_names.len()
        } else {
            0
        } as u32,
        pp_enabled_layer_names: if validation.is_enable {
            enable_layer_names.as_ptr()
        } else {
            ptr::null()
        },
        enabled_extension_count: enable_extension_names.len() as u32,
        pp_enabled_extension_names: if enable_extension_names.is_empty() {
            ptr::null()
        } else {
            enable_extension_names.as_ptr()
        },
        p_enabled_features: &physical_device_features,
        _marker: PhantomData,
    };

    let device: ash::Device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create logical device")
    };

    (device, queue_selection)
}
