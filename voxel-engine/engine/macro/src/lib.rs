mod component;
mod define_buffer;
mod define_descriptor_pool;
mod define_descriptor_set;
mod define_image;
mod define_image_view;
mod define_pipeline;
mod define_reusable_command_buffer;
mod define_sampler;
mod query_item;
mod scenes;

use proc_macro::TokenStream;

/// Macro to implement QueryItem for tuples of references
///
/// Example usage:
/// ```ignore
/// impl_query_item_tuple!((&T1, &T2));
/// impl_query_item_tuple!((&T1, &mut T2));
/// impl_query_item_tuple!((&mut T1, &mut T2));
/// ```
#[proc_macro]
pub fn impl_query_item_tuple(input: TokenStream) -> TokenStream {
    query_item::impl_query_item_tuple_impl(input)
}

#[proc_macro_derive(Component)]
pub fn derive_component(input: TokenStream) -> TokenStream {
    component::derive_component_impl(input)
}

#[proc_macro_derive(Scenes)]
pub fn derive_scenes(input: TokenStream) -> TokenStream {
    scenes::derive_scenes_impl(input)
}

/// Define a buffer marker type with static ID storage.
///
/// Generates: struct definition and `BufferMarker` impl with `ids_slice()`
/// returning a static `AtomicUsize` array for thread-safe BufferId access.
#[proc_macro]
pub fn define_buffer(input: TokenStream) -> TokenStream {
    define_buffer::define_buffer_impl(input)
}

/// Define a sampler marker type with static ID storage.
///
/// Generates: struct definition and `SamplerMarker` impl with `id()`
/// returning a static `AtomicUsize` for thread-safe SamplerId access.
#[proc_macro]
pub fn define_sampler(input: TokenStream) -> TokenStream {
    define_sampler::define_sampler_impl(input)
}

/// Define a descriptor pool marker type with static ID storage.
#[proc_macro]
pub fn define_descriptor_pool(input: TokenStream) -> TokenStream {
    define_descriptor_pool::define_descriptor_pool_impl(input)
}

/// Define a descriptor set marker type with static ID storage.
#[proc_macro]
pub fn define_descriptor_set(input: TokenStream) -> TokenStream {
    define_descriptor_set::define_descriptor_set_impl(input)
}

/// Define a graphics pipeline marker type with static ID storage.
///
/// Generates: struct definition and `GraphicsPipelineMarker` impl with `id()`
/// returning a static `AtomicUsize` for thread-safe GraphicsPipelineId access.
#[proc_macro]
pub fn define_graphics_pipeline(input: TokenStream) -> TokenStream {
    define_pipeline::define_graphics_pipeline_impl(input)
}

/// Define a compute pipeline marker type with static ID storage.
///
/// Generates: struct definition and `ComputePipelineMarker` impl with `id()`
/// returning a static `AtomicUsize` for thread-safe ComputePipelineId access.
#[proc_macro]
pub fn define_compute_pipeline(input: TokenStream) -> TokenStream {
    define_pipeline::define_compute_pipeline_impl(input)
}

/// Define an image marker type with static ID storage.
///
/// Generates: struct definition and `ImageMarker` impl with `ids_slice()`
/// returning a static `AtomicUsize` array for thread-safe ImageId access.
#[proc_macro]
pub fn define_image(input: TokenStream) -> TokenStream {
    define_image::define_image_impl(input)
}

/// Define an image view marker type with static ID storage.
///
/// Generates: struct definition and `ImageViewMarker` impl with `ids_slice()`
/// returning a static `AtomicUsize` array for thread-safe ImageViewId access.
#[proc_macro]
pub fn define_image_view(input: TokenStream) -> TokenStream {
    define_image_view::define_image_view_impl(input)
}

/// Define a reusable command buffer marker type with static index storage.
#[proc_macro]
pub fn define_reusable_command_buffer(input: TokenStream) -> TokenStream {
    define_reusable_command_buffer::define_reusable_command_buffer_impl(input)
}
