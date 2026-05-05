include!(concat!(env!("OUT_DIR"), "/atlas.rs"));

use voxel_engine::builders::configs::descriptor_set_configs::UIDescriptorSet;
use voxel_engine::builders::configs::image_configs::{UITextureImage, UITextureImageView};
use voxel_engine::builders::configs::sampler_configs::UITextureSampler;
use voxel_engine::vk;
use voxel_engine::*;

pub fn setup_texture_atlas(world: &mut World) {
    let pixels: Vec<u8> = ATLAS_BYTES.to_vec();
    let width = ATLAS_WIDTH;
    let height = ATLAS_HEIGHT;

    let mut renderer = world.get_resource_mut::<Renderer>().unwrap();

    renderer.add_command(CreateImagesCommand::<UITextureImage>::new(Some((
        width, height,
    ))));
    renderer.add_command(TransitionLayoutCommand::<
        GraphicsTransferMutex,
        UITextureImage,
    >::new(
        0,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(
        UploadToImageCommand::<GraphicsTransferMutex, UITextureImage>::new(
            0, pixels, width, height, 1,
        ),
    );
    renderer.add_command(TransitionLayoutCommand::<
        GraphicsTransferMutex,
        UITextureImage,
    >::new(
        0,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(CreateImageViewsCommand::<UITextureImageView>::new());
    renderer.add_command(BindImageCommand::<
        UIDescriptorSet,
        UITextureImageView,
        UITextureSampler,
    >::new(0, 0, 0, 1));
}
