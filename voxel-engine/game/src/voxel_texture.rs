include!(concat!(env!("OUT_DIR"), "/voxel_texture.rs"));

use voxel_engine::builders::configs::descriptor_set_configs::MainStaticDescriptorSet;
use voxel_engine::builders::configs::image_configs::{
    VoxelTexture3DImage, VoxelTexture3DImageView,
};
use voxel_engine::builders::configs::sampler_configs::VoxelTexture3DSampler;
use voxel_engine::vk;
use voxel_engine::*;

pub fn setup_voxel_3d_texture(world: &mut World) {
    let pixels: Vec<u8> = VOXEL_3D_TEXTURE_BYTES.to_vec();
    let expected_size =
        (VOXEL_3D_TEXTURE_WIDTH * VOXEL_3D_TEXTURE_HEIGHT * VOXEL_3D_TEXTURE_DEPTH * 4) as usize;
    assert_eq!(pixels.len(), expected_size);

    let mut renderer = world.get_resource_mut::<Renderer>().unwrap();

    renderer.add_command(CreateImagesCommand::<VoxelTexture3DImage>::new(None));
    renderer.add_command(TransitionLayoutCommand::<
        GraphicsTransferMutex,
        VoxelTexture3DImage,
    >::new(
        0,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(UploadToImageCommand::<
        GraphicsTransferMutex,
        VoxelTexture3DImage,
    >::new(
        0,
        pixels,
        VOXEL_3D_TEXTURE_WIDTH,
        VOXEL_3D_TEXTURE_HEIGHT,
        VOXEL_3D_TEXTURE_DEPTH,
    ));
    renderer.add_command(TransitionLayoutCommand::<
        GraphicsTransferMutex,
        VoxelTexture3DImage,
    >::new(
        0,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
    ));
    renderer.add_command(CreateImageViewsCommand::<VoxelTexture3DImageView>::new());
    renderer.add_command(BindImageCommand::<
        MainStaticDescriptorSet,
        VoxelTexture3DImageView,
        VoxelTexture3DSampler,
    >::new(0, 0, 0, 1));
}
