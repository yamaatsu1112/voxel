use image::{GenericImage, ImageBuffer, Rgba};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct AssetsConfig {
    textures: BTreeMap<String, String>,
}

type ImageRgba = ImageBuffer<Rgba<u8>, Vec<u8>>;

fn main() {
    if std::env::var("CI").is_err() {
        compile_shaders();
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let assets_toml = manifest_dir.join("assets.toml");
    println!("cargo:rerun-if-changed={}", assets_toml.display());

    let config_str = std::fs::read_to_string(&assets_toml).expect("Failed to read assets.toml");
    let config: AssetsConfig = toml::from_str(&config_str).expect("Failed to parse assets.toml");

    let mut images: Vec<(String, ImageRgba)> = Vec::new();
    for (name, rel_path) in &config.textures {
        let abs_path = manifest_dir.join(rel_path);
        println!("cargo:rerun-if-changed={}", abs_path.display());
        let img = image::open(&abs_path)
            .unwrap_or_else(|e| panic!("Failed to open texture '{}': {}", abs_path.display(), e))
            .into_rgba8();
        images.push((name.clone(), img));
    }

    // Pack textures side by side (sorted by name for determinism)
    let atlas_height = images
        .iter()
        .map(|(_, img)| img.height())
        .max()
        .unwrap_or(1);
    let atlas_width: u32 = images.iter().map(|(_, img)| img.width()).sum();

    let mut atlas = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(atlas_width, atlas_height);
    let mut x_offset: u32 = 0;

    // Build texture region info (sorted by name)
    let mut regions: Vec<(String, u32, u32, u32)> = Vec::new(); // (name, x_offset, w, h)
    for (name, img) in &images {
        let w = img.width();
        let h = img.height();
        atlas
            .copy_from(img, x_offset, 0)
            .expect("Failed to copy image into atlas");
        regions.push((name.clone(), x_offset, w, h));
        x_offset += w;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Write atlas.bin (raw RGBA bytes)
    let atlas_bin_path = out_dir.join("atlas.bin");
    let raw_bytes: Vec<u8> = atlas.into_raw();
    std::fs::write(&atlas_bin_path, &raw_bytes).expect("Failed to write atlas.bin");

    // Write atlas.rs
    let atlas_rs_path = out_dir.join("atlas.rs");
    let mut rs = String::new();

    rs.push_str("#[allow(dead_code)]\n");
    rs.push_str("pub struct TextureRegion {\n");
    rs.push_str("    pub u_offset: f32,\n");
    rs.push_str("    pub v_offset: f32,\n");
    rs.push_str("    pub u_size: f32,\n");
    rs.push_str("    pub v_size: f32,\n");
    rs.push_str("}\n\n");

    for (name, x_off, w, h) in &regions {
        let u_offset = *x_off as f32 / atlas_width as f32;
        let v_offset = 0.0_f32;
        let u_size = *w as f32 / atlas_width as f32;
        let v_size = *h as f32 / atlas_height as f32;
        let const_name = name.to_uppercase();
        rs.push_str(&format!(
            "#[allow(dead_code)]\npub const {}: TextureRegion = TextureRegion {{ u_offset: {:.6}f32, v_offset: {:.6}f32, u_size: {:.6}f32, v_size: {:.6}f32 }};\n",
            const_name, u_offset, v_offset, u_size, v_size
        ));
    }

    rs.push('\n');
    rs.push_str("pub const ATLAS_BYTES: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/atlas.bin\"));\n");
    rs.push_str(&format!("pub const ATLAS_WIDTH: u32 = {};\n", atlas_width));
    rs.push_str(&format!(
        "pub const ATLAS_HEIGHT: u32 = {};\n",
        atlas_height
    ));

    std::fs::write(&atlas_rs_path, &rs).expect("Failed to write atlas.rs");

    // =========================================================================
    // Generate 3D voxel texture (8x8x2048, RGBA8)
    // 256 tiles of 8x8x8 packed along Z axis
    // =========================================================================
    generate_voxel_3d_texture(&out_dir);
}

/// Generate a 3D texture with procedural patterns for each voxel type.
/// Texture dimensions: 8 x 8 x 2048 (256 tiles of 8x8x8)
/// Format: RGBA8 (4 bytes per texel)
fn generate_voxel_3d_texture(out_dir: &Path) {
    const TILE_SIZE: usize = 8;
    const NUM_TILES: usize = 256;
    const TEX_WIDTH: usize = TILE_SIZE;
    const TEX_HEIGHT: usize = TILE_SIZE;
    const TEX_DEPTH: usize = TILE_SIZE * NUM_TILES;
    const TEXEL_SIZE: usize = 4; // RGBA8

    let mut data = vec![0u8; TEX_WIDTH * TEX_HEIGHT * TEX_DEPTH * TEXEL_SIZE];

    for tile_id in 0..NUM_TILES {
        let z_base = tile_id * TILE_SIZE;
        for lz in 0..TILE_SIZE {
            for ly in 0..TILE_SIZE {
                for lx in 0..TILE_SIZE {
                    let gz = z_base + lz;
                    let offset = ((gz * TEX_HEIGHT + ly) * TEX_WIDTH + lx) * TEXEL_SIZE;

                    let (r, g, b, a) = generate_tile_color(tile_id, lx, ly, lz);
                    data[offset] = r;
                    data[offset + 1] = g;
                    data[offset + 2] = b;
                    data[offset + 3] = a;
                }
            }
        }
    }

    let voxel_tex_path = out_dir.join("voxel_3d_texture.bin");
    std::fs::write(&voxel_tex_path, &data).expect("Failed to write voxel_3d_texture.bin");

    // Write voxel_texture.rs with constants
    let voxel_tex_rs_path = out_dir.join("voxel_texture.rs");
    let rs = format!(
        "pub const VOXEL_3D_TEXTURE_BYTES: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/voxel_3d_texture.bin\"));\n\
         pub const VOXEL_3D_TEXTURE_WIDTH: u32 = {};\n\
         pub const VOXEL_3D_TEXTURE_HEIGHT: u32 = {};\n\
         pub const VOXEL_3D_TEXTURE_DEPTH: u32 = {};\n",
        TEX_WIDTH, TEX_HEIGHT, TEX_DEPTH,
    );
    std::fs::write(&voxel_tex_rs_path, &rs).expect("Failed to write voxel_texture.rs");
}

/// Generate color for a texel within a tile.
/// Each tile_id gets a distinct procedural pattern.
fn generate_tile_color(tile_id: usize, lx: usize, ly: usize, lz: usize) -> (u8, u8, u8, u8) {
    match tile_id {
        0 => {
            // ID 0: Air (transparent, should not be rendered)
            (0, 0, 0, 0)
        }
        1 => {
            // ID 1: Stone - gray with subtle noise
            let base = 128u8;
            let noise = ((lx * 17 + ly * 31 + lz * 13) % 20) as u8;
            (
                base.wrapping_add(noise),
                base.wrapping_add(noise),
                base.wrapping_add(noise),
                255,
            )
        }
        _ => {
            // All other IDs: generate color from ID using hue rotation
            let hue = (tile_id * 137) % 360; // Golden angle-ish distribution
            let (r, g, b) = hsv_to_rgb(hue as f32, 0.7, 0.8);
            // Add subtle checkerboard pattern
            let checker = ((lx / 4) + (ly / 4) + (lz / 4)).is_multiple_of(2);
            let factor = if checker { 1.0f32 } else { 0.85 };
            (
                (r as f32 * factor) as u8,
                (g as f32 * factor) as u8,
                (b as f32 * factor) as u8,
                255,
            )
        }
    }
}

fn compile_shaders() {
    use std::process::Command;

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shader_dir = manifest_dir.join("src/shaders");
    println!("cargo:rerun-if-changed={}", shader_dir.display());

    // Engine shader directory for include path (svo.slang etc.)
    let engine_shader_dir = manifest_dir
        .join("..")
        .join("engine")
        .join("voxel")
        .join("src")
        .join("shaders");

    let shaders: &[(&str, &str)] = &[];

    // Find slangc
    let slangc = env::var_os("PATH")
        .and_then(|path_var| {
            env::split_paths(&path_var)
                .map(|dir| dir.join("slangc"))
                .find(|candidate| candidate.is_file())
        })
        .expect("slangc not found in PATH");

    for &(src, dst) in shaders {
        let src_path = shader_dir.join(src);
        let dst_path = shader_dir.join(dst);

        println!(
            "cargo:warning=Compiling shader {} -> {}",
            src_path.display(),
            dst_path.display()
        );

        let output = Command::new(&slangc)
            .arg(&src_path)
            .arg("-target")
            .arg("spirv")
            .arg("-profile")
            .arg("glsl_450")
            .arg("-entry")
            .arg("main")
            .arg("-stage")
            .arg("compute")
            .arg("-I")
            .arg(&engine_shader_dir)
            .arg("-O2")
            .arg("-o")
            .arg(&dst_path)
            .output()
            .unwrap_or_else(|e| panic!("Failed to spawn slangc for {}: {e}", src_path.display()));

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            panic!(
                "slangc failed for {} with status {}\nstderr: {}\nstdout: {}",
                src_path.display(),
                output.status,
                stderr,
                stdout
            );
        }
    }
}

/// Convert HSV to RGB (h in 0-360, s and v in 0-1)
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}
