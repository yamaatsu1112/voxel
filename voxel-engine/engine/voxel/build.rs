use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

enum ShaderCompiler {
    Slangc(SlangStage),
}

#[derive(Clone, Copy)]
enum SvoShaderVariant {
    Individual,
    Grouped,
    Sv64Individual,
    Sv64Grouped,
}

impl SvoShaderVariant {
    fn output_suffix(self) -> &'static str {
        match self {
            SvoShaderVariant::Individual => "individual",
            SvoShaderVariant::Grouped => "grouped",
            SvoShaderVariant::Sv64Individual => "sv64_individual",
            SvoShaderVariant::Sv64Grouped => "sv64_grouped",
        }
    }

    fn shader_dir(self) -> &'static str {
        match self {
            SvoShaderVariant::Individual => "src/voxel/svo/individual/shaders",
            SvoShaderVariant::Grouped => "src/voxel/svo/grouped/shaders",
            SvoShaderVariant::Sv64Individual => "src/voxel/svo/sv64_individual/shaders",
            SvoShaderVariant::Sv64Grouped => "src/voxel/svo/sv64_grouped/shaders",
        }
    }
}

#[derive(Clone, Copy)]
enum SlangStage {
    Vertex,
    Fragment,
    Compute,
}

impl SlangStage {
    fn as_str(&self) -> &'static str {
        match self {
            SlangStage::Vertex => "vertex",
            SlangStage::Fragment => "fragment",
            SlangStage::Compute => "compute",
        }
    }
}

struct ShaderMapping {
    src: &'static str,
    dst: &'static str,
    compiler: ShaderCompiler,
    uses_svo: bool,
}

const COMMON_SHADER_MAPPINGS: &[ShaderMapping] = &[
    ShaderMapping {
        src: "vert.slang",
        dst: "vert.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Vertex),
        uses_svo: false,
    },
    ShaderMapping {
        src: "frag.slang",
        dst: "frag.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Fragment),
        uses_svo: true,
    },
    ShaderMapping {
        src: "prepare_dispatch.slang",
        dst: "prepare_dispatch.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: false,
    },
    ShaderMapping {
        src: "gbuffer_write_frag.slang",
        dst: "gbuffer_write_frag.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Fragment),
        uses_svo: true,
    },
    ShaderMapping {
        src: "gbuffer_color_write_frag.slang",
        dst: "gbuffer_color_write_frag.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Fragment),
        uses_svo: true,
    },
    ShaderMapping {
        src: "dynamic_gbuffer_write_frag.slang",
        dst: "dynamic_gbuffer_write_frag.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Fragment),
        uses_svo: true,
    },
    ShaderMapping {
        src: "dynamic_gbuffer_color_write_frag.slang",
        dst: "dynamic_gbuffer_color_write_frag.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Fragment),
        uses_svo: true,
    },
    ShaderMapping {
        src: "gbuffer_id_write_frag.slang",
        dst: "gbuffer_id_write_frag.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Fragment),
        uses_svo: true,
    },
    ShaderMapping {
        src: "dynamic_gbuffer_id_write_frag.slang",
        dst: "dynamic_gbuffer_id_write_frag.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Fragment),
        uses_svo: true,
    },
    ShaderMapping {
        src: "frag_3dtex.slang",
        dst: "frag_3dtex.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Fragment),
        uses_svo: true,
    },
    ShaderMapping {
        src: "raycast_compute.slang",
        dst: "raycast_compute.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "collision_compute.slang",
        dst: "collision_compute.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
];

const EDIT_SHADER_MAPPINGS: &[ShaderMapping] = &[
    ShaderMapping {
        src: "voxel.slang",
        dst: "voxel",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "voxel_destroy.slang",
        dst: "voxel_destroy",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "dynamic_voxel.slang",
        dst: "dynamic_voxel",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
];

const INDIVIDUAL_SPLIT_EDIT_SHADER_MAPPINGS: &[ShaderMapping] = &[
    ShaderMapping {
        src: "edit/voxel_collect_materialize.slang",
        dst: "voxel_collect_materialize_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_scan_node_offsets.slang",
        dst: "voxel_scan_node_offsets_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_scan_leaf_offsets.slang",
        dst: "voxel_scan_leaf_offsets_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_snapshot_allocation.slang",
        dst: "voxel_snapshot_allocation_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_initialize_materialized.slang",
        dst: "voxel_initialize_materialized_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_link_materialized.slang",
        dst: "voxel_link_materialized_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_commit_materialized.slang",
        dst: "voxel_commit_materialized_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_apply_place.slang",
        dst: "voxel_apply_place_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_apply_destroy.slang",
        dst: "voxel_apply_destroy_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_collect_free.slang",
        dst: "voxel_collect_free_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_scan_free_offsets.slang",
        dst: "voxel_scan_free_offsets_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/voxel_free_batch.slang",
        dst: "voxel_free_batch_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
    ShaderMapping {
        src: "edit/dynamic_voxel_clear.slang",
        dst: "dynamic_voxel_clear_individual.spv",
        compiler: ShaderCompiler::Slangc(SlangStage::Compute),
        uses_svo: true,
    },
];

fn main() {
    let active_variant = detect_current_svo_variant(Path::new("src/voxel/svo/svo_config.rs"))
        .unwrap_or_else(|error| panic!("Failed to detect active SVO shader variant: {error}"));
    println!(
        "cargo:rustc-env=VOXEL_ACTIVE_SHADER_SUFFIX={}",
        active_variant.output_suffix()
    );

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let shader_dir = out_dir.join("shaders");
    println!("cargo:rustc-env=VOXEL_SHADER_DIR={}", shader_dir.display());

    if env::var("CI").is_ok() {
        return;
    }

    fs::create_dir_all(&shader_dir)
        .unwrap_or_else(|error| panic!("Failed to create {}: {error}", shader_dir.display()));
    let shader_src_dir = PathBuf::from("src/shaders");
    let grouped_shader_dir = PathBuf::from(SvoShaderVariant::Grouped.shader_dir());
    let individual_shader_dir = PathBuf::from(SvoShaderVariant::Individual.shader_dir());
    let sv64_individual_shader_dir = PathBuf::from(SvoShaderVariant::Sv64Individual.shader_dir());
    let sv64_grouped_shader_dir = PathBuf::from(SvoShaderVariant::Sv64Grouped.shader_dir());
    println!("cargo:rerun-if-changed={}", shader_src_dir.display());
    println!("cargo:rerun-if-changed={}", grouped_shader_dir.display());
    println!("cargo:rerun-if-changed={}", individual_shader_dir.display());
    println!(
        "cargo:rerun-if-changed={}",
        sv64_individual_shader_dir.display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        sv64_grouped_shader_dir.display()
    );
    println!("cargo:rerun-if-changed=src/voxel/svo/svo_config.rs");

    if let Err(error) = compile_shaders(&shader_src_dir, &shader_dir) {
        panic!("Failed to compile shaders: {error}");
    }
}

fn compile_shaders(shader_src_dir: &Path, shader_dst_dir: &Path) -> Result<(), String> {
    let compiler_path = which_compiler("slangc")
        .ok_or_else(|| "slangc not found in PATH. Please install Slang compiler.".to_string())?;
    let svo_shader_dir = PathBuf::from("src/voxel/svo");

    for mapping in COMMON_SHADER_MAPPINGS {
        if mapping.uses_svo {
            for variant in [
                SvoShaderVariant::Grouped,
                SvoShaderVariant::Individual,
                SvoShaderVariant::Sv64Individual,
                SvoShaderVariant::Sv64Grouped,
            ] {
                let variant_shader_dir = PathBuf::from(variant.shader_dir());
                let src_path = variant_shader_dir.join(mapping.src);
                let dst_path = shader_dst_dir.join(suffixed_output_name(mapping.dst, variant));
                compile_with_slangc(
                    &compiler_path,
                    &[&variant_shader_dir, shader_src_dir, &svo_shader_dir],
                    &src_path,
                    &dst_path,
                    stage(mapping),
                )?;
            }
        } else {
            let src_path = shader_src_dir.join(mapping.src);
            let dst_path = shader_dst_dir.join(mapping.dst);
            compile_with_slangc(
                &compiler_path,
                &[shader_src_dir, &svo_shader_dir],
                &src_path,
                &dst_path,
                stage(mapping),
            )?;
        }
    }

    for variant in [
        SvoShaderVariant::Grouped,
        SvoShaderVariant::Sv64Individual,
        SvoShaderVariant::Sv64Grouped,
    ] {
        let variant_shader_dir = PathBuf::from(variant.shader_dir());
        for mapping in EDIT_SHADER_MAPPINGS {
            let src_path = variant_shader_dir.join(mapping.src);
            let dst_path =
                shader_dst_dir.join(format!("{}_{}.spv", mapping.dst, variant.output_suffix()));
            compile_with_slangc(
                &compiler_path,
                &[&variant_shader_dir, shader_src_dir],
                &src_path,
                &dst_path,
                stage(mapping),
            )?;
        }
    }

    let individual_shader_dir = PathBuf::from(SvoShaderVariant::Individual.shader_dir());
    for mapping in INDIVIDUAL_SPLIT_EDIT_SHADER_MAPPINGS {
        let src_path = individual_shader_dir.join(mapping.src);
        let dst_path = shader_dst_dir.join(mapping.dst);
        compile_with_slangc(
            &compiler_path,
            &[
                &individual_shader_dir,
                &individual_shader_dir.join("edit"),
                &individual_shader_dir.join("edit/allocation"),
                shader_src_dir,
            ],
            &src_path,
            &dst_path,
            stage(mapping),
        )?;
    }

    Ok(())
}

fn detect_current_svo_variant(path: &Path) -> Result<SvoShaderVariant, String> {
    let config = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;

    if config.contains("super::sv64_individual::Variant")
        || config.contains("super::sv64_individual::Sv64IndividualVariant")
    {
        Ok(SvoShaderVariant::Sv64Individual)
    } else if config.contains("super::sv64_grouped::Variant")
        || config.contains("super::sv64_grouped::Sv64GroupedVariant")
    {
        Ok(SvoShaderVariant::Sv64Grouped)
    } else if config.contains("super::individual::Variant")
        || config.contains("super::individual::SvoIndividualVariant")
    {
        Ok(SvoShaderVariant::Individual)
    } else if config.contains("super::grouped::Variant")
        || config.contains("super::grouped::SvoGroupedVariant")
    {
        Ok(SvoShaderVariant::Grouped)
    } else {
        Err(format!(
            "Could not detect CurrentSvo variant from {}",
            path.display()
        ))
    }
}

fn stage(mapping: &ShaderMapping) -> SlangStage {
    match mapping.compiler {
        ShaderCompiler::Slangc(stage) => stage,
    }
}

fn suffixed_output_name(base: &str, variant: SvoShaderVariant) -> String {
    let stem = Path::new(base)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("shader output file must have a valid stem");
    let extension = Path::new(base)
        .extension()
        .and_then(|extension| extension.to_str())
        .expect("shader output file must have a valid extension");
    format!("{stem}_{}.{}", variant.output_suffix(), extension)
}

fn compile_with_slangc(
    slangc: &Path,
    include_dirs: &[&Path],
    src: &Path,
    dst: &Path,
    stage: SlangStage,
) -> Result<(), String> {
    let mut command = Command::new(slangc);
    command
        .arg(src)
        .arg("-target")
        .arg("spirv")
        .arg("-profile")
        .arg("glsl_450")
        .arg("-entry")
        .arg("main")
        .arg("-stage")
        .arg(stage.as_str());

    for include_dir in include_dirs {
        command.arg("-I").arg(include_dir);
    }

    let output = command
        .arg("-O2")
        .arg("-o")
        .arg(dst)
        .output()
        .map_err(|error| format!("Failed to spawn slangc for {}: {error}", src.display()))?;

    if !output.status.success() {
        return Err(format!(
            "slangc failed for {} with status {}",
            src.display(),
            output.status
        ));
    }

    Ok(())
}

fn which_compiler(name: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    let exe_name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };

    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(&exe_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}
