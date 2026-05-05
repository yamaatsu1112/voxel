use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let shader_dst_dir = out_dir.join("shaders");
    println!("cargo:rustc-env=UI_SHADER_DIR={}", shader_dst_dir.display());

    if std::env::var("CI").is_ok() {
        return;
    }

    fs::create_dir_all(&shader_dst_dir)
        .unwrap_or_else(|error| panic!("Failed to create {}: {error}", shader_dst_dir.display()));
    let shader_src_dir = PathBuf::from("src/shaders");
    println!("cargo:rerun-if-changed={}", shader_src_dir.display());

    compile_shader(
        &shader_src_dir,
        &shader_dst_dir,
        "ui-vert.slang",
        "ui-vert.spv",
        "vertex",
    );
    compile_shader(
        &shader_src_dir,
        &shader_dst_dir,
        "ui-frag.slang",
        "ui-frag.spv",
        "fragment",
    );
}

fn compile_shader(shader_src_dir: &Path, shader_dst_dir: &Path, src: &str, dst: &str, stage: &str) {
    let slangc = which_compiler("slangc").expect("slangc not found in PATH");
    let src_path = shader_src_dir.join(src);
    let dst_path = shader_dst_dir.join(dst);
    let output = Command::new(slangc)
        .arg(&src_path)
        .arg("-target")
        .arg("spirv")
        .arg("-profile")
        .arg("glsl_450")
        .arg("-entry")
        .arg("main")
        .arg("-stage")
        .arg(stage)
        .arg("-I")
        .arg(shader_src_dir)
        .arg("-O2")
        .arg("-o")
        .arg(&dst_path)
        .output()
        .expect("failed to spawn slangc");

    if !output.status.success() {
        panic!("slangc failed for {}", src_path.display());
    }
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
