#[cfg(feature = "cuda")]
fn build_cuda() {
    use std::process::Command;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/kernel.ptx");

    // Find nvcc: check PATH, then common install locations
    let nvcc = ["nvcc", "/opt/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc"]
        .iter()
        .find(|p| Command::new(p).arg("--version").output().is_ok())
        .expect("failed to find nvcc — is CUDA toolkit installed?");

    let status = Command::new(nvcc)
        .args([
            "--ptx",
            "-o",
            &ptx_path,
            "cuda/kernel.cu",
            "-I",
            "cuda/",
            "--std=c++17",
            "-arch=sm_75", // Turing and above (CUDA 13.1 minimum)
        ])
        .status()
        .expect("failed to run nvcc — is CUDA toolkit installed?");

    if !status.success() {
        panic!("nvcc compilation failed");
    }

    println!("cargo:rerun-if-changed=cuda/");
}

#[cfg(feature = "vulkan")]
fn build_vulkan() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let compiler =
        shaderc::Compiler::new().expect("failed to create shaderc compiler — is cmake installed?");
    let mut options =
        shaderc::CompileOptions::new().expect("failed to create compile options");

    // Set up include resolution from vulkan/ directory
    options.set_include_callback(|name, _type, _source, _depth| {
        let path = format!("vulkan/{name}");
        match std::fs::read_to_string(&path) {
            Ok(content) => Ok(shaderc::ResolvedInclude {
                resolved_name: path,
                content,
            }),
            Err(e) => Err(format!("Failed to include {path}: {e}")),
        }
    });

    // Compile each .comp file
    for entry in std::fs::read_dir("vulkan").expect("vulkan/ directory must exist") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "comp") {
            let name = path.file_name().unwrap().to_str().unwrap();
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
            let spirv = compiler
                .compile_into_spirv(
                    &source,
                    shaderc::ShaderKind::Compute,
                    name,
                    "main",
                    Some(&options),
                )
                .unwrap_or_else(|e| panic!("failed to compile {name}: {e}"));
            let spv_path = format!("{out_dir}/{name}.spv");
            std::fs::write(&spv_path, spirv.as_binary_u8())
                .unwrap_or_else(|e| panic!("failed to write {spv_path}: {e}"));
        }
    }

    println!("cargo:rerun-if-changed=vulkan/");
}

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
    #[cfg(feature = "vulkan")]
    build_vulkan();
}
