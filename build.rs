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

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}
