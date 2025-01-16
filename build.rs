use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed={}", "grid.cu");

    let ptx_out = "src/grid.ptx";
    let cuda_src = "src/grid.cu";

    let nvcc_status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_out)
        .arg(&cuda_src)
        .status()
        .unwrap();

    assert!(
        nvcc_status.success(),
        "Failed to compile CUDA source to PTX."
    );
}
