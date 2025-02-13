use std::{env, path::PathBuf, process::Command};

use bindgen::CargoCallbacks;
use regex::Regex;

fn main() {
    println!("cargo:rerun-if-changed=cuda");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cuda_src = "src/cuda/kernels/grid.cu";
    let ptx_out = out_dir.join("grid.ptx");

    let nvcc_status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_out)
        .arg(cuda_src)
        .status()
        .unwrap();

    assert!(
        nvcc_status.success(),
        "Failed to compile CUDA source to PTX."
    );

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/cuda/includes/wrapper.h")
        .blocklist_type(".*_t")
        .blocklist_type("__u_.*")
        .blocklist_item("__glibc_c99_flexarr_available")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(CargoCallbacks))
        // we use "no_copy" and "no_debug" here because we don't know if we can safely generate them for our structs in C code (they may contain raw pointers)
        .no_copy("*")
        .no_debug("*")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // we need to make modifications to the generated code
    let generated_bindings = bindings.to_string();

    // Regex to find raw pointers to float and replace them with CudaSlice<f32>
    // You can copy this regex to add/modify other types of pointers, for example "*mut i32"
    let pointer_regex = Regex::new(r"\*mut f32").unwrap();
    let modified_bindings = pointer_regex.replace_all(&generated_bindings, "CudaSlice<f32>");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(out_path.join("bindings.rs"), modified_bindings.as_bytes())
        .expect("Failed to write bindings");
}
