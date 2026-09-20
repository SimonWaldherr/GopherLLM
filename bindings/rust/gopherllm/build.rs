use std::env;
use std::path::{Path, PathBuf};

/// Locates the prebuilt libgopherllm shared library (see
/// ../../../scripts/build-capi.sh) and tells cargo how to link it.
///
/// Search order:
///   1. `GOPHERLLM_LIB_DIR` env var, if set (an explicit override, e.g. for a
///      packaged/installed library outside this checkout).
///   2. `<repo>/build/capi`, the default output of scripts/build-capi.sh, for
///      in-repo development.
///
/// The FFI declarations in src/ffi.rs are hand-written against
/// bindings/c/shim's small, stable surface rather than generated with
/// bindgen, so building this crate needs no libclang.
fn main() {
    let lib_dir = locate_lib_dir();
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=gopherllm");
    // Without this, a binary linked against the dylib only finds it via
    // DYLD_LIBRARY_PATH/LD_LIBRARY_PATH at run time, which `cargo run`/`cargo
    // test` do not set for you. Baking the build-time path into the rpath
    // keeps `cargo test` working the same way `go run`/`go test` do for the
    // Go library itself, with no extra environment setup.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rerun-if-env-changed=GOPHERLLM_LIB_DIR");
}

fn locate_lib_dir() -> PathBuf {
    if let Ok(dir) = env::var("GOPHERLLM_LIB_DIR") {
        return PathBuf::from(dir);
    }
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo");
    let default_dir = Path::new(&manifest_dir).join("../../../build/capi");
    if !default_dir.join(lib_filename()).exists() {
        panic!(
            "libgopherllm not found at {} and GOPHERLLM_LIB_DIR is not set.\n\
             Build it first: ../../../scripts/build-capi.sh (from the gopherllm crate directory),\n\
             or point GOPHERLLM_LIB_DIR at a directory containing {}.",
            default_dir.display(),
            lib_filename(),
        );
    }
    default_dir
}

fn lib_filename() -> &'static str {
    if cfg!(target_os = "macos") {
        "libgopherllm.dylib"
    } else if cfg!(target_os = "windows") {
        "gopherllm.dll"
    } else {
        "libgopherllm.so"
    }
}
