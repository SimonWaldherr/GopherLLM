//! Raw FFI declarations mirroring the C ABI's public header,
//! bindings/c/include/gopherllm.h, which documents each function's contract
//! (ownership, NULL conventions, threading). Hand-written rather than
//! bindgen-generated: the C surface is small and changes rarely, and
//! hand-written bindings need no libclang at build time. This crate declares
//! the subset it uses.

use std::os::raw::{c_char, c_int, c_void};

/// Matches GoUintptr (size_t) in the generated header: an opaque engine
/// handle, never a real pointer a caller should dereference.
pub type EngineHandle = usize;

pub type DeltaCb = extern "C" fn(*const c_char, *mut c_void);
pub type CompleteCb = extern "C" fn(*const c_char, *mut c_void);
pub type ErrorCb = extern "C" fn(*const c_char, *mut c_void);

extern "C" {
    pub fn gopherllm_engine_new() -> EngineHandle;
    pub fn gopherllm_engine_free(handle: EngineHandle);
    pub fn gopherllm_load(handle: EngineHandle, path: *const c_char, options_json: *const c_char) -> *mut c_char;
    pub fn gopherllm_unload(handle: EngineHandle) -> *mut c_char;
    pub fn gopherllm_is_loaded(handle: EngineHandle) -> c_int;
    pub fn gopherllm_model_name(handle: EngineHandle) -> *mut c_char;
    pub fn gopherllm_info_json(handle: EngineHandle) -> *mut c_char;
    pub fn gopherllm_cancel(handle: EngineHandle);
    pub fn gopherllm_generate(
        handle: EngineHandle,
        prompt: *const c_char,
        options_json: *const c_char,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    pub fn gopherllm_generate_stream(
        handle: EngineHandle,
        prompt: *const c_char,
        options_json: *const c_char,
        on_delta: DeltaCb,
        on_complete: CompleteCb,
        on_error: ErrorCb,
        userdata: *mut c_void,
    ) -> *mut c_char;
    pub fn gopherllm_free_string(s: *mut c_char);
}
