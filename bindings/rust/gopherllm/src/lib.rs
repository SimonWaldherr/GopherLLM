//! Safe Rust bindings for [GopherLLM](https://github.com/SimonWaldherr/GopherLLM),
//! a pure-Go GGUF inference engine, over its C ABI (`bindings/c/shim` in the
//! GopherLLM repo). Mirrors the same small surface the Swift/Obj-C binding
//! (`mobile.Engine`) uses: load a GGUF, generate or stream a completion,
//! read basic model info.
//!
//! ```no_run
//! use gopherllm::{Engine, LoadOptions, GenerationOptions};
//!
//! let engine = Engine::new();
//! engine.load("model.gguf", &LoadOptions::default())?;
//! let text = engine.generate("Hello!", &GenerationOptions { max_tokens: Some(64), ..Default::default() })?;
//! println!("{text}");
//! # Ok::<(), gopherllm::Error>(())
//! ```
mod ffi;

use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_void};
use std::ptr;

/// An error returned by the engine or the underlying Go runtime. The message
/// is whatever GopherLLM itself reported; it is not further categorized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn json_err<E: fmt::Display>(e: E) -> Error {
    Error(e.to_string())
}

/// Options for [`Engine::load`]. Field-for-field the same shape
/// `mobile.Engine.Load`'s `optionsJSON` takes; every field is optional and
/// falls back to GopherLLM's own default when omitted.
#[derive(Debug, Default, Clone, Serialize)]
pub struct LoadOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prepare_quantized: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_of_core: Option<bool>,
    /// One of "none", "core", "all".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefault: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metal: Option<bool>,
}

/// Options for [`Engine::generate`]/[`Engine::generate_stream`]. Field-for-field
/// the same shape `mobile.Engine.Generate`'s `optionsJSON` takes.
#[derive(Debug, Default, Clone, Serialize)]
pub struct GenerationOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop: Vec<String>,
}

/// Stable, inexpensive model metadata; see [`Engine::info`].
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub file_size_bytes: i64,
    pub architecture: String,
    pub context_length: i32,
    pub vocab_size: i32,
    pub mapped: bool,
    pub metal_available: bool,
}

/// The result of a completed [`Engine::generate_stream`] call.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerateResult {
    pub text: String,
    pub finish_reason: String,
    pub generated_tokens: i32,
}

/// One loaded (or loadable) model. Mirrors `mobile.Engine`: independent
/// instances are unrelated, and calls that touch model memory are serialized
/// on the Go side, so it is safe (if pointless) to call an `Engine` from
/// multiple threads concurrently -- they simply queue.
pub struct Engine {
    handle: ffi::EngineHandle,
}

// The handle is just an opaque integer key into a Go-side table guarded by
// its own mutex (mobile.Engine); nothing here is thread-affine.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Creates a new, unloaded engine.
    pub fn new() -> Self {
        Engine {
            handle: unsafe { ffi::gopherllm_engine_new() },
        }
    }

    /// Loads a GGUF model file. Fails if a model is already loaded (call
    /// [`Engine::unload`] first) or if the file cannot be opened.
    pub fn load(&self, path: &str, options: &LoadOptions) -> Result<()> {
        let c_path = CString::new(path).map_err(json_err)?;
        let c_json = CString::new(serde_json::to_string(options).map_err(json_err)?).map_err(json_err)?;
        let err = unsafe { ffi::gopherllm_load(self.handle, c_path.as_ptr(), c_json.as_ptr()) };
        take_error(err)
    }

    /// Releases the current model, if any. The engine itself remains usable
    /// for a subsequent [`Engine::load`].
    pub fn unload(&self) -> Result<()> {
        take_error(unsafe { ffi::gopherllm_unload(self.handle) })
    }

    pub fn is_loaded(&self) -> bool {
        unsafe { ffi::gopherllm_is_loaded(self.handle) != 0 }
    }

    /// The loaded model's display name, or `""` if nothing is loaded.
    pub fn model_name(&self) -> String {
        take_string(unsafe { ffi::gopherllm_model_name(self.handle) })
    }

    /// Stable, inexpensive model metadata.
    pub fn info(&self) -> Result<ModelInfo> {
        let json = take_string(unsafe { ffi::gopherllm_info_json(self.handle) });
        serde_json::from_str(&json).map_err(json_err)
    }

    /// Interrupts an in-flight [`Engine::generate`]/[`Engine::generate_stream`]
    /// call, if any. Safe to call at any time.
    pub fn cancel(&self) {
        unsafe { ffi::gopherllm_cancel(self.handle) }
    }

    /// Runs a non-streaming chat completion and returns the generated text.
    pub fn generate(&self, prompt: &str, options: &GenerationOptions) -> Result<String> {
        let c_prompt = CString::new(prompt).map_err(json_err)?;
        let c_json = CString::new(serde_json::to_string(options).map_err(json_err)?).map_err(json_err)?;
        let mut error_out: *mut c_char = ptr::null_mut();
        let text_ptr = unsafe { ffi::gopherllm_generate(self.handle, c_prompt.as_ptr(), c_json.as_ptr(), &mut error_out) };
        if !error_out.is_null() {
            return Err(Error(take_string(error_out)));
        }
        Ok(take_string(text_ptr))
    }

    /// Runs a streaming chat completion, calling `on_delta` once per text
    /// increment as it is generated, then returns the final result. Blocks
    /// until generation completes, fails, or is [`Engine::cancel`]ed.
    pub fn generate_stream<F: FnMut(&str)>(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        on_delta: F,
    ) -> Result<GenerateResult> {
        let c_prompt = CString::new(prompt).map_err(json_err)?;
        let c_json = CString::new(serde_json::to_string(options).map_err(json_err)?).map_err(json_err)?;
        let mut on_delta = on_delta;
        let mut state = StreamState {
            on_delta: &mut on_delta,
            complete: None,
            error: None,
        };
        let userdata = &mut state as *mut StreamState as *mut c_void;
        let call_err = unsafe {
            ffi::gopherllm_generate_stream(
                self.handle,
                c_prompt.as_ptr(),
                c_json.as_ptr(),
                trampoline_delta,
                trampoline_complete,
                trampoline_error,
                userdata,
            )
        };
        if !call_err.is_null() {
            return Err(Error(take_string(call_err)));
        }
        if let Some(message) = state.error {
            return Err(Error(message));
        }
        state.complete.ok_or_else(|| Error("stream ended without a result".to_string()))
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ffi::gopherllm_engine_free(self.handle) }
    }
}

/// Carries the caller's closure and the streamed outcome across the C
/// callback boundary. Lives entirely on the stack of the `generate_stream`
/// call that owns it: gopherllm_generate_stream is synchronous (it does not
/// return until every callback has already fired), so there is no lifetime
/// or ownership hazard in borrowing it by raw pointer.
struct StreamState<'a> {
    on_delta: &'a mut dyn FnMut(&str),
    complete: Option<GenerateResult>,
    error: Option<String>,
}

extern "C" fn trampoline_delta(text: *const c_char, userdata: *mut c_void) {
    let state = unsafe { &mut *(userdata as *mut StreamState) };
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    (state.on_delta)(&text);
}

extern "C" fn trampoline_complete(result_json: *const c_char, userdata: *mut c_void) {
    let state = unsafe { &mut *(userdata as *mut StreamState) };
    let json = unsafe { CStr::from_ptr(result_json) }.to_string_lossy();
    state.complete = serde_json::from_str(&json).ok();
}

extern "C" fn trampoline_error(message: *const c_char, userdata: *mut c_void) {
    let state = unsafe { &mut *(userdata as *mut StreamState) };
    state.error = Some(unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned());
}

/// Consumes a `char*` GopherLLM returned (per bindings/c/shim's convention,
/// every returned string must be freed exactly once via
/// gopherllm_free_string), returning "" for NULL.
fn take_string(ptr: *mut c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let s = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
    unsafe { ffi::gopherllm_free_string(ptr) };
    s
}

fn take_error(ptr: *mut c_char) -> Result<()> {
    if ptr.is_null() {
        Ok(())
    } else {
        Err(Error(take_string(ptr)))
    }
}
