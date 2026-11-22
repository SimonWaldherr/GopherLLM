/*
 * gopherllm.h - the C ABI of GopherLLM, a pure-Go GGUF inference engine.
 *
 * Functions forward to the Go mobile, speech and Hub APIs. The Swift, Rust
 * and Python bindings call this ABI directly.
 *
 * Conventions
 *   - Strings are NUL-terminated UTF-8. Options, messages and results are
 *     JSON; an empty string or NULL options argument means "defaults".
 *   - Every char * this API returns is owned by the caller and must be
 *     released exactly once with gopherllm_free_string.
 *   - Functions that can fail either return an error string (NULL means
 *     success) or take `char **error_out`: on failure they return NULL / -1
 *     and, when error_out is non-NULL, store an error string there (caller
 *     frees); on success they store NULL there.
 *   - All functions may be called from any thread. Calls that run the model
 *     block until done; gopherllm_cancel interrupts them from another thread.
 */
#ifndef GOPHERLLM_H
#define GOPHERLLM_H

#include <stdint.h>

#if defined(__clang__)
#define GOPHERLLM_NULLABLE _Nullable
#define GOPHERLLM_NONNULL _Nonnull
#else
#define GOPHERLLM_NULLABLE
#define GOPHERLLM_NONNULL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* An opaque engine handle; 0 is never a valid engine. */
typedef uintptr_t gopherllm_engine;
typedef uintptr_t gopherllm_speech;
typedef uintptr_t gopherllm_operation;

/* Streaming callbacks, invoked on the calling thread before
 * gopherllm_*_stream returns: on_delta once per new piece of text, in order,
 * then exactly one of on_complete (with the result JSON gopherllm_chat
 * returns) or on_error. The strings are only valid during the call.
 * userdata is passed through unmodified. A callback may call gopherllm_cancel
 * but no other function on the same engine. */
typedef void (*gopherllm_delta_cb)(const char *GOPHERLLM_NONNULL delta, void *GOPHERLLM_NULLABLE userdata);
typedef void (*gopherllm_complete_cb)(const char *GOPHERLLM_NONNULL result_json, void *GOPHERLLM_NULLABLE userdata);
typedef void (*gopherllm_error_cb)(const char *GOPHERLLM_NONNULL message, void *GOPHERLLM_NULLABLE userdata);

/* Progress is per file. Strings are borrowed for the callback only. A callback
 * may run on a download worker, including concurrently for different files.
 * Cached files may complete without a progress event. */
typedef void (*gopherllm_hub_progress_cb)(const char *GOPHERLLM_NONNULL file, int64_t completed,
                                           int64_t total, int done, const char *GOPHERLLM_NULLABLE failure,
                                           void *GOPHERLLM_NULLABLE userdata);

/* The cgo build of the library declares the functions itself, with Go's
 * types, and only needs the typedefs above. */
#ifndef GOPHERLLM_CGO_EXPORTS

/* ---- Library ---------------------------------------------------------- */

/* The library version, e.g. "0.3.0-go". */
char *GOPHERLLM_NONNULL gopherllm_version(void);

/* {"version","goos","goarch","cpus","metal_available","metal_status"}. */
char *GOPHERLLM_NONNULL gopherllm_runtime_info_json(void);

/* Reads only the GGUF header at path (fast for any file size) and returns
 * {"name","architecture","supported","parameters","tensor_bytes",
 *  "bits_per_weight","quantization","context_length","layers","vocab_size",
 *  "chat_template","kv_cache_bytes_per_token"}. */
char *GOPHERLLM_NULLABLE gopherllm_inspect_model(const char *GOPHERLLM_NONNULL path, char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);

/* Releases a string returned by this API. NULL is ignored. */
void gopherllm_free_string(char *GOPHERLLM_NULLABLE s);

/* ---- Engine lifecycle ------------------------------------------------- */

/* Creates an engine with no model loaded. Free it with gopherllm_engine_free. */
gopherllm_engine gopherllm_engine_new(void);

/* Cancels running work, releases the model and invalidates the handle.
 * Safe on 0 or an already-freed handle. */
void gopherllm_engine_free(gopherllm_engine engine);

/* Loads a GGUF. options_json: {"threads":int (0 = automatic),
 * "metal":bool, "prefault":"none"|"core"|"all", "prepare_quantized":bool,
 * "out_of_core":bool}. Returns NULL on success or an error string. */
char *GOPHERLLM_NULLABLE gopherllm_load(gopherllm_engine engine, const char *GOPHERLLM_NONNULL path, const char *GOPHERLLM_NULLABLE options_json);

/* Cancels running work and releases the model; the engine stays usable.
 * Returns NULL on success or an error string. */
char *GOPHERLLM_NULLABLE gopherllm_unload(gopherllm_engine engine);

/* 1 if a model is loaded, else 0 (also for an invalid handle). */
int gopherllm_is_loaded(gopherllm_engine engine);

/* The loaded model's name, or "" when none is loaded. */
char *GOPHERLLM_NONNULL gopherllm_model_name(gopherllm_engine engine);

/* gopherllm_inspect_model's fields for the loaded model plus
 * "file_size_bytes","mapped","out_of_core","load_time_ms",
 * "metal_available"; "{}" when none is loaded. */
char *GOPHERLLM_NONNULL gopherllm_info_json(gopherllm_engine engine);

/* Number of tokens text encodes to (including BOS), or -1 on error. */
int gopherllm_count_tokens(gopherllm_engine engine, const char *GOPHERLLM_NONNULL text, char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);

/* Stops the engine's running generation, if any. Safe at any time. */
void gopherllm_cancel(gopherllm_engine engine);

/* ---- Generation ------------------------------------------------------- */
/*
 * options_json: {"max_tokens":int, "temperature":float, "top_p":float,
 * "top_k":int, "min_p":float, "repeat_penalty":float, "seed":uint64,
 * "system_prompt":string, "stop":[string],
 * "context_window_mode":"full"|"recent"|"autoCompress", "json_object":bool}.
 *
 * messages_json: [{"role":"system"|"user"|"assistant","content":string}, ...];
 * a leading system message replaces system_prompt.
 *
 * Result JSON: {"text","finish_reason","generated_tokens","reasoning_text",
 * "prompt_tokens","ttft_ms","prefill_ms","decode_ms","total_ms",
 * "tokens_per_second","context_window"}.
 *
 * Requests on one engine run one at a time, in arrival order.
 */

/* Answers a single user prompt; returns the generated text. */
char *GOPHERLLM_NULLABLE gopherllm_generate(gopherllm_engine engine, const char *GOPHERLLM_NONNULL prompt, const char *GOPHERLLM_NULLABLE options_json, char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);

/* Continues a conversation; returns the result JSON. */
char *GOPHERLLM_NULLABLE gopherllm_chat(gopherllm_engine engine, const char *GOPHERLLM_NONNULL messages_json, const char *GOPHERLLM_NULLABLE options_json, char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);

/* Streaming variants. The return value only reports a call that could not
 * start (a NULL callback or an invalid handle); every other outcome arrives
 * through on_complete or on_error before the function returns NULL. */
char *GOPHERLLM_NULLABLE gopherllm_generate_stream(gopherllm_engine engine, const char *GOPHERLLM_NONNULL prompt, const char *GOPHERLLM_NULLABLE options_json,
                                                   gopherllm_delta_cb GOPHERLLM_NULLABLE on_delta, gopherllm_complete_cb GOPHERLLM_NULLABLE on_complete,
                                                   gopherllm_error_cb GOPHERLLM_NULLABLE on_error, void *GOPHERLLM_NULLABLE userdata);

char *GOPHERLLM_NULLABLE gopherllm_chat_stream(gopherllm_engine engine, const char *GOPHERLLM_NONNULL messages_json, const char *GOPHERLLM_NULLABLE options_json,
                                               gopherllm_delta_cb GOPHERLLM_NULLABLE on_delta, gopherllm_complete_cb GOPHERLLM_NULLABLE on_complete,
                                               gopherllm_error_cb GOPHERLLM_NULLABLE on_error, void *GOPHERLLM_NULLABLE userdata);

/* ---- Operation-scoped cancellation ----------------------------------- */

/* One handle per blocking call, or 0 for a non-cancelable call. Cancel
 * may run from any thread and also works before the call starts. Free the
 * handle after the call and its callbacks return. */
gopherllm_operation gopherllm_operation_new(void);
void gopherllm_operation_cancel(gopherllm_operation operation);
void gopherllm_operation_free(gopherllm_operation operation);

/* ---- Speech ---------------------------------------------------------- */

int gopherllm_speech_available(const char *GOPHERLLM_NONNULL backend);

/* Opens a local model once. Options: {"backend":"voxtral"|"phonon",
 * "cpu_only":bool (Voxtral)}. 0 on failure. */
gopherllm_speech gopherllm_speech_open(const char *GOPHERLLM_NONNULL path, const char *GOPHERLLM_NONNULL options_json,
                                     gopherllm_operation operation, char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);
void gopherllm_speech_free(gopherllm_speech speech);

/* Borrows normalized float32 mono PCM at 16 kHz for the call (1 to 480000
 * samples). Returns {"text":string}. With an activation phrase, only the
 * command after the matched phrase is returned and "matched" is included. */
char *GOPHERLLM_NULLABLE gopherllm_speech_transcribe(gopherllm_speech speech, const float *GOPHERLLM_NONNULL samples, uint32_t sample_count,
                                                    const char *GOPHERLLM_NULLABLE activation_phrase, gopherllm_operation operation,
                                                    char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);

/* ---- Hugging Face Hub ------------------------------------------------- */

/* Uses huggingface.co. Options: {"cache_dir":absolute path, "token":string,
 * "offline":bool}. Search also accepts limit, format and pipeline_tag.
 * Exact downloads select permitted model artifacts. With NULL files_json,
 * download resolves a GGUF selector and all shards. Result is a path array. */
char *GOPHERLLM_NULLABLE gopherllm_hub_search(const char *GOPHERLLM_NONNULL query, const char *GOPHERLLM_NONNULL options_json,
                                            gopherllm_operation operation, char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);
char *GOPHERLLM_NULLABLE gopherllm_hub_inspect(const char *GOPHERLLM_NONNULL reference, const char *GOPHERLLM_NONNULL options_json,
                                             gopherllm_operation operation, char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);
char *GOPHERLLM_NULLABLE gopherllm_hub_download(const char *GOPHERLLM_NONNULL reference, const char *GOPHERLLM_NULLABLE files_json,
                                              const char *GOPHERLLM_NONNULL options_json, gopherllm_operation operation,
                                              gopherllm_hub_progress_cb GOPHERLLM_NULLABLE on_progress, void *GOPHERLLM_NULLABLE userdata,
                                              char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);

#endif /* GOPHERLLM_CGO_EXPORTS */

#ifdef __cplusplus
}
#endif

#endif /* GOPHERLLM_H */
