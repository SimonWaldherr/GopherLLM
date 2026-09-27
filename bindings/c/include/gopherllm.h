/*
 * gopherllm.h - the C ABI of GopherLLM, a pure-Go GGUF inference engine.
 *
 * Every function forwards to mobile.Engine (see ../../../mobile); the Swift,
 * Rust and Python bindings are thin layers over exactly these functions.
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

/* Streaming callbacks, invoked on the calling thread before
 * gopherllm_*_stream returns: on_delta once per new piece of text, in order,
 * then exactly one of on_complete (with the result JSON gopherllm_chat
 * returns) or on_error. The strings are only valid during the call.
 * userdata is passed through unmodified. A callback may call gopherllm_cancel
 * but no other function on the same engine. */
typedef void (*gopherllm_delta_cb)(const char *GOPHERLLM_NONNULL delta, void *GOPHERLLM_NULLABLE userdata);
typedef void (*gopherllm_complete_cb)(const char *GOPHERLLM_NONNULL result_json, void *GOPHERLLM_NULLABLE userdata);
typedef void (*gopherllm_error_cb)(const char *GOPHERLLM_NONNULL message, void *GOPHERLLM_NULLABLE userdata);

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

#endif /* GOPHERLLM_CGO_EXPORTS */

#ifdef __cplusplus
}
#endif

#endif /* GOPHERLLM_H */
