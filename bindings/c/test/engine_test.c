/*
 * End-to-end test of the C ABI through the public header, linked against
 * the static library exactly as the Swift XCFramework links it. Run it with
 * scripts/test-capi.sh (or `make capi-test`), which builds the library and
 * writes internal/testmodel's GGUF; argv[1] is that model's path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gopherllm.h"

/* What internal/testmodel generates for "Hallo" with DETERMINISTIC. */
#define EXPECTED_TEXT "$(F=bBz\".C\\[}3WeO`>i[\"cZ"
#define DETERMINISTIC "{\"max_tokens\":24,\"seed\":1}"

static int failures;

#define CHECK(cond, ...)                                              \
	do {                                                              \
		if (!(cond)) {                                                \
			failures++;                                               \
			fprintf(stderr, "%s:%d: check failed: %s: ", __FILE__, __LINE__, #cond); \
			fprintf(stderr, __VA_ARGS__);                             \
			fputc('\n', stderr);                                      \
		}                                                             \
	} while (0)

static int contains(const char *s, const char *needle) { return s && strstr(s, needle) != NULL; }

struct stream_state {
	gopherllm_engine engine;
	char text[4096];
	int deltas, completes, errors, cancel_on_delta;
	char last[512];
};

static void on_delta(const char *delta, void *userdata) {
	struct stream_state *st = userdata;
	st->deltas++;
	strncat(st->text, delta, sizeof st->text - strlen(st->text) - 1);
	if (st->cancel_on_delta)
		gopherllm_cancel(st->engine);
}

static void on_complete(const char *result_json, void *userdata) {
	struct stream_state *st = userdata;
	st->completes++;
	snprintf(st->last, sizeof st->last, "%s", result_json);
}

static void on_error(const char *message, void *userdata) {
	struct stream_state *st = userdata;
	st->errors++;
	snprintf(st->last, sizeof st->last, "%s", message);
}

static void test_library(const char *model) {
	char *version = gopherllm_version();
	CHECK(version && version[0], "empty version");
	char *runtime = gopherllm_runtime_info_json();
	CHECK(contains(runtime, version), "runtime info %s lacks version %s", runtime, version);
	gopherllm_free_string(runtime);
	gopherllm_free_string(version);

	char *err = NULL;
	char *info = gopherllm_inspect_model(model, &err);
	CHECK(err == NULL && contains(info, "\"name\":\"gopherllm-synth-testmodel\"") && contains(info, "\"supported\":true"),
	      "inspect: %s / %s", info, err);
	gopherllm_free_string(info);

	info = gopherllm_inspect_model("/nonexistent/model.gguf", &err);
	CHECK(info == NULL && err != NULL, "inspecting a missing file succeeded");
	gopherllm_free_string(err);
	CHECK(gopherllm_inspect_model("/nonexistent/model.gguf", NULL) == NULL, "NULL error_out");
	gopherllm_free_string(NULL);
}

static void test_engine(const char *model) {
	char *err = gopherllm_load(0, model, NULL);
	CHECK(contains(err, "invalid engine handle"), "load on handle 0: %s", err);
	gopherllm_free_string(err);

	gopherllm_engine engine = gopherllm_engine_new();
	CHECK(engine != 0, "engine_new returned 0");
	CHECK(!gopherllm_is_loaded(engine), "fresh engine is loaded");

	err = gopherllm_load(engine, model, "{\"threads\":-1}");
	CHECK(err != NULL, "invalid load options accepted");
	gopherllm_free_string(err);
	err = gopherllm_load(engine, model, "{\"threads\":1}");
	CHECK(err == NULL, "load: %s", err);
	gopherllm_free_string(err);
	CHECK(gopherllm_is_loaded(engine), "not loaded after load");

	char *name = gopherllm_model_name(engine);
	CHECK(strcmp(name, "gopherllm-synth-testmodel") == 0, "model name %s", name);
	gopherllm_free_string(name);
	char *info = gopherllm_info_json(engine);
	CHECK(contains(info, "\"architecture\":\"llama\"") && contains(info, "\"mapped\""), "info %s", info);
	gopherllm_free_string(info);

	char *error_out = NULL;
	CHECK(gopherllm_count_tokens(engine, "Hallo", &error_out) == 6 && error_out == NULL, "count_tokens");

	char *text = gopherllm_generate(engine, "Hallo", DETERMINISTIC, &error_out);
	CHECK(error_out == NULL && text && strcmp(text, EXPECTED_TEXT) == 0, "generate: %s / %s", text, error_out);
	gopherllm_free_string(text);

	const char *messages = "[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"Hallo\"}]";
	char *result = gopherllm_chat(engine, messages, DETERMINISTIC, &error_out);
	CHECK(error_out == NULL && contains(result, "\"finish_reason\":\"length\"") && contains(result, "\"generated_tokens\":24"),
	      "chat: %s / %s", result, error_out);
	gopherllm_free_string(result);

	result = gopherllm_chat(engine, "[{\"role\":\"robot\",\"content\":\"x\"}]", NULL, &error_out);
	CHECK(result == NULL && contains(error_out, "role"), "chat with a bad role: %s", error_out);
	gopherllm_free_string(error_out);

	struct stream_state st = {.engine = engine};
	char *start = gopherllm_chat_stream(engine, "[{\"role\":\"user\",\"content\":\"Hallo\"}]", DETERMINISTIC, on_delta, on_complete, on_error, &st);
	CHECK(start == NULL && st.completes == 1 && st.errors == 0 && strcmp(st.text, EXPECTED_TEXT) == 0,
	      "chat_stream: start=%s completes=%d errors=%d text=%s", start, st.completes, st.errors, st.text);
	CHECK(contains(st.last, "\"generated_tokens\":24"), "stream result %s", st.last);

	memset(&st, 0, sizeof st);
	st.engine = engine;
	st.cancel_on_delta = 1;
	start = gopherllm_generate_stream(engine, "Hallo", "{\"max_tokens\":200}", on_delta, on_complete, on_error, &st);
	CHECK(start == NULL && st.errors == 1 && st.completes == 0 && st.deltas < 200 && contains(st.last, "canceled"),
	      "canceled stream: errors=%d completes=%d deltas=%d last=%s", st.errors, st.completes, st.deltas, st.last);

	memset(&st, 0, sizeof st);
	start = gopherllm_chat_stream(engine, "not json", NULL, on_delta, on_complete, on_error, &st);
	CHECK(start == NULL && st.errors == 1 && st.completes == 0, "invalid messages reach on_error");
	start = gopherllm_chat_stream(engine, "[]", NULL, NULL, on_complete, on_error, &st);
	CHECK(start != NULL, "NULL callback accepted");
	gopherllm_free_string(start);

	err = gopherllm_unload(engine);
	CHECK(err == NULL && !gopherllm_is_loaded(engine), "unload: %s", err);
	text = gopherllm_generate(engine, "Hallo", NULL, &error_out);
	CHECK(text == NULL && contains(error_out, "not loaded"), "generate without a model: %s", error_out);
	gopherllm_free_string(error_out);
	CHECK(gopherllm_count_tokens(engine, "x", NULL) == -1, "count_tokens without a model");

	gopherllm_engine_free(engine);
	gopherllm_engine_free(engine); /* a stale handle is harmless */
	CHECK(!gopherllm_is_loaded(engine), "freed engine reports loaded");
}

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
		return 2;
	}
	test_library(argv[1]);
	test_engine(argv[1]);
	if (failures) {
		fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	puts("C ABI: all checks passed");
	return 0;
}
