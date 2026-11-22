#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "hestia.h"

static void contains(const char *text, const char *part) {
    if (text == NULL || strstr(text, part) == NULL) {
        fprintf(stderr, "missing %s in %s\n", part, text == NULL ? "<empty>" : text);
        assert(0);
    }
}

int main(int argc, char **argv) {
    assert(argc == 2 || argc == 3);
    char *error = NULL;
    char config[4096] = "{}";
    if (argc == 3) {
        assert(strpbrk(argv[2], "\"\\") == NULL);
        int size = snprintf(config, sizeof(config), "{\"embed_model_path\":\"%s\"}", argv[2]);
        assert(size > 0 && (size_t)size < sizeof(config));
    }
    hestia_runtime runtime = hestia_open(config, argv[1], NULL, 0, &error);
    if (runtime == 0) fprintf(stderr, "open failed: %s\n", error);
    assert(runtime != 0 && error == NULL);

    char *result = hestia_command(runtime, "test", "Schalte die Stehlampe im Wohnzimmer ein.", 0, &error);
    assert(error == NULL);
    contains(result, "Stehlampe ist an.");
    gopherllm_free_string(result);

    result = hestia_command(runtime, "test", "Alle Lampen im Wohnzimmer an.", 0, &error);
    assert(error == NULL);
    contains(result, "\"Status\":\"clarifying\"");
    contains(result, "\"ID\":\"turn_");
    const char *start = strstr(result, "\"ID\":\"") + 6;
    const char *end = strchr(start, '"');
    char turn[64] = {0};
    assert(end != NULL && (size_t)(end - start) < sizeof(turn));
    memcpy(turn, start, (size_t)(end - start));
    gopherllm_free_string(result);

    result = hestia_confirm(runtime, turn, NULL, 0, &error);
    assert(error == NULL);
    contains(result, "2 Geräte erfolgreich geschaltet.");
    gopherllm_free_string(result);

    if (argc == 3) {
        const char *text = "Was sagen meine Unterlagen zur Wartung?";
        result = hestia_add_document(runtime, "Wartung", text, 0, &error);
        assert(result != NULL && error == NULL);
        gopherllm_free_string(result);
        result = hestia_command(runtime, "test", text, 0, &error);
        assert(error == NULL);
        contains(result, "\"Result\":\"Was sagen meine Unterlagen zur Wartung?\"");
        gopherllm_free_string(result);
    }
    hestia_close(runtime);
    puts(argc == 3 ? "Hestia C bridge: device control and document lookup passed"
                  : "Hestia C bridge: command, clarification and confirmation passed");
}
