#ifndef GLLM_COREML_FFN_RUNTIME_H
#define GLLM_COREML_FFN_RUNTIME_H

#include <stdbool.h>

// Match ffn.go. Reusable float32 input/output storage is at most 8 MiB.
#define GLLM_COREML_FFN_MAX_TOKENS 128
#define GLLM_COREML_FFN_MAX_DIMENSION 16384
#define GLLM_COREML_FFN_MAX_ELEMENTS (1 << 20)

typedef struct {
	int tokens;
	int dimension;
	// Caller frees this UTF-8 source binding after a successful open.
	char *manifest;
} GLLMCoreMLFFNInfo;

void *gllm_coreml_ffn_open(const char *path, int compute_units, GLLMCoreMLFFNInfo *info, char **error);
bool gllm_coreml_ffn_predict(void *ptr, const float *input, float *output, char **error);
void gllm_coreml_ffn_release(void *ptr);

#endif
