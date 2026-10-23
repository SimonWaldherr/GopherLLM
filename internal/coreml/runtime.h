#ifndef GLLM_COREML_RUNTIME_H
#define GLLM_COREML_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>

// Match model.go: at most 16 MiB of reusable I/O across three int32 inputs
// and a float32 output, without any additional model-weight allocation.
#define GLLM_COREML_MAX_BATCH_SIZE 8
#define GLLM_COREML_MAX_BUFFER_ELEMENTS (1 << 20)

typedef struct {
	int batch_size;
	int sequence_length;
	int dimension;
	bool neural_engine_available;
	int neural_engine_cores;
} GLLMCoreMLInfo;

typedef struct {
	int64_t operations;
	int64_t cpu_operations;
	int64_t gpu_operations;
	int64_t neural_engine_operations;
	int64_t unassigned_operations;
} GLLMCoreMLPlan;

bool gllm_coreml_supported(void);
void *gllm_coreml_open(const char *path, int compute_units, const char *ids,
	const char *mask, const char *types, const char *output,
	GLLMCoreMLInfo *info, char **error);
bool gllm_coreml_predict(void *ptr, const int32_t *ids, const int32_t *mask,
	const int32_t *types, float *output, char **error);
void gllm_coreml_release(void *ptr);

bool gllm_coreml_inspect_supported(void);
bool gllm_coreml_inspect(const char *path, int compute_units, GLLMCoreMLPlan *plan, char **error);

#endif
