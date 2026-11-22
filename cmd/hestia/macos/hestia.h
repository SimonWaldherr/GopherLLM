#ifndef HESTIA_H
#define HESTIA_H
#include "gopherllm.h"

#ifndef GOPHERLLM_CGO_EXPORTS
typedef uintptr_t hestia_runtime;
/* data_dir is the app-owned directory. The token may be NULL. */
hestia_runtime hestia_open(const char *GOPHERLLM_NONNULL configuration, const char *GOPHERLLM_NONNULL data_dir,
                          const char *GOPHERLLM_NULLABLE token, gopherllm_operation operation,
                          char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);
void hestia_close(hestia_runtime runtime);
char *GOPHERLLM_NULLABLE hestia_command(hestia_runtime runtime, const char *GOPHERLLM_NONNULL conversation,
                                       const char *GOPHERLLM_NONNULL text, gopherllm_operation operation,
                                       char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);
char *GOPHERLLM_NULLABLE hestia_confirm(hestia_runtime runtime, const char *GOPHERLLM_NONNULL turn,
                                       const char *GOPHERLLM_NULLABLE device, gopherllm_operation operation,
                                       char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);
char *GOPHERLLM_NULLABLE hestia_add_document(hestia_runtime runtime, const char *GOPHERLLM_NONNULL title,
                                            const char *GOPHERLLM_NONNULL text, gopherllm_operation operation,
                                            char *GOPHERLLM_NULLABLE *GOPHERLLM_NULLABLE error_out);
#endif
#endif
