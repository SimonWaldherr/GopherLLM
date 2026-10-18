#ifndef GOPHERLLM_SME_TILE_H
#define GOPHERLLM_SME_TILE_H

#include <stddef.h>
#include <stdint.h>

// These have ordinary, non-streaming C interfaces, including at the cgo ABI.
size_t gopherllm_sme_lanes(void);
int gopherllm_sme_tile(const int8_t *a, const int8_t *b, int32_t *out,
                      size_t rows, size_t cols, size_t groups, size_t lanes);

#endif
