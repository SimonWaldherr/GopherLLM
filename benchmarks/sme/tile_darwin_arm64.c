//go:build gopherllm_sme_experiment && darwin && arm64 && cgo && !ios

#include "tile.h"
#include <arm_sme.h>

// No global -march flag: optional instructions are confined to these functions.
// Locally-streaming keeps an ordinary C ABI. Private ZA state lets Clang emit
// the AAPCS64 lazy-save protocol instead of handwritten SMSTART/SMSTOP assembly.
__arm_locally_streaming __attribute__((target("sme")))
size_t gopherllm_sme_lanes(void) {
    return svcntsw();
}

__arm_new("za") __arm_locally_streaming __attribute__((target("sme")))
int gopherllm_sme_tile(const int8_t *a, const int8_t *b, int32_t *out,
                      size_t rows, size_t cols, size_t groups, size_t lanes) {
    // VL is thread state. Recheck within the same streaming region as the
    // computation before accessing packed buffers or storing any output.
    if (svcntsw() != lanes) {
        return -1;
    }
    const svbool_t pa = svwhilelt_b8((uint64_t)0, (uint64_t)(rows * 4));
    const svbool_t pb = svwhilelt_b8((uint64_t)0, (uint64_t)(cols * 4));
    const svbool_t pc = svwhilelt_b32((uint64_t)0, (uint64_t)cols);
    svzero_za();
    for (size_t k = 0; k < groups; ++k) {
        const svint8_t av = svld1_s8(pa, a + k * lanes * 4);
        const svint8_t bv = svld1_s8(pb, b + k * lanes * 4);
        svmopa_za32_s8_m(0, pa, pb, av, bv);
    }
    for (size_t row = 0; row < rows; ++row) {
        svst1_hor_za32(0, (uint32_t)row, pc, out + row * cols);
    }
    return 0;
}
