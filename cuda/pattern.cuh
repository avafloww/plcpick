#pragma once
#include <cstdint>

// Glob pattern matching for CUDA (same algorithm as Rust side)

namespace pattern {

// Match pattern against text. Pattern supports '*' wildcard.
// Both pattern and text are byte arrays with explicit lengths.
__device__ bool glob_match(const char *pattern, uint32_t pat_len,
                           const char *text, uint32_t text_len) {
    uint32_t pi = 0, ti = 0;
    uint32_t star = 0xFFFFFFFF;
    uint32_t star_t = 0;

    while (ti < text_len) {
        if (pi < pat_len && pattern[pi] == text[ti]) {
            pi++;
            ti++;
        } else if (pi < pat_len && pattern[pi] == '*') {
            star = pi;
            star_t = ti;
            pi++;
        } else if (star != 0xFFFFFFFF) {
            pi = star + 1;
            star_t++;
            ti = star_t;
        } else {
            return false;
        }
    }

    while (pi < pat_len && pattern[pi] == '*') {
        pi++;
    }

    return pi == pat_len;
}

} // namespace pattern
