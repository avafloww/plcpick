// Glob pattern matching for Vulkan/GLSL
// Same algorithm as the Rust and CUDA implementations.
//
// This file is #included by .comp files. Do NOT put #version or #extension here.
// Requires: GL_EXT_shader_explicit_arithmetic_types_int8

// Match pattern against text. Pattern supports '*' wildcard.
// Both pattern and text are uint8_t arrays with explicit lengths.
// Max pattern length: 24, text length: always 24.
bool pattern_glob_match(in uint8_t pattern[24], uint pat_len, in uint8_t text[24], uint text_len) {
    uint pi = 0u;
    uint ti = 0u;
    uint star = 0xFFFFFFFFu;
    uint star_t = 0u;

    while (ti < text_len) {
        if (pi < pat_len && pattern[pi] == text[ti]) {
            pi++;
            ti++;
        } else if (pi < pat_len && uint(pattern[pi]) == 0x2Au) { // '*'
            star = pi;
            star_t = ti;
            pi++;
        } else if (star != 0xFFFFFFFFu) {
            pi = star + 1u;
            star_t++;
            ti = star_t;
        } else {
            return false;
        }
    }

    while (pi < pat_len && uint(pattern[pi]) == 0x2Au) {
        pi++;
    }

    return pi == pat_len;
}
