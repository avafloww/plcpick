// Glob pattern matching for wgpu/WGSL
// Same algorithm as the Rust, CUDA, and Vulkan implementations.

// Match pattern against text. Pattern supports '*' wildcard.
// Pattern is read from a storage buffer; text is a function-local array.
// Max pattern/text length: 24.
fn pattern_glob_match(
    pattern: ptr<storage, array<u32>, read>,
    pat_len: u32,
    text: ptr<function, array<u32, 24>>,
    text_len: u32,
) -> bool {
    var pi = 0u;
    var ti = 0u;
    var star = 0xFFFFFFFFu;
    var star_t = 0u;

    while (ti < text_len) {
        if (pi < pat_len && (*pattern)[pi] == (*text)[ti]) {
            pi++;
            ti++;
        } else if (pi < pat_len && (*pattern)[pi] == 0x2Au) { // '*'
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

    while (pi < pat_len && (*pattern)[pi] == 0x2Au) {
        pi++;
    }

    return pi == pat_len;
}
