// SHA-256 implementation for Vulkan/GLSL
// Reference: FIPS 180-4
//
// This file is #included by .comp files. Do NOT put #version or #extension here.
// Requires: GL_EXT_shader_explicit_arithmetic_types_int8
//           GL_EXT_shader_explicit_arithmetic_types_int64

const uint SHA256_K[64] = uint[64](
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
);

uint sha256_rotr(uint x, int n) {
    return (x >> n) | (x << (32 - n));
}

uint sha256_ch(uint x, uint y, uint z) {
    return (x & y) ^ (~x & z);
}

uint sha256_maj(uint x, uint y, uint z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

uint sha256_sigma0(uint x) {
    return sha256_rotr(x, 2) ^ sha256_rotr(x, 13) ^ sha256_rotr(x, 22);
}

uint sha256_sigma1(uint x) {
    return sha256_rotr(x, 6) ^ sha256_rotr(x, 11) ^ sha256_rotr(x, 25);
}

uint sha256_gamma0(uint x) {
    return sha256_rotr(x, 7) ^ sha256_rotr(x, 18) ^ (x >> 3);
}

uint sha256_gamma1(uint x) {
    return sha256_rotr(x, 17) ^ sha256_rotr(x, 19) ^ (x >> 10);
}

void sha256_init(inout uint h[8]) {
    h[0] = 0x6a09e667u;
    h[1] = 0xbb67ae85u;
    h[2] = 0x3c6ef372u;
    h[3] = 0xa54ff53au;
    h[4] = 0x510e527fu;
    h[5] = 0x9b05688cu;
    h[6] = 0x1f83d9abu;
    h[7] = 0x5be0cd19u;
}

// Process a single 64-byte block from a uint8_t array at the given offset
void sha256_process_block(inout uint h[8], in uint8_t block[128], uint offset) {
    uint w[64];

    // Load and convert from big-endian
    for (int i = 0; i < 16; i++) {
        uint idx = offset + uint(i) * 4u;
        w[i] = (uint(block[idx]) << 24) |
               (uint(block[idx + 1u]) << 16) |
               (uint(block[idx + 2u]) << 8) |
               uint(block[idx + 3u]);
    }

    // Message schedule
    for (int i = 16; i < 64; i++) {
        w[i] = sha256_gamma1(w[i - 2]) + w[i - 7] + sha256_gamma0(w[i - 15]) + w[i - 16];
    }

    uint a = h[0], b = h[1], c = h[2], d = h[3];
    uint e = h[4], f = h[5], g = h[6], hh = h[7];

    for (int i = 0; i < 64; i++) {
        uint t1 = hh + sha256_sigma1(e) + sha256_ch(e, f, g) + SHA256_K[i] + w[i];
        uint t2 = sha256_sigma0(a) + sha256_maj(a, b, c);
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

// Hash an arbitrary-length message (up to ~256 bytes), output 32 bytes
// data_len is the actual length of meaningful data in the data array
// The data array must be large enough: at least data_len bytes.
// We use a fixed 256-byte input array to avoid variable-length issues.
void sha256_hash(in uint8_t data[256], uint len, inout uint8_t hash_out[32]) {
    uint h[8];
    sha256_init(h);

    // Process complete 64-byte blocks
    uint i = 0u;
    uint8_t block[128];
    while (i + 64u <= len) {
        for (uint j = 0u; j < 64u; j++) {
            block[j] = data[i + j];
        }
        sha256_process_block(h, block, 0u);
        i += 64u;
    }

    // Padding
    uint rem = len - i;
    for (uint j = 0u; j < 128u; j++) block[j] = uint8_t(0u);
    for (uint j = 0u; j < rem; j++) block[j] = data[i + j];
    block[rem] = uint8_t(0x80u);

    uint pad_len = (rem < 56u) ? 64u : 128u;

    // Append bit length (big-endian, 64-bit)
    uint64_t bit_len = uint64_t(len) * 8ul;
    block[pad_len - 8u] = uint8_t(uint(bit_len >> 56) & 0xFFu);
    block[pad_len - 7u] = uint8_t(uint(bit_len >> 48) & 0xFFu);
    block[pad_len - 6u] = uint8_t(uint(bit_len >> 40) & 0xFFu);
    block[pad_len - 5u] = uint8_t(uint(bit_len >> 32) & 0xFFu);
    block[pad_len - 4u] = uint8_t(uint((bit_len >> 24)) & 0xFFu);
    block[pad_len - 3u] = uint8_t(uint((bit_len >> 16)) & 0xFFu);
    block[pad_len - 2u] = uint8_t(uint((bit_len >> 8)) & 0xFFu);
    block[pad_len - 1u] = uint8_t(uint(bit_len) & 0xFFu);

    // Process padded blocks
    for (uint j = 0u; j < pad_len; j += 64u) {
        sha256_process_block(h, block, j);
    }

    // Write output (big-endian)
    for (int j = 0; j < 8; j++) {
        hash_out[j * 4]     = uint8_t((h[j] >> 24) & 0xFFu);
        hash_out[j * 4 + 1] = uint8_t((h[j] >> 16) & 0xFFu);
        hash_out[j * 4 + 2] = uint8_t((h[j] >> 8) & 0xFFu);
        hash_out[j * 4 + 3] = uint8_t(h[j] & 0xFFu);
    }
}
