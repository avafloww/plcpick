#pragma once
#include <cstdint>

// SHA-256 implementation for CUDA
// Reference: FIPS 180-4

namespace sha256 {

__constant__ static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

struct State {
    uint32_t h[8];
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__device__ void init(State &state) {
    state.h[0] = 0x6a09e667;
    state.h[1] = 0xbb67ae85;
    state.h[2] = 0x3c6ef372;
    state.h[3] = 0xa54ff53a;
    state.h[4] = 0x510e527f;
    state.h[5] = 0x9b05688c;
    state.h[6] = 0x1f83d9ab;
    state.h[7] = 0x5be0cd19;
}

// Process a single 64-byte block
__device__ void process_block(State &state, const uint8_t *block) {
    uint32_t w[64];

    // Load and convert from big-endian
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }

    // Message schedule
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = state.h[0], b = state.h[1], c = state.h[2], d = state.h[3];
    uint32_t e = state.h[4], f = state.h[5], g = state.h[6], h = state.h[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state.h[0] += a;
    state.h[1] += b;
    state.h[2] += c;
    state.h[3] += d;
    state.h[4] += e;
    state.h[5] += f;
    state.h[6] += g;
    state.h[7] += h;
}

// Hash an arbitrary-length message, output 32 bytes
__device__ void hash(const uint8_t *data, uint32_t len, uint8_t *out) {
    State state;
    init(state);

    // Process complete blocks
    uint32_t i = 0;
    while (i + 64 <= len) {
        process_block(state, data + i);
        i += 64;
    }

    // Padding: remaining bytes + 0x80 + zeros + 8-byte length
    uint8_t pad[128]; // at most 2 blocks needed
    uint32_t rem = len - i;
    for (uint32_t j = 0; j < rem; j++) {
        pad[j] = data[i + j];
    }
    pad[rem] = 0x80;

    uint32_t pad_len;
    if (rem < 56) {
        pad_len = 64;
    } else {
        pad_len = 128;
    }

    for (uint32_t j = rem + 1; j < pad_len - 8; j++) {
        pad[j] = 0;
    }

    // Append bit length (big-endian, 64-bit)
    uint64_t bit_len = (uint64_t)len * 8;
    pad[pad_len - 8] = (uint8_t)(bit_len >> 56);
    pad[pad_len - 7] = (uint8_t)(bit_len >> 48);
    pad[pad_len - 6] = (uint8_t)(bit_len >> 40);
    pad[pad_len - 5] = (uint8_t)(bit_len >> 32);
    pad[pad_len - 4] = (uint8_t)(bit_len >> 24);
    pad[pad_len - 3] = (uint8_t)(bit_len >> 16);
    pad[pad_len - 2] = (uint8_t)(bit_len >> 8);
    pad[pad_len - 1] = (uint8_t)(bit_len);

    // Process padded blocks
    for (uint32_t j = 0; j < pad_len; j += 64) {
        process_block(state, pad + j);
    }

    // Write output (big-endian)
    for (int j = 0; j < 8; j++) {
        out[j * 4]     = (uint8_t)(state.h[j] >> 24);
        out[j * 4 + 1] = (uint8_t)(state.h[j] >> 16);
        out[j * 4 + 2] = (uint8_t)(state.h[j] >> 8);
        out[j * 4 + 3] = (uint8_t)(state.h[j]);
    }
}

} // namespace sha256
