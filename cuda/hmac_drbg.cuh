#pragma once
#include <cstdint>
#include "sha256.cuh"

// HMAC-SHA256 and RFC 6979 deterministic nonce generation for CUDA

namespace hmac {

// HMAC-SHA256: key is always 32 bytes, message is variable length
__device__ void hmac_sha256(const uint8_t *key32, const uint8_t *msg, uint32_t msg_len, uint8_t *out32) {
    uint8_t ipad[64];
    uint8_t opad[64];

    // Build ipad and opad
    for (int i = 0; i < 32; i++) {
        ipad[i] = key32[i] ^ 0x36;
        opad[i] = key32[i] ^ 0x5c;
    }
    for (int i = 32; i < 64; i++) {
        ipad[i] = 0x36;
        opad[i] = 0x5c;
    }

    // inner = SHA256(ipad || msg)
    // Build buffer: ipad (64) + msg (msg_len)
    // To avoid large stack allocation, process in two steps using SHA256 internals
    sha256::State inner_state;
    sha256::init(inner_state);
    sha256::process_block(inner_state, ipad);

    // Now hash the message with the ipad state
    // We need to continue hashing msg with the state after processing ipad
    // Total length so far: 64 bytes
    uint32_t total_inner = 64 + msg_len;
    uint32_t i = 0;
    while (i + 64 <= msg_len) {
        sha256::process_block(inner_state, msg + i);
        i += 64;
    }

    // Pad the remaining bytes
    uint8_t pad[128];
    uint32_t rem = msg_len - i;
    for (uint32_t j = 0; j < rem; j++) pad[j] = msg[i + j];
    pad[rem] = 0x80;

    uint32_t pad_len;
    if (rem < 56) {
        pad_len = 64;
    } else {
        pad_len = 128;
    }

    for (uint32_t j = rem + 1; j < pad_len - 8; j++) pad[j] = 0;

    uint64_t bit_len = (uint64_t)total_inner * 8;
    pad[pad_len - 8] = (uint8_t)(bit_len >> 56);
    pad[pad_len - 7] = (uint8_t)(bit_len >> 48);
    pad[pad_len - 6] = (uint8_t)(bit_len >> 40);
    pad[pad_len - 5] = (uint8_t)(bit_len >> 32);
    pad[pad_len - 4] = (uint8_t)(bit_len >> 24);
    pad[pad_len - 3] = (uint8_t)(bit_len >> 16);
    pad[pad_len - 2] = (uint8_t)(bit_len >> 8);
    pad[pad_len - 1] = (uint8_t)(bit_len);

    for (uint32_t j = 0; j < pad_len; j += 64) {
        sha256::process_block(inner_state, pad + j);
    }

    uint8_t inner_hash[32];
    for (int j = 0; j < 8; j++) {
        inner_hash[j * 4]     = (uint8_t)(inner_state.h[j] >> 24);
        inner_hash[j * 4 + 1] = (uint8_t)(inner_state.h[j] >> 16);
        inner_hash[j * 4 + 2] = (uint8_t)(inner_state.h[j] >> 8);
        inner_hash[j * 4 + 3] = (uint8_t)(inner_state.h[j]);
    }

    // outer = SHA256(opad || inner_hash)
    sha256::State outer_state;
    sha256::init(outer_state);
    sha256::process_block(outer_state, opad);

    // inner_hash is 32 bytes, needs padding in a 64-byte block
    uint8_t outer_block[64];
    for (int j = 0; j < 32; j++) outer_block[j] = inner_hash[j];
    outer_block[32] = 0x80;
    for (int j = 33; j < 56; j++) outer_block[j] = 0;
    // total length = 64 + 32 = 96 bytes = 768 bits
    uint64_t outer_bits = 768;
    outer_block[56] = (uint8_t)(outer_bits >> 56);
    outer_block[57] = (uint8_t)(outer_bits >> 48);
    outer_block[58] = (uint8_t)(outer_bits >> 40);
    outer_block[59] = (uint8_t)(outer_bits >> 32);
    outer_block[60] = (uint8_t)(outer_bits >> 24);
    outer_block[61] = (uint8_t)(outer_bits >> 16);
    outer_block[62] = (uint8_t)(outer_bits >> 8);
    outer_block[63] = (uint8_t)(outer_bits);

    sha256::process_block(outer_state, outer_block);

    for (int j = 0; j < 8; j++) {
        out32[j * 4]     = (uint8_t)(outer_state.h[j] >> 24);
        out32[j * 4 + 1] = (uint8_t)(outer_state.h[j] >> 16);
        out32[j * 4 + 2] = (uint8_t)(outer_state.h[j] >> 8);
        out32[j * 4 + 3] = (uint8_t)(outer_state.h[j]);
    }
}

// RFC 6979 deterministic nonce generation for secp256k1
// Inputs: private key (32 bytes big-endian), message hash (32 bytes)
// Output: nonce k as U256
__device__ void rfc6979_nonce(secp256k1::U256 &k_out,
                               const uint8_t *privkey32,
                               const uint8_t *hash32) {
    // Step a: h1 = hash32 (already provided)
    // Step b: V = 0x01 01 01 ... 01 (32 bytes)
    uint8_t V[32];
    for (int i = 0; i < 32; i++) V[i] = 0x01;

    // Step c: K = 0x00 00 00 ... 00 (32 bytes)
    uint8_t K[32];
    for (int i = 0; i < 32; i++) K[i] = 0x00;

    // Step d: K = HMAC_K(V || 0x00 || privkey || hash)
    uint8_t hmac_input[97]; // 32 + 1 + 32 + 32
    for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
    hmac_input[32] = 0x00;
    for (int i = 0; i < 32; i++) hmac_input[33 + i] = privkey32[i];
    for (int i = 0; i < 32; i++) hmac_input[65 + i] = hash32[i];
    hmac_sha256(K, hmac_input, 97, K);

    // Step e: V = HMAC_K(V)
    hmac_sha256(K, V, 32, V);

    // Step f: K = HMAC_K(V || 0x01 || privkey || hash)
    for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
    hmac_input[32] = 0x01;
    // privkey and hash already in place
    hmac_sha256(K, hmac_input, 97, K);

    // Step g: V = HMAC_K(V)
    hmac_sha256(K, V, 32, V);

    // Step h: generate k
    for (;;) {
        // V = HMAC_K(V)
        hmac_sha256(K, V, 32, V);

        // T = V (for secp256k1, qlen = 256 = hlen, so one round suffices)
        secp256k1::load_scalar(k_out, V);

        // Check k is in [1, n-1]
        if (!secp256k1::is_zero(k_out) && secp256k1::cmp(k_out, secp256k1::N) < 0) {
            return;
        }

        // k is invalid, update K and V and retry
        for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
        hmac_input[32] = 0x00;
        hmac_sha256(K, hmac_input, 33, K);
        hmac_sha256(K, V, 32, V);
    }
}

} // namespace hmac
