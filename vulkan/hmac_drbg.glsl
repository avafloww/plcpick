// HMAC-SHA256 and RFC 6979 deterministic nonce generation for Vulkan/GLSL
//
// This file is #included by .comp files. Do NOT put #version or #extension here.
// Requires: sha256.glsl and secp256k1.glsl to be included before this file.

// Load a 32-byte big-endian scalar from uint8_t array into U256 (little-endian limbs)
void hmac_load_scalar_bytes(inout U256 r, in uint8_t bytes32[32]) {
    for (int i = 0; i < 8; i++) {
        int offset = (7 - i) * 4;
        r.d[i] = (uint(bytes32[offset]) << 24) |
                 (uint(bytes32[offset + 1]) << 16) |
                 (uint(bytes32[offset + 2]) << 8) |
                 uint(bytes32[offset + 3]);
    }
}

// Store U256 as 32-byte big-endian into uint8_t array
void hmac_store_scalar_bytes(inout uint8_t bytes32[32], in U256 a) {
    for (int i = 0; i < 8; i++) {
        int offset = (7 - i) * 4;
        bytes32[offset]     = uint8_t((a.d[i] >> 24) & 0xFFu);
        bytes32[offset + 1] = uint8_t((a.d[i] >> 16) & 0xFFu);
        bytes32[offset + 2] = uint8_t((a.d[i] >> 8) & 0xFFu);
        bytes32[offset + 3] = uint8_t(a.d[i] & 0xFFu);
    }
}

// HMAC-SHA256: key is always 32 bytes, message up to 97 bytes
void hmac_sha256(in uint8_t key32[32], in uint8_t msg[97], uint msg_len, inout uint8_t out32[32]) {
    uint8_t ipad[128]; // 64 bytes used, but we need 128 for process_block signature
    uint8_t opad[128];

    // Build ipad and opad
    for (int i = 0; i < 32; i++) {
        ipad[i] = uint8_t(uint(key32[i]) ^ 0x36u);
        opad[i] = uint8_t(uint(key32[i]) ^ 0x5cu);
    }
    for (int i = 32; i < 64; i++) {
        ipad[i] = uint8_t(0x36u);
        opad[i] = uint8_t(0x5cu);
    }
    for (int i = 64; i < 128; i++) {
        ipad[i] = uint8_t(0u);
        opad[i] = uint8_t(0u);
    }

    // inner = SHA256(ipad || msg)
    uint h[8];
    sha256_init(h);
    sha256_process_block(h, ipad, 0u);

    // Now hash the message with the ipad state
    uint total_inner = 64u + msg_len;
    uint i = 0u;

    // Build block buffer for remaining message processing
    uint8_t pad[128];

    // Process complete 64-byte message blocks
    while (i + 64u <= msg_len) {
        for (uint j = 0u; j < 64u; j++) pad[j] = msg[i + j];
        for (uint j = 64u; j < 128u; j++) pad[j] = uint8_t(0u);
        sha256_process_block(h, pad, 0u);
        i += 64u;
    }

    // Pad the remaining bytes
    uint rem = msg_len - i;
    for (uint j = 0u; j < 128u; j++) pad[j] = uint8_t(0u);
    for (uint j = 0u; j < rem; j++) pad[j] = msg[i + j];
    pad[rem] = uint8_t(0x80u);

    uint pad_len = (rem < 56u) ? 64u : 128u;

    uint64_t bit_len = uint64_t(total_inner) * 8ul;
    pad[pad_len - 8u] = uint8_t(uint(bit_len >> 56) & 0xFFu);
    pad[pad_len - 7u] = uint8_t(uint(bit_len >> 48) & 0xFFu);
    pad[pad_len - 6u] = uint8_t(uint(bit_len >> 40) & 0xFFu);
    pad[pad_len - 5u] = uint8_t(uint(bit_len >> 32) & 0xFFu);
    pad[pad_len - 4u] = uint8_t(uint(bit_len >> 24) & 0xFFu);
    pad[pad_len - 3u] = uint8_t(uint(bit_len >> 16) & 0xFFu);
    pad[pad_len - 2u] = uint8_t(uint(bit_len >> 8) & 0xFFu);
    pad[pad_len - 1u] = uint8_t(uint(bit_len) & 0xFFu);

    for (uint j = 0u; j < pad_len; j += 64u) {
        sha256_process_block(h, pad, j);
    }

    uint8_t inner_hash[32];
    for (int j = 0; j < 8; j++) {
        inner_hash[j * 4]     = uint8_t((h[j] >> 24) & 0xFFu);
        inner_hash[j * 4 + 1] = uint8_t((h[j] >> 16) & 0xFFu);
        inner_hash[j * 4 + 2] = uint8_t((h[j] >> 8) & 0xFFu);
        inner_hash[j * 4 + 3] = uint8_t(h[j] & 0xFFu);
    }

    // outer = SHA256(opad || inner_hash)
    sha256_init(h);
    sha256_process_block(h, opad, 0u);

    // inner_hash is 32 bytes, needs padding in a 64-byte block
    uint8_t outer_block[128];
    for (int j = 0; j < 32; j++) outer_block[j] = inner_hash[j];
    outer_block[32] = uint8_t(0x80u);
    for (int j = 33; j < 56; j++) outer_block[j] = uint8_t(0u);
    // total length = 64 + 32 = 96 bytes = 768 bits
    uint64_t outer_bits = 768ul;
    outer_block[56] = uint8_t(uint(outer_bits >> 56) & 0xFFu);
    outer_block[57] = uint8_t(uint(outer_bits >> 48) & 0xFFu);
    outer_block[58] = uint8_t(uint(outer_bits >> 40) & 0xFFu);
    outer_block[59] = uint8_t(uint(outer_bits >> 32) & 0xFFu);
    outer_block[60] = uint8_t(uint(outer_bits >> 24) & 0xFFu);
    outer_block[61] = uint8_t(uint(outer_bits >> 16) & 0xFFu);
    outer_block[62] = uint8_t(uint(outer_bits >> 8) & 0xFFu);
    outer_block[63] = uint8_t(uint(outer_bits) & 0xFFu);
    for (int j = 64; j < 128; j++) outer_block[j] = uint8_t(0u);

    sha256_process_block(h, outer_block, 0u);

    for (int j = 0; j < 8; j++) {
        out32[j * 4]     = uint8_t((h[j] >> 24) & 0xFFu);
        out32[j * 4 + 1] = uint8_t((h[j] >> 16) & 0xFFu);
        out32[j * 4 + 2] = uint8_t((h[j] >> 8) & 0xFFu);
        out32[j * 4 + 3] = uint8_t(h[j] & 0xFFu);
    }
}

// RFC 6979 deterministic nonce generation for secp256k1
// Inputs: private key (32 bytes big-endian), message hash (32 bytes)
// Output: nonce k as U256
void hmac_rfc6979_nonce(inout U256 k_out, in uint8_t privkey32[32], in uint8_t hash32[32]) {
    // Step b: V = 0x01 01 01 ... 01 (32 bytes)
    uint8_t V[32];
    for (int i = 0; i < 32; i++) V[i] = uint8_t(0x01u);

    // Step c: K = 0x00 00 00 ... 00 (32 bytes)
    uint8_t K[32];
    for (int i = 0; i < 32; i++) K[i] = uint8_t(0x00u);

    // Step d: K = HMAC_K(V || 0x00 || privkey || hash)
    uint8_t hmac_input[97]; // 32 + 1 + 32 + 32
    for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
    hmac_input[32] = uint8_t(0x00u);
    for (int i = 0; i < 32; i++) hmac_input[33 + i] = privkey32[i];
    for (int i = 0; i < 32; i++) hmac_input[65 + i] = hash32[i];
    hmac_sha256(K, hmac_input, 97u, K);

    // Step e: V = HMAC_K(V)
    uint8_t hmac_short[97];
    for (int i = 0; i < 32; i++) hmac_short[i] = V[i];
    for (int i = 32; i < 97; i++) hmac_short[i] = uint8_t(0u);
    hmac_sha256(K, hmac_short, 32u, V);

    // Step f: K = HMAC_K(V || 0x01 || privkey || hash)
    for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
    hmac_input[32] = uint8_t(0x01u);
    // privkey and hash already in place from step d
    for (int i = 0; i < 32; i++) hmac_input[33 + i] = privkey32[i];
    for (int i = 0; i < 32; i++) hmac_input[65 + i] = hash32[i];
    hmac_sha256(K, hmac_input, 97u, K);

    // Step g: V = HMAC_K(V)
    for (int i = 0; i < 32; i++) hmac_short[i] = V[i];
    for (int i = 32; i < 97; i++) hmac_short[i] = uint8_t(0u);
    hmac_sha256(K, hmac_short, 32u, V);

    // Step h: generate k
    for (int iter = 0; iter < 10; iter++) {
        // V = HMAC_K(V)
        for (int i = 0; i < 32; i++) hmac_short[i] = V[i];
        for (int i = 32; i < 97; i++) hmac_short[i] = uint8_t(0u);
        hmac_sha256(K, hmac_short, 32u, V);

        // T = V (for secp256k1, qlen = 256 = hlen, so one round suffices)
        hmac_load_scalar_bytes(k_out, V);

        // Check k is in [1, n-1]
        if (!secp_is_zero(k_out) && secp_cmp(k_out, SECP_N) < 0) {
            return;
        }

        // k is invalid, update K and V and retry
        for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
        hmac_input[32] = uint8_t(0x00u);
        for (int i = 33; i < 97; i++) hmac_input[i] = uint8_t(0u);
        hmac_sha256(K, hmac_input, 33u, K);
        for (int i = 0; i < 32; i++) hmac_short[i] = V[i];
        hmac_sha256(K, hmac_short, 32u, V);
    }
}
