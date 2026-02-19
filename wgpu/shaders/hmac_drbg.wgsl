// HMAC-SHA256 and RFC 6979 deterministic nonce generation for wgpu/WGSL
//
// Depends on: sha256.wgsl (sha256_process_block, SHA256_INIT, bytes_to_u32_be)
//             field.wgsl (array<u32, 8> as U256 convention)
//
// Byte arrays use array<u32, N> with one byte value (0-255) per u32 element.

// secp256k1 curve order N in little-endian u32 limbs
const SECP_N = array<u32, 8>(
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
);

fn scalar_is_zero(a: array<u32, 8>) -> bool {
    return (a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7]) == 0u;
}

// Compare two 256-bit scalars (little-endian limbs). Returns -1, 0, or 1.
fn scalar_cmp(a: array<u32, 8>, b: array<u32, 8>) -> i32 {
    for (var i = 7i; i >= 0i; i--) {
        if (a[i] < b[i]) { return -1i; }
        if (a[i] > b[i]) { return 1i; }
    }
    return 0i;
}

// Load 32 big-endian bytes (one byte per u32 element) into little-endian u32 limbs
fn hmac_load_scalar_bytes(bytes32: ptr<function, array<u32, 32>>) -> array<u32, 8> {
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let offset = (7u - i) * 4u;
        r[i] = ((*bytes32)[offset] << 24u) |
               ((*bytes32)[offset + 1u] << 16u) |
               ((*bytes32)[offset + 2u] << 8u) |
               (*bytes32)[offset + 3u];
    }
    return r;
}

// Store little-endian u32 limbs as 32 big-endian bytes (one byte per u32 element)
fn hmac_store_scalar_bytes(a: array<u32, 8>, out: ptr<function, array<u32, 32>>) {
    for (var i = 0u; i < 8u; i++) {
        let offset = (7u - i) * 4u;
        (*out)[offset]      = (a[i] >> 24u) & 0xFFu;
        (*out)[offset + 1u] = (a[i] >> 16u) & 0xFFu;
        (*out)[offset + 2u] = (a[i] >> 8u) & 0xFFu;
        (*out)[offset + 3u] = a[i] & 0xFFu;
    }
}

// HMAC-SHA256. Key is 8 u32 words (SHA256 state-sized = 32 bytes as big-endian words).
// msg: up to 97 bytes (one byte per u32 element). msg_byte_len: actual length.
// Returns HMAC result as 8 u32 words (SHA256 state format).
fn hmac_sha256(key: array<u32, 8>, msg: ptr<function, array<u32, 97>>, msg_byte_len: u32) -> array<u32, 8> {
    // Build ipad (64 bytes) and opad (64 bytes) from the 32-byte key
    // Key is 8 u32 words in big-endian; extract individual bytes
    var ipad_block: array<u32, 16>;
    var opad_block: array<u32, 16>;

    // First 8 words: key XOR pad
    for (var i = 0u; i < 8u; i++) {
        let k = key[i];
        let kb0 = (k >> 24u) & 0xFFu;
        let kb1 = (k >> 16u) & 0xFFu;
        let kb2 = (k >> 8u) & 0xFFu;
        let kb3 = k & 0xFFu;
        ipad_block[i] = bytes_to_u32_be(kb0 ^ 0x36u, kb1 ^ 0x36u, kb2 ^ 0x36u, kb3 ^ 0x36u);
        opad_block[i] = bytes_to_u32_be(kb0 ^ 0x5Cu, kb1 ^ 0x5Cu, kb2 ^ 0x5Cu, kb3 ^ 0x5Cu);
    }
    // Remaining 8 words: pad byte repeated (key is zero-padded to 64 bytes)
    for (var i = 8u; i < 16u; i++) {
        ipad_block[i] = bytes_to_u32_be(0x36u, 0x36u, 0x36u, 0x36u);
        opad_block[i] = bytes_to_u32_be(0x5Cu, 0x5Cu, 0x5Cu, 0x5Cu);
    }

    // inner = SHA256(ipad || msg)
    var state = SHA256_INIT;
    sha256_process_block(&state, &ipad_block);

    // Process message after the ipad block
    let total_inner = 64u + msg_byte_len;
    var i = 0u;

    // Process complete 64-byte message blocks
    var block: array<u32, 16>;
    while (i + 64u <= msg_byte_len) {
        for (var j = 0u; j < 16u; j++) {
            let idx = i + j * 4u;
            block[j] = bytes_to_u32_be((*msg)[idx], (*msg)[idx + 1u], (*msg)[idx + 2u], (*msg)[idx + 3u]);
        }
        sha256_process_block(&state, &block);
        i += 64u;
    }

    // Pad remaining bytes
    let rem = msg_byte_len - i;
    var pad: array<u32, 128>;
    for (var j = 0u; j < 128u; j++) {
        pad[j] = 0u;
    }
    for (var j = 0u; j < rem; j++) {
        pad[j] = (*msg)[i + j];
    }
    pad[rem] = 0x80u;

    var pad_byte_len: u32;
    if (rem < 56u) {
        pad_byte_len = 64u;
    } else {
        pad_byte_len = 128u;
    }

    // Append bit length (big-endian 64-bit, but total_inner <= ~161, fits in u32)
    let inner_bit_len = total_inner * 8u;
    pad[pad_byte_len - 4u] = (inner_bit_len >> 24u) & 0xFFu;
    pad[pad_byte_len - 3u] = (inner_bit_len >> 16u) & 0xFFu;
    pad[pad_byte_len - 2u] = (inner_bit_len >> 8u) & 0xFFu;
    pad[pad_byte_len - 1u] = inner_bit_len & 0xFFu;

    // Process padded block(s)
    var block_off = 0u;
    while (block_off < pad_byte_len) {
        for (var j = 0u; j < 16u; j++) {
            let idx = block_off + j * 4u;
            block[j] = bytes_to_u32_be(pad[idx], pad[idx + 1u], pad[idx + 2u], pad[idx + 3u]);
        }
        sha256_process_block(&state, &block);
        block_off += 64u;
    }

    // inner_hash = state (as 8 u32 words)
    let inner_hash = state;

    // outer = SHA256(opad || inner_hash)
    state = SHA256_INIT;
    sha256_process_block(&state, &opad_block);

    // inner_hash is 32 bytes, pack into a single padded block
    var outer_block: array<u32, 16>;
    // First 8 words: inner_hash (already in big-endian word format)
    for (var j = 0u; j < 8u; j++) {
        outer_block[j] = inner_hash[j];
    }
    // Byte 32 = 0x80, rest zero
    outer_block[8] = 0x80000000u;
    for (var j = 9u; j < 14u; j++) {
        outer_block[j] = 0u;
    }
    // Total length = 64 + 32 = 96 bytes = 768 bits (big-endian u64)
    outer_block[14] = 0u;          // high 32 bits of bit length
    outer_block[15] = 768u;        // low 32 bits of bit length

    sha256_process_block(&state, &outer_block);

    return state;
}

// Convert HMAC key from byte array (32 bytes, one per u32) to 8 u32 words (big-endian packed)
fn hmac_key_from_bytes(key_bytes: ptr<function, array<u32, 32>>) -> array<u32, 8> {
    var key: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let idx = i * 4u;
        key[i] = bytes_to_u32_be((*key_bytes)[idx], (*key_bytes)[idx + 1u], (*key_bytes)[idx + 2u], (*key_bytes)[idx + 3u]);
    }
    return key;
}

// Convert 8 u32 words (big-endian packed) to byte array (32 bytes, one per u32)
fn hmac_key_to_bytes(key: array<u32, 8>, out: ptr<function, array<u32, 32>>) {
    for (var i = 0u; i < 8u; i++) {
        let idx = i * 4u;
        (*out)[idx]      = (key[i] >> 24u) & 0xFFu;
        (*out)[idx + 1u] = (key[i] >> 16u) & 0xFFu;
        (*out)[idx + 2u] = (key[i] >> 8u) & 0xFFu;
        (*out)[idx + 3u] = key[i] & 0xFFu;
    }
}

// RFC 6979 deterministic nonce generation for secp256k1
// privkey and hash are little-endian u32 limbs (field element format).
// Returns nonce k as little-endian u32 limbs.
fn hmac_rfc6979_nonce(privkey: array<u32, 8>, hash: array<u32, 8>) -> array<u32, 8> {
    // Convert privkey and hash from LE limbs to BE bytes for HMAC input
    var privkey_bytes: array<u32, 32>;
    var hash_bytes: array<u32, 32>;
    hmac_store_scalar_bytes(privkey, &privkey_bytes);
    hmac_store_scalar_bytes(hash, &hash_bytes);

    // Step b: V = 0x01 repeated 32 times
    var V_bytes: array<u32, 32>;
    for (var i = 0u; i < 32u; i++) { V_bytes[i] = 0x01u; }

    // Step c: K = 0x00 repeated 32 times
    var K_bytes: array<u32, 32>;
    for (var i = 0u; i < 32u; i++) { K_bytes[i] = 0x00u; }

    // Step d: K = HMAC_K(V || 0x00 || privkey || hash)
    var hmac_input: array<u32, 97>;
    for (var i = 0u; i < 32u; i++) { hmac_input[i] = V_bytes[i]; }
    hmac_input[32] = 0x00u;
    for (var i = 0u; i < 32u; i++) { hmac_input[33u + i] = privkey_bytes[i]; }
    for (var i = 0u; i < 32u; i++) { hmac_input[65u + i] = hash_bytes[i]; }

    var K = hmac_key_from_bytes(&K_bytes);
    K = hmac_sha256(K, &hmac_input, 97u);
    hmac_key_to_bytes(K, &K_bytes);

    // Step e: V = HMAC_K(V)
    var hmac_short: array<u32, 97>;
    for (var i = 0u; i < 32u; i++) { hmac_short[i] = V_bytes[i]; }
    for (var i = 32u; i < 97u; i++) { hmac_short[i] = 0u; }
    let V_words = hmac_sha256(K, &hmac_short, 32u);
    hmac_key_to_bytes(V_words, &V_bytes);

    // Step f: K = HMAC_K(V || 0x01 || privkey || hash)
    for (var i = 0u; i < 32u; i++) { hmac_input[i] = V_bytes[i]; }
    hmac_input[32] = 0x01u;
    for (var i = 0u; i < 32u; i++) { hmac_input[33u + i] = privkey_bytes[i]; }
    for (var i = 0u; i < 32u; i++) { hmac_input[65u + i] = hash_bytes[i]; }
    K = hmac_sha256(K, &hmac_input, 97u);
    hmac_key_to_bytes(K, &K_bytes);

    // Step g: V = HMAC_K(V)
    for (var i = 0u; i < 32u; i++) { hmac_short[i] = V_bytes[i]; }
    for (var i = 32u; i < 97u; i++) { hmac_short[i] = 0u; }
    let V_words2 = hmac_sha256(K, &hmac_short, 32u);
    hmac_key_to_bytes(V_words2, &V_bytes);

    // Step h: generate candidate k values
    for (var iter = 0u; iter < 10u; iter++) {
        // V = HMAC_K(V)
        for (var j = 0u; j < 32u; j++) { hmac_short[j] = V_bytes[j]; }
        for (var j = 32u; j < 97u; j++) { hmac_short[j] = 0u; }
        let V_candidate = hmac_sha256(K, &hmac_short, 32u);
        hmac_key_to_bytes(V_candidate, &V_bytes);

        // T = V (for secp256k1, qlen = 256 = hlen, one round suffices)
        let k_out = hmac_load_scalar_bytes(&V_bytes);

        // Check k is in [1, n-1]
        if (!scalar_is_zero(k_out) && scalar_cmp(k_out, SECP_N) < 0i) {
            return k_out;
        }

        // k is invalid — update K and V, retry
        for (var j = 0u; j < 32u; j++) { hmac_input[j] = V_bytes[j]; }
        hmac_input[32] = 0x00u;
        for (var j = 33u; j < 97u; j++) { hmac_input[j] = 0u; }
        K = hmac_sha256(K, &hmac_input, 33u);
        hmac_key_to_bytes(K, &K_bytes);

        for (var j = 0u; j < 32u; j++) { hmac_short[j] = V_bytes[j]; }
        for (var j = 32u; j < 97u; j++) { hmac_short[j] = 0u; }
        let V_retry = hmac_sha256(K, &hmac_short, 32u);
        hmac_key_to_bytes(V_retry, &V_bytes);
    }

    // Should never reach here for valid inputs
    return array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}
