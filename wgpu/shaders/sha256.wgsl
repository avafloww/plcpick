// SHA-256 implementation for wgpu/WGSL
// Reference: FIPS 180-4
//
// Byte arrays use array<u32, N> with one byte value (0-255) per u32 element.
// No u8 or u64 types in WGSL.

const SHA256_MAX_INPUT: u32 = 512u;

const SHA256_K = array<u32, 64>(
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

const SHA256_INIT = array<u32, 8>(
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
);

fn sha256_rotr(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

fn sha256_ch(x: u32, y: u32, z: u32) -> u32 {
    return (x & y) ^ (~x & z);
}

fn sha256_maj(x: u32, y: u32, z: u32) -> u32 {
    return (x & y) ^ (x & z) ^ (y & z);
}

fn sha256_sigma0(x: u32) -> u32 {
    return sha256_rotr(x, 2u) ^ sha256_rotr(x, 13u) ^ sha256_rotr(x, 22u);
}

fn sha256_sigma1(x: u32) -> u32 {
    return sha256_rotr(x, 6u) ^ sha256_rotr(x, 11u) ^ sha256_rotr(x, 25u);
}

fn sha256_gamma0(x: u32) -> u32 {
    return sha256_rotr(x, 7u) ^ sha256_rotr(x, 18u) ^ (x >> 3u);
}

fn sha256_gamma1(x: u32) -> u32 {
    return sha256_rotr(x, 17u) ^ sha256_rotr(x, 19u) ^ (x >> 10u);
}

// Pack 4 byte values into a big-endian u32 word
fn bytes_to_u32_be(b0: u32, b1: u32, b2: u32, b3: u32) -> u32 {
    return (b0 << 24u) | (b1 << 16u) | (b2 << 8u) | b3;
}

// Process a single 64-byte block. block is 16 u32 words (already big-endian packed).
fn sha256_process_block(state: ptr<function, array<u32, 8>>, block: ptr<function, array<u32, 16>>) {
    var w: array<u32, 64>;

    // Copy block words
    for (var i = 0u; i < 16u; i++) {
        w[i] = (*block)[i];
    }

    // Message schedule
    for (var i = 16u; i < 64u; i++) {
        w[i] = sha256_gamma1(w[i - 2u]) + w[i - 7u] + sha256_gamma0(w[i - 15u]) + w[i - 16u];
    }

    var a = (*state)[0]; var b = (*state)[1]; var c = (*state)[2]; var d = (*state)[3];
    var e = (*state)[4]; var f = (*state)[5]; var g = (*state)[6]; var hh = (*state)[7];

    for (var i = 0u; i < 64u; i++) {
        let t1 = hh + sha256_sigma1(e) + sha256_ch(e, f, g) + SHA256_K[i] + w[i];
        let t2 = sha256_sigma0(a) + sha256_maj(a, b, c);
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    (*state)[0] += a; (*state)[1] += b; (*state)[2] += c; (*state)[3] += d;
    (*state)[4] += e; (*state)[5] += f; (*state)[6] += g; (*state)[7] += hh;
}

// Hash an arbitrary-length message (up to SHA256_MAX_INPUT bytes).
// data: one byte value per u32 element. byte_len: actual data length.
// Returns SHA256 state as 8 u32 words (big-endian word order).
fn sha256_hash(data: ptr<function, array<u32, 512>>, byte_len: u32) -> array<u32, 8> {
    var state = SHA256_INIT;

    // Process complete 64-byte blocks
    var pos = 0u;
    var block: array<u32, 16>;
    while (pos + 64u <= byte_len) {
        for (var j = 0u; j < 16u; j++) {
            let idx = pos + j * 4u;
            block[j] = bytes_to_u32_be((*data)[idx], (*data)[idx + 1u], (*data)[idx + 2u], (*data)[idx + 3u]);
        }
        sha256_process_block(&state, &block);
        pos += 64u;
    }

    // Build padded final block(s) — up to 128 bytes (2 blocks)
    var pad: array<u32, 128>;
    for (var j = 0u; j < 128u; j++) {
        pad[j] = 0u;
    }
    let rem = byte_len - pos;
    for (var j = 0u; j < rem; j++) {
        pad[j] = (*data)[pos + j];
    }
    pad[rem] = 0x80u;

    var pad_byte_len: u32;
    if (rem < 56u) {
        pad_byte_len = 64u;
    } else {
        pad_byte_len = 128u;
    }

    // Append bit length as big-endian 64-bit value.
    // Our inputs are <= 512 bytes, so bit_len fits in u32 (max 4096).
    // High 4 bytes are always zero.
    let bit_len = byte_len * 8u;
    pad[pad_byte_len - 4u] = (bit_len >> 24u) & 0xFFu;
    pad[pad_byte_len - 3u] = (bit_len >> 16u) & 0xFFu;
    pad[pad_byte_len - 2u] = (bit_len >> 8u) & 0xFFu;
    pad[pad_byte_len - 1u] = bit_len & 0xFFu;

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

    return state;
}
