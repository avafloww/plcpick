// Base32, Base58, Base64url encoding for Vulkan/GLSL
//
// This file is #included by .comp files. Do NOT put #version or #extension here.
// Requires: GL_EXT_shader_explicit_arithmetic_types_int8

const uint8_t ENC_BASE32_ALPHABET[32] = uint8_t[32](
    uint8_t(0x61u), uint8_t(0x62u), uint8_t(0x63u), uint8_t(0x64u), // abcd
    uint8_t(0x65u), uint8_t(0x66u), uint8_t(0x67u), uint8_t(0x68u), // efgh
    uint8_t(0x69u), uint8_t(0x6Au), uint8_t(0x6Bu), uint8_t(0x6Cu), // ijkl
    uint8_t(0x6Du), uint8_t(0x6Eu), uint8_t(0x6Fu), uint8_t(0x70u), // mnop
    uint8_t(0x71u), uint8_t(0x72u), uint8_t(0x73u), uint8_t(0x74u), // qrst
    uint8_t(0x75u), uint8_t(0x76u), uint8_t(0x77u), uint8_t(0x78u), // uvwx
    uint8_t(0x79u), uint8_t(0x7Au), uint8_t(0x32u), uint8_t(0x33u), // yz23
    uint8_t(0x34u), uint8_t(0x35u), uint8_t(0x36u), uint8_t(0x37u)  // 4567
);

// Encode 15 bytes to 24-char base32 lowercase (no padding)
// 15 bytes = 120 bits, 120/5 = 24 characters exactly
void enc_base32_encode_15bytes(in uint8_t data15[15], inout uint8_t out24[24]) {
    uint oi = 0u;
    uint accum = 0u;
    uint bits = 0u;

    for (uint i = 0u; i < 15u; i++) {
        accum = (accum << 8) | uint(data15[i]);
        bits += 8u;
        while (bits >= 5u) {
            bits -= 5u;
            out24[oi] = ENC_BASE32_ALPHABET[(accum >> bits) & 0x1Fu];
            oi++;
        }
    }
}

// --- Base58 (Bitcoin alphabet) ---

const uint8_t ENC_BASE58_ALPHABET[58] = uint8_t[58](
    uint8_t(0x31u), uint8_t(0x32u), uint8_t(0x33u), uint8_t(0x34u), // 1234
    uint8_t(0x35u), uint8_t(0x36u), uint8_t(0x37u), uint8_t(0x38u), // 5678
    uint8_t(0x39u), uint8_t(0x41u), uint8_t(0x42u), uint8_t(0x43u), // 9ABC
    uint8_t(0x44u), uint8_t(0x45u), uint8_t(0x46u), uint8_t(0x47u), // DEFG
    uint8_t(0x48u), uint8_t(0x4Au), uint8_t(0x4Bu), uint8_t(0x4Cu), // HJKL
    uint8_t(0x4Du), uint8_t(0x4Eu), uint8_t(0x50u), uint8_t(0x51u), // MNPQ
    uint8_t(0x52u), uint8_t(0x53u), uint8_t(0x54u), uint8_t(0x55u), // RSTU
    uint8_t(0x56u), uint8_t(0x57u), uint8_t(0x58u), uint8_t(0x59u), // VWXY
    uint8_t(0x5Au), uint8_t(0x61u), uint8_t(0x62u), uint8_t(0x63u), // Zabc
    uint8_t(0x64u), uint8_t(0x65u), uint8_t(0x66u), uint8_t(0x67u), // defg
    uint8_t(0x68u), uint8_t(0x69u), uint8_t(0x6Au), uint8_t(0x6Bu), // hijk
    uint8_t(0x6Du), uint8_t(0x6Eu), uint8_t(0x6Fu), uint8_t(0x70u), // mnop
    uint8_t(0x71u), uint8_t(0x72u), uint8_t(0x73u), uint8_t(0x74u), // qrst
    uint8_t(0x75u), uint8_t(0x76u), uint8_t(0x77u), uint8_t(0x78u), // uvwx
    uint8_t(0x79u), uint8_t(0x7Au)                                   // yz
);

// Encode 35 bytes to exactly 48 base58 chars
// (multicodec prefix + compressed pubkey)
void enc_base58_encode_35bytes(in uint8_t data35[35], inout uint8_t out48[48]) {
    // Count leading zeros
    uint leading_zeros = 0u;
    while (leading_zeros < 35u && data35[leading_zeros] == uint8_t(0u)) {
        leading_zeros++;
    }

    // Work buffer for repeated division
    uint8_t work[35];
    for (uint i = 0u; i < 35u; i++) work[i] = data35[i];

    // Temporary reverse buffer
    uint8_t tmp[48];
    uint tmp_len = 0u;

    uint start = leading_zeros;
    while (start < 35u) {
        uint remainder = 0u;
        uint new_start = start;
        bool found_nonzero = false;

        for (uint i = start; i < 35u; i++) {
            uint digit = (remainder << 8) + uint(work[i]);
            work[i] = uint8_t(digit / 58u);
            remainder = digit % 58u;
            if (uint(work[i]) != 0u && !found_nonzero) {
                new_start = i;
                found_nonzero = true;
            }
        }

        tmp[tmp_len] = ENC_BASE58_ALPHABET[remainder];
        tmp_len++;

        if (!found_nonzero) {
            start = 35u; // all zeros, done
        } else {
            start = new_start;
        }
    }

    // Prepend '1' for each leading zero byte
    uint oi = 0u;
    for (uint i = 0u; i < leading_zeros; i++) {
        out48[oi] = uint8_t(0x31u); // '1'
        oi++;
    }

    // Reverse tmp into out
    for (int i = int(tmp_len) - 1; i >= 0; i--) {
        out48[oi] = tmp[i];
        oi++;
    }
}

// --- Base64url (no padding) ---

const uint8_t ENC_BASE64URL_ALPHABET[64] = uint8_t[64](
    uint8_t(0x41u), uint8_t(0x42u), uint8_t(0x43u), uint8_t(0x44u), // ABCD
    uint8_t(0x45u), uint8_t(0x46u), uint8_t(0x47u), uint8_t(0x48u), // EFGH
    uint8_t(0x49u), uint8_t(0x4Au), uint8_t(0x4Bu), uint8_t(0x4Cu), // IJKL
    uint8_t(0x4Du), uint8_t(0x4Eu), uint8_t(0x4Fu), uint8_t(0x50u), // MNOP
    uint8_t(0x51u), uint8_t(0x52u), uint8_t(0x53u), uint8_t(0x54u), // QRST
    uint8_t(0x55u), uint8_t(0x56u), uint8_t(0x57u), uint8_t(0x58u), // UVWX
    uint8_t(0x59u), uint8_t(0x5Au), uint8_t(0x61u), uint8_t(0x62u), // YZab
    uint8_t(0x63u), uint8_t(0x64u), uint8_t(0x65u), uint8_t(0x66u), // cdef
    uint8_t(0x67u), uint8_t(0x68u), uint8_t(0x69u), uint8_t(0x6Au), // ghij
    uint8_t(0x6Bu), uint8_t(0x6Cu), uint8_t(0x6Du), uint8_t(0x6Eu), // klmn
    uint8_t(0x6Fu), uint8_t(0x70u), uint8_t(0x71u), uint8_t(0x72u), // opqr
    uint8_t(0x73u), uint8_t(0x74u), uint8_t(0x75u), uint8_t(0x76u), // stuv
    uint8_t(0x77u), uint8_t(0x78u), uint8_t(0x79u), uint8_t(0x7Au), // wxyz
    uint8_t(0x30u), uint8_t(0x31u), uint8_t(0x32u), uint8_t(0x33u), // 0123
    uint8_t(0x34u), uint8_t(0x35u), uint8_t(0x36u), uint8_t(0x37u), // 4567
    uint8_t(0x38u), uint8_t(0x39u), uint8_t(0x2Du), uint8_t(0x5Fu)  // 89-_
);

// Encode 64 bytes to exactly 86 base64url chars (no padding)
// 64 bytes = 21*3 + 1, so 21*4 + 2 = 86 characters
void enc_base64url_encode_64bytes(in uint8_t data64[64], inout uint8_t out86[86]) {
    uint oi = 0u;
    uint i = 0u;

    // Process 3 bytes at a time (21 full groups)
    while (i + 3u <= 64u) {
        uint n = (uint(data64[i]) << 16) | (uint(data64[i + 1u]) << 8) | uint(data64[i + 2u]);
        out86[oi]     = ENC_BASE64URL_ALPHABET[(n >> 18) & 0x3Fu];
        out86[oi + 1u] = ENC_BASE64URL_ALPHABET[(n >> 12) & 0x3Fu];
        out86[oi + 2u] = ENC_BASE64URL_ALPHABET[(n >> 6) & 0x3Fu];
        out86[oi + 3u] = ENC_BASE64URL_ALPHABET[n & 0x3Fu];
        oi += 4u;
        i += 3u;
    }

    // Handle remainder: 64 mod 3 = 1, so 1 byte remaining
    uint rem = 64u - i;
    if (rem == 1u) {
        uint n = uint(data64[i]) << 16;
        out86[oi]     = ENC_BASE64URL_ALPHABET[(n >> 18) & 0x3Fu];
        out86[oi + 1u] = ENC_BASE64URL_ALPHABET[(n >> 12) & 0x3Fu];
    } else if (rem == 2u) {
        uint n = (uint(data64[i]) << 16) | (uint(data64[i + 1u]) << 8);
        out86[oi]     = ENC_BASE64URL_ALPHABET[(n >> 18) & 0x3Fu];
        out86[oi + 1u] = ENC_BASE64URL_ALPHABET[(n >> 12) & 0x3Fu];
        out86[oi + 2u] = ENC_BASE64URL_ALPHABET[(n >> 6) & 0x3Fu];
    }
}
