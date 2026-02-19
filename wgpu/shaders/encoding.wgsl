// Base32, Base58, Base64url encoding for wgpu/WGSL
// One ASCII char value per u32 element throughout.

const ENC_BASE32_ALPHABET = array<u32, 32>(
    0x61u, 0x62u, 0x63u, 0x64u, // abcd
    0x65u, 0x66u, 0x67u, 0x68u, // efgh
    0x69u, 0x6Au, 0x6Bu, 0x6Cu, // ijkl
    0x6Du, 0x6Eu, 0x6Fu, 0x70u, // mnop
    0x71u, 0x72u, 0x73u, 0x74u, // qrst
    0x75u, 0x76u, 0x77u, 0x78u, // uvwx
    0x79u, 0x7Au, 0x32u, 0x33u, // yz23
    0x34u, 0x35u, 0x36u, 0x37u, // 4567
);

// Encode 15 bytes to 24-char base32 lowercase (no padding)
// 15 bytes = 120 bits, 120/5 = 24 characters exactly
fn enc_base32_15bytes(data: ptr<function, array<u32, 15>>, out: ptr<function, array<u32, 24>>) {
    var oi = 0u;
    var accum = 0u;
    var bits = 0u;

    for (var i = 0u; i < 15u; i++) {
        accum = (accum << 8u) | (*data)[i];
        bits += 8u;
        while (bits >= 5u) {
            bits -= 5u;
            (*out)[oi] = ENC_BASE32_ALPHABET[(accum >> bits) & 0x1Fu];
            oi++;
        }
    }
}

// --- Base58 (Bitcoin alphabet) ---

const ENC_BASE58_ALPHABET = array<u32, 58>(
    0x31u, 0x32u, 0x33u, 0x34u, // 1234
    0x35u, 0x36u, 0x37u, 0x38u, // 5678
    0x39u, 0x41u, 0x42u, 0x43u, // 9ABC
    0x44u, 0x45u, 0x46u, 0x47u, // DEFG
    0x48u, 0x4Au, 0x4Bu, 0x4Cu, // HJKL
    0x4Du, 0x4Eu, 0x50u, 0x51u, // MNPQ
    0x52u, 0x53u, 0x54u, 0x55u, // RSTU
    0x56u, 0x57u, 0x58u, 0x59u, // VWXY
    0x5Au, 0x61u, 0x62u, 0x63u, // Zabc
    0x64u, 0x65u, 0x66u, 0x67u, // defg
    0x68u, 0x69u, 0x6Au, 0x6Bu, // hijk
    0x6Du, 0x6Eu, 0x6Fu, 0x70u, // mnop
    0x71u, 0x72u, 0x73u, 0x74u, // qrst
    0x75u, 0x76u, 0x77u, 0x78u, // uvwx
    0x79u, 0x7Au,               // yz
);

// Encode 35 bytes to exactly 48 base58 chars
// (multicodec prefix + compressed pubkey)
fn enc_base58_35bytes(data: ptr<function, array<u32, 35>>, out: ptr<function, array<u32, 48>>) {
    // Count leading zeros
    var leading_zeros = 0u;
    while (leading_zeros < 35u && (*data)[leading_zeros] == 0u) {
        leading_zeros++;
    }

    // Work buffer for repeated division
    var work: array<u32, 35>;
    for (var i = 0u; i < 35u; i++) {
        work[i] = (*data)[i];
    }

    // Temporary reverse buffer
    var tmp: array<u32, 48>;
    var tmp_len = 0u;

    var start = leading_zeros;
    while (start < 35u) {
        var remainder = 0u;
        var new_start = start;
        var found_nonzero = false;

        for (var i = start; i < 35u; i++) {
            let digit = (remainder << 8u) + work[i];
            work[i] = digit / 58u;
            remainder = digit % 58u;
            if (work[i] != 0u && !found_nonzero) {
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
    var oi = 0u;
    for (var i = 0u; i < leading_zeros; i++) {
        (*out)[oi] = 0x31u; // '1'
        oi++;
    }

    // Reverse tmp into out
    for (var i = i32(tmp_len) - 1; i >= 0; i--) {
        (*out)[oi] = tmp[i];
        oi++;
    }
}

// --- Base64url (no padding) ---

const ENC_BASE64URL_ALPHABET = array<u32, 64>(
    0x41u, 0x42u, 0x43u, 0x44u, // ABCD
    0x45u, 0x46u, 0x47u, 0x48u, // EFGH
    0x49u, 0x4Au, 0x4Bu, 0x4Cu, // IJKL
    0x4Du, 0x4Eu, 0x4Fu, 0x50u, // MNOP
    0x51u, 0x52u, 0x53u, 0x54u, // QRST
    0x55u, 0x56u, 0x57u, 0x58u, // UVWX
    0x59u, 0x5Au, 0x61u, 0x62u, // YZab
    0x63u, 0x64u, 0x65u, 0x66u, // cdef
    0x67u, 0x68u, 0x69u, 0x6Au, // ghij
    0x6Bu, 0x6Cu, 0x6Du, 0x6Eu, // klmn
    0x6Fu, 0x70u, 0x71u, 0x72u, // opqr
    0x73u, 0x74u, 0x75u, 0x76u, // stuv
    0x77u, 0x78u, 0x79u, 0x7Au, // wxyz
    0x30u, 0x31u, 0x32u, 0x33u, // 0123
    0x34u, 0x35u, 0x36u, 0x37u, // 4567
    0x38u, 0x39u, 0x2Du, 0x5Fu, // 89-_
);

// Encode 64 bytes to exactly 86 base64url chars (no padding)
// 64 bytes = 21*3 + 1, so 21*4 + 2 = 86 characters
fn enc_base64url_64bytes(data: ptr<function, array<u32, 64>>, out: ptr<function, array<u32, 86>>) {
    var oi = 0u;
    var i = 0u;

    // Process 3 bytes at a time (21 full groups)
    while (i + 3u <= 64u) {
        let n = ((*data)[i] << 16u) | ((*data)[i + 1u] << 8u) | (*data)[i + 2u];
        (*out)[oi]      = ENC_BASE64URL_ALPHABET[(n >> 18u) & 0x3Fu];
        (*out)[oi + 1u] = ENC_BASE64URL_ALPHABET[(n >> 12u) & 0x3Fu];
        (*out)[oi + 2u] = ENC_BASE64URL_ALPHABET[(n >> 6u) & 0x3Fu];
        (*out)[oi + 3u] = ENC_BASE64URL_ALPHABET[n & 0x3Fu];
        oi += 4u;
        i += 3u;
    }

    // Handle remainder: 64 mod 3 = 1, so 1 byte remaining
    let rem = 64u - i;
    if (rem == 1u) {
        let n = (*data)[i] << 16u;
        (*out)[oi]      = ENC_BASE64URL_ALPHABET[(n >> 18u) & 0x3Fu];
        (*out)[oi + 1u] = ENC_BASE64URL_ALPHABET[(n >> 12u) & 0x3Fu];
    } else if (rem == 2u) {
        let n = ((*data)[i] << 16u) | ((*data)[i + 1u] << 8u);
        (*out)[oi]      = ENC_BASE64URL_ALPHABET[(n >> 18u) & 0x3Fu];
        (*out)[oi + 1u] = ENC_BASE64URL_ALPHABET[(n >> 12u) & 0x3Fu];
        (*out)[oi + 2u] = ENC_BASE64URL_ALPHABET[(n >> 6u) & 0x3Fu];
    }
}
