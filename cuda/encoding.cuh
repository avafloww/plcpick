#pragma once
#include <cstdint>

// Base32, Base58, Base64url encoding for CUDA

namespace encoding {

// --- Base32 (RFC 4648, lowercase, no padding) ---

__device__ void base32_encode(const uint8_t *data, uint32_t len, char *out, uint32_t *out_len) {
    static const char ALPHABET[] = "abcdefghijklmnopqrstuvwxyz234567";

    uint32_t oi = 0;
    uint32_t buffer = 0;
    uint32_t bits = 0;

    for (uint32_t i = 0; i < len; i++) {
        buffer = (buffer << 8) | data[i];
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            out[oi++] = ALPHABET[(buffer >> bits) & 0x1F];
        }
    }

    if (bits > 0) {
        out[oi++] = ALPHABET[(buffer << (5 - bits)) & 0x1F];
    }

    *out_len = oi;
}

// Encode first 15 bytes of SHA256 hash to 24-char base32 lowercase suffix
// (15 bytes = 120 bits, 120/5 = 24 characters, no remainder)
__device__ void base32_encode_15bytes(const uint8_t *data15, char *out24) {
    static const char ALPHABET[] = "abcdefghijklmnopqrstuvwxyz234567";

    uint32_t oi = 0;
    uint32_t buffer = 0;
    uint32_t bits = 0;

    for (uint32_t i = 0; i < 15; i++) {
        buffer = (buffer << 8) | data15[i];
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            out24[oi++] = ALPHABET[(buffer >> bits) & 0x1F];
        }
    }
    // 15 * 8 = 120 bits, 120 / 5 = 24 chars, no remainder
}

// --- Base58 (Bitcoin alphabet) ---

__device__ void base58_encode(const uint8_t *data, uint32_t len, char *out, uint32_t *out_len) {
    static const char ALPHABET[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Count leading zeros
    uint32_t leading_zeros = 0;
    while (leading_zeros < len && data[leading_zeros] == 0) {
        leading_zeros++;
    }

    // Work buffer: base58 encoding of up to 64 bytes needs at most ~88 chars
    // For our 35-byte input, output is always 48 chars
    char tmp[90];
    uint32_t tmp_len = 0;

    // Convert from base-256 to base-58 using repeated division
    // Work on a copy of the data
    uint8_t work[64];
    for (uint32_t i = 0; i < len; i++) work[i] = data[i];

    uint32_t start = leading_zeros;
    while (start < len) {
        uint32_t remainder = 0;
        uint32_t new_start = start;
        bool found_nonzero = false;

        for (uint32_t i = start; i < len; i++) {
            uint32_t digit = (remainder << 8) + work[i];
            work[i] = (uint8_t)(digit / 58);
            remainder = digit % 58;
            if (work[i] != 0 && !found_nonzero) {
                new_start = i;
                found_nonzero = true;
            }
        }

        tmp[tmp_len++] = ALPHABET[remainder];
        if (!found_nonzero) {
            start = len; // all zeros, we're done
        } else {
            start = new_start;
        }
    }

    // Prepend '1' for each leading zero byte
    uint32_t oi = 0;
    for (uint32_t i = 0; i < leading_zeros; i++) {
        out[oi++] = '1';
    }

    // Reverse tmp into out
    for (int i = (int)tmp_len - 1; i >= 0; i--) {
        out[oi++] = tmp[i];
    }

    *out_len = oi;
}

// Encode 35 bytes (multicodec prefix + compressed pubkey) to exactly 48 base58 chars
__device__ void base58_encode_35bytes(const uint8_t *data35, char *out48) {
    uint32_t out_len;
    base58_encode(data35, 35, out48, &out_len);
    // Should always be 48 for our input (first byte 0xe7, never produces leading zeros)
}

// --- Base64url (no padding) ---

__device__ void base64url_encode(const uint8_t *data, uint32_t len, char *out, uint32_t *out_len) {
    static const char ALPHABET[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    uint32_t oi = 0;
    uint32_t i = 0;

    // Process 3 bytes at a time
    while (i + 3 <= len) {
        uint32_t n = ((uint32_t)data[i] << 16) | ((uint32_t)data[i + 1] << 8) | data[i + 2];
        out[oi++] = ALPHABET[(n >> 18) & 0x3F];
        out[oi++] = ALPHABET[(n >> 12) & 0x3F];
        out[oi++] = ALPHABET[(n >> 6) & 0x3F];
        out[oi++] = ALPHABET[n & 0x3F];
        i += 3;
    }

    // Handle remainder (no padding)
    uint32_t rem = len - i;
    if (rem == 1) {
        uint32_t n = (uint32_t)data[i] << 16;
        out[oi++] = ALPHABET[(n >> 18) & 0x3F];
        out[oi++] = ALPHABET[(n >> 12) & 0x3F];
    } else if (rem == 2) {
        uint32_t n = ((uint32_t)data[i] << 16) | ((uint32_t)data[i + 1] << 8);
        out[oi++] = ALPHABET[(n >> 18) & 0x3F];
        out[oi++] = ALPHABET[(n >> 12) & 0x3F];
        out[oi++] = ALPHABET[(n >> 6) & 0x3F];
    }

    *out_len = oi;
}

// Encode 64-byte ECDSA signature to exactly 86 base64url chars (no padding)
__device__ void base64url_encode_64bytes(const uint8_t *data64, char *out86) {
    uint32_t out_len;
    base64url_encode(data64, 64, out86, &out_len);
    // 64 bytes → 86 chars (64 = 21*3 + 1, so 21*4 + 2 = 86)
}

} // namespace encoding
