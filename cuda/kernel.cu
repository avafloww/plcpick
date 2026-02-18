#include <cstdint>
#include "secp256k1.cuh"
#include "sha256.cuh"
#include "hmac_drbg.cuh"
#include "encoding.cuh"
#include "pattern.cuh"

// Match result written by GPU threads
struct GpuMatch {
    uint8_t privkey[32];   // private key scalar (big-endian)
    uint8_t signature[64]; // ECDSA signature (r || s, big-endian)
    char suffix[24];       // DID suffix (base32)
    uint32_t found;        // 1 if this slot has a match
};

// Kernel parameters passed from host
struct KernelParams {
    // CBOR templates
    const uint8_t *unsigned_template;
    uint32_t unsigned_template_len;
    const uint8_t *signed_template;
    uint32_t signed_template_len;

    // Byte offsets for patching (within templates)
    uint32_t unsigned_pubkey_offsets[2]; // 2 locations in unsigned template
    uint32_t signed_pubkey_offsets[2];   // 2 locations in signed template
    uint32_t signed_sig_offset;          // 1 location in signed template

    // Pattern for matching
    const char *pattern;
    uint32_t pattern_len;

    // Per-thread state
    uint8_t *scalars;          // N * 32 bytes: current scalar per thread (big-endian)
    secp256k1::JacobianPoint *pubkeys; // N points: current pubkey per thread

    // Stride for incremental keys
    secp256k1::U256 stride;           // total_threads as U256
    secp256k1::JacobianPoint stride_G; // stride * G, precomputed

    // Output
    GpuMatch *matches;
    uint32_t *match_count;     // atomic counter
    uint32_t max_matches;

    // Control
    uint32_t iterations_per_thread;
    uint32_t is_first_launch;  // 1 if this is the first kernel launch (need full scalar mul)
};

extern "C" __global__ void mine_kernel(KernelParams params) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load current scalar
    secp256k1::U256 scalar;
    secp256k1::load_scalar(scalar, params.scalars + tid * 32);

    // Load or compute current pubkey
    secp256k1::JacobianPoint pubkey;
    if (params.is_first_launch) {
        // First launch: compute pubkey from scalar via full scalar multiplication
        secp256k1::scalar_mul_G(pubkey, scalar);
    } else {
        pubkey = params.pubkeys[tid];
    }

    // Local working buffers
    uint8_t unsigned_buf[512]; // enough for CBOR template
    uint8_t signed_buf[512];
    uint8_t compressed_pubkey[33];
    uint8_t multicodec_buf[35]; // 2-byte prefix + 33-byte pubkey
    char base58_pubkey[48];
    uint8_t msg_hash[32];
    uint8_t did_hash[32];
    char suffix[24];

    for (uint32_t iter = 0; iter < params.iterations_per_thread; iter++) {
        // 1. Get compressed public key
        secp256k1::get_compressed_pubkey(compressed_pubkey, pubkey);

        // 2. Prepend multicodec prefix
        multicodec_buf[0] = 0xe7;
        multicodec_buf[1] = 0x01;
        for (int i = 0; i < 33; i++) multicodec_buf[i + 2] = compressed_pubkey[i];

        // 3. Base58 encode
        encoding::base58_encode_35bytes(multicodec_buf, base58_pubkey);

        // 4. Build unsigned CBOR: copy template and patch pubkey at 2 offsets
        for (uint32_t i = 0; i < params.unsigned_template_len; i++) {
            unsigned_buf[i] = params.unsigned_template[i];
        }
        // did:key:z prefix is 9 bytes, then 48 base58 chars
        // The offsets point to the start of the 48-char base58 payload
        for (int loc = 0; loc < 2; loc++) {
            uint32_t off = params.unsigned_pubkey_offsets[loc];
            for (int j = 0; j < 48; j++) {
                unsigned_buf[off + j] = (uint8_t)base58_pubkey[j];
            }
        }

        // 5. SHA256 the unsigned CBOR → message hash
        sha256::hash(unsigned_buf, params.unsigned_template_len, msg_hash);

        // 6. ECDSA sign: RFC 6979 nonce, then compute (r, s)
        uint8_t privkey_bytes[32];
        secp256k1::store_scalar(privkey_bytes, scalar);

        secp256k1::U256 nonce_k;
        hmac::rfc6979_nonce(nonce_k, privkey_bytes, msg_hash);

        // R = k * G
        secp256k1::JacobianPoint R;
        secp256k1::scalar_mul_G(R, nonce_k);

        // Convert R to affine to get r = R.x mod n
        secp256k1::U256 rx, ry;
        secp256k1::jacobian_to_affine(rx, ry, R);

        // r = rx mod n (rx is already < p ≈ n, but could be >= n)
        secp256k1::U256 r_val = rx;
        if (secp256k1::cmp(r_val, secp256k1::N) >= 0) {
            secp256k1::sub256(r_val, r_val, secp256k1::N);
        }

        // s = k^(-1) * (hash + r * privkey) mod n
        secp256k1::U256 hash_scalar;
        secp256k1::load_scalar(hash_scalar, msg_hash);

        secp256k1::U256 r_times_priv;
        secp256k1::scalar_mul(r_times_priv, r_val, scalar);

        secp256k1::U256 hash_plus_rpriv;
        secp256k1::scalar_add(hash_plus_rpriv, hash_scalar, r_times_priv);

        secp256k1::U256 k_inv;
        secp256k1::scalar_inv(k_inv, nonce_k);

        secp256k1::U256 s_val;
        secp256k1::scalar_mul(s_val, k_inv, hash_plus_rpriv);

        // Low-s normalization (BIP-62)
        if (secp256k1::cmp(s_val, secp256k1::N_HALF) > 0) {
            secp256k1::sub256(s_val, secp256k1::N, s_val);
        }

        // 7. Encode signature as 64 bytes (r || s, big-endian)
        uint8_t sig_bytes[64];
        secp256k1::store_scalar(sig_bytes, r_val);
        secp256k1::store_scalar(sig_bytes + 32, s_val);

        // 8. Base64url encode signature
        char base64_sig[86];
        encoding::base64url_encode_64bytes(sig_bytes, base64_sig);

        // 9. Build signed CBOR: copy template and patch pubkey + signature
        for (uint32_t i = 0; i < params.signed_template_len; i++) {
            signed_buf[i] = params.signed_template[i];
        }
        for (int loc = 0; loc < 2; loc++) {
            uint32_t off = params.signed_pubkey_offsets[loc];
            for (int j = 0; j < 48; j++) {
                signed_buf[off + j] = (uint8_t)base58_pubkey[j];
            }
        }
        {
            uint32_t off = params.signed_sig_offset;
            for (int j = 0; j < 86; j++) {
                signed_buf[off + j] = (uint8_t)base64_sig[j];
            }
        }

        // 10. SHA256 the signed CBOR → DID hash
        sha256::hash(signed_buf, params.signed_template_len, did_hash);

        // 11. Base32 encode first 15 bytes → 24-char suffix
        encoding::base32_encode_15bytes(did_hash, suffix);

        // 12. Pattern match
        if (pattern::glob_match(params.pattern, params.pattern_len, suffix, 24)) {
            // Found a match!
            uint32_t slot = atomicAdd(params.match_count, 1);
            if (slot < params.max_matches) {
                GpuMatch &m = params.matches[slot];
                for (int j = 0; j < 32; j++) m.privkey[j] = privkey_bytes[j];
                for (int j = 0; j < 64; j++) m.signature[j] = sig_bytes[j];
                for (int j = 0; j < 24; j++) m.suffix[j] = suffix[j];
                m.found = 1;
            }
        }

        // 13. Increment scalar and update pubkey for next iteration
        secp256k1::scalar_add(scalar, scalar, params.stride);
        secp256k1::point_add(pubkey, pubkey, params.stride_G);
    }

    // Save state for next kernel launch
    secp256k1::store_scalar(params.scalars + tid * 32, scalar);
    params.pubkeys[tid] = pubkey;
}

// Init kernel: populate the G_TABLE in constant memory with [1*G, 2*G, ..., 15*G]
// Must be called once before any kernel that uses scalar_mul_G.
extern "C" __global__ void init_g_table() {
    // G_TABLE[0] = 1*G (affine)
    secp256k1::G_TABLE[0].x = secp256k1::GX;
    secp256k1::G_TABLE[0].y = secp256k1::GY;

    // Compute (i+1)*G by repeated addition
    secp256k1::JacobianPoint acc;
    acc.x = secp256k1::GX;
    acc.y = secp256k1::GY;
    acc.z = {};
    acc.z.d[0] = 1;

    for (int i = 1; i < 15; i++) {
        secp256k1::point_add_affine(acc, acc, secp256k1::GX, secp256k1::GY);
        secp256k1::U256 ax, ay;
        secp256k1::jacobian_to_affine(ax, ay, acc);
        secp256k1::G_TABLE[i].x = ax;
        secp256k1::G_TABLE[i].y = ay;
    }
}

// Init kernel: compute stride * G for incremental key optimization
// Runs with a single thread. Writes stride as U256 and stride_G as JacobianPoint.
extern "C" __global__ void compute_stride_g(uint32_t stride_val,
                                             uint32_t *stride_out,      // 8 u32s = U256
                                             uint32_t *stride_g_out) {  // 24 u32s = JacobianPoint
    secp256k1::U256 s = {};
    s.d[0] = stride_val;

    // Write stride U256
    for (int i = 0; i < 8; i++) {
        stride_out[i] = s.d[i];
    }

    // Compute stride * G
    secp256k1::JacobianPoint result;
    secp256k1::scalar_mul_G(result, s);

    // Write JacobianPoint (x, y, z as consecutive U256s)
    for (int i = 0; i < 8; i++) {
        stride_g_out[i]      = result.x.d[i];
        stride_g_out[8 + i]  = result.y.d[i];
        stride_g_out[16 + i] = result.z.d[i];
    }
}

// Test kernel: run SHA256 on input and return hash
extern "C" __global__ void test_sha256(const uint8_t *input, uint32_t input_len, uint8_t *output) {
    sha256::hash(input, input_len, output);
}

// Test kernel: scalar multiplication k * G, return compressed pubkey
extern "C" __global__ void test_scalar_mul_G(const uint8_t *scalar32, uint8_t *pubkey33) {
    secp256k1::U256 k;
    secp256k1::load_scalar(k, scalar32);
    secp256k1::JacobianPoint P;
    secp256k1::scalar_mul_G(P, k);
    secp256k1::get_compressed_pubkey(pubkey33, P);
}

// Test kernel: full ECDSA sign and return signature
extern "C" __global__ void test_ecdsa_sign(const uint8_t *privkey32,
                                            const uint8_t *msg_hash32,
                                            uint8_t *sig_out64) {
    secp256k1::U256 scalar;
    secp256k1::load_scalar(scalar, privkey32);

    secp256k1::U256 nonce_k;
    hmac::rfc6979_nonce(nonce_k, privkey32, msg_hash32);

    secp256k1::JacobianPoint R;
    secp256k1::scalar_mul_G(R, nonce_k);

    secp256k1::U256 rx, ry;
    secp256k1::jacobian_to_affine(rx, ry, R);

    secp256k1::U256 r_val = rx;
    if (secp256k1::cmp(r_val, secp256k1::N) >= 0) {
        secp256k1::sub256(r_val, r_val, secp256k1::N);
    }

    secp256k1::U256 hash_scalar;
    secp256k1::load_scalar(hash_scalar, msg_hash32);

    secp256k1::U256 r_times_priv;
    secp256k1::scalar_mul(r_times_priv, r_val, scalar);

    secp256k1::U256 hash_plus_rpriv;
    secp256k1::scalar_add(hash_plus_rpriv, hash_scalar, r_times_priv);

    secp256k1::U256 k_inv;
    secp256k1::scalar_inv(k_inv, nonce_k);

    secp256k1::U256 s_val;
    secp256k1::scalar_mul(s_val, k_inv, hash_plus_rpriv);

    if (secp256k1::cmp(s_val, secp256k1::N_HALF) > 0) {
        secp256k1::sub256(s_val, secp256k1::N, s_val);
    }

    secp256k1::store_scalar(sig_out64, r_val);
    secp256k1::store_scalar(sig_out64 + 32, s_val);
}

// Test kernel: scalar_mul(a, b) mod n → result (all as 8 x uint32 LE limbs)
extern "C" __global__ void test_scalar_mul(const uint32_t *a_limbs, const uint32_t *b_limbs, uint32_t *r_limbs) {
    secp256k1::U256 a, b, r;
    for (int i = 0; i < 8; i++) { a.d[i] = a_limbs[i]; b.d[i] = b_limbs[i]; }
    secp256k1::scalar_mul(r, a, b);
    for (int i = 0; i < 8; i++) r_limbs[i] = r.d[i];
}

// Test kernel: scalar_inv(a) mod n → result (all as 8 x uint32 LE limbs)
extern "C" __global__ void test_scalar_inv(const uint32_t *a_limbs, uint32_t *r_limbs) {
    secp256k1::U256 a, r;
    for (int i = 0; i < 8; i++) a.d[i] = a_limbs[i];
    secp256k1::scalar_inv(r, a);
    for (int i = 0; i < 8; i++) r_limbs[i] = r.d[i];
}

// Test kernel: field_mul(a, b) → result (all as 8 x uint32 LE limbs)
extern "C" __global__ void test_field_mul(const uint32_t *a_limbs, const uint32_t *b_limbs, uint32_t *r_limbs) {
    secp256k1::U256 a, b, r;
    for (int i = 0; i < 8; i++) { a.d[i] = a_limbs[i]; b.d[i] = b_limbs[i]; }
    secp256k1::field_mul(r, a, b);
    for (int i = 0; i < 8; i++) r_limbs[i] = r.d[i];
}

// Test kernel: point_double(G) → compressed pubkey (33 bytes)
extern "C" __global__ void test_point_double_g(uint8_t *pubkey33) {
    secp256k1::JacobianPoint p;
    p.x = secp256k1::GX;
    p.y = secp256k1::GY;
    p.z = {};
    p.z.d[0] = 1;

    secp256k1::JacobianPoint r;
    secp256k1::point_double(r, p);
    secp256k1::get_compressed_pubkey(pubkey33, r);
}

// Test kernel: base58 encode 35 bytes
extern "C" __global__ void test_base58(const uint8_t *input35, char *output48) {
    encoding::base58_encode_35bytes(input35, output48);
}

// Test kernel: base64url encode 64 bytes
extern "C" __global__ void test_base64url(const uint8_t *input64, char *output86) {
    encoding::base64url_encode_64bytes(input64, output86);
}

// Test kernel: base32 encode 15 bytes
extern "C" __global__ void test_base32(const uint8_t *input15, char *output24) {
    encoding::base32_encode_15bytes(input15, output24);
}
