// EC Pass — Pass 1 of two-pass mining pipeline
// Heavy compute: scalar_mul_G, ECDSA signing, pubkey/sig encoding
//
// Composed with: field.wgsl, curve.wgsl, scalar.wgsl, sha256.wgsl, hmac_drbg.wgsl, encoding.wgsl
//
// Per-thread output (written to results buffer):
//   [0..8)    privkey — 8 u32 LE limbs (32 bytes)
//   [8..56)   base58_pubkey — 48 u32 chars (one ASCII char per u32)
//   [56..142) base64url_sig — 86 u32 chars
//   Total: 142 u32s per thread

const EC_RESULT_STRIDE: u32 = 142u;

struct EcPassParams {
    is_first_launch: u32,
    num_threads: u32,
    unsigned_template_byte_len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> scalars: array<u32>;
@group(0) @binding(1) var<storage, read> g_table_x: array<u32>;
@group(0) @binding(2) var<storage, read> g_table_y: array<u32>;
@group(0) @binding(3) var<storage, read> stride_g_xy: array<u32>;       // 16 u32s: x[8] + y[8]
@group(0) @binding(4) var<storage, read> unsigned_template: array<u32>; // CBOR bytes (one per u32)
@group(0) @binding(5) var<storage, read> unsigned_pubkey_offsets: array<u32>; // [offset1, offset2, len1, len2]
@group(0) @binding(6) var<storage, read_write> results: array<u32>;
@group(0) @binding(7) var<uniform> params: EcPassParams;

// 4-bit windowed scalar multiplication: scalar * G using precomputed table
// Accesses g_table_x/g_table_y bindings directly (WGSL can't pass storage ptrs)
fn scalar_mul_g_windowed(scalar: array<u32, 8>) -> JacobianPoint {
    var result = jac_infinity();

    // Process 64 nibbles (256 bits / 4 bits per nibble), high to low
    for (var i = 63i; i >= 0; i--) {
        // Double 4 times
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);

        // Extract 4-bit nibble from scalar
        let limb_idx = u32(i) / 8u;
        let nibble_idx = u32(i) % 8u;
        let nibble = (scalar[limb_idx] >> (nibble_idx * 4u)) & 0xFu;

        if (nibble != 0u) {
            // Look up g_table[nibble - 1]
            let table_offset = (nibble - 1u) * 8u;
            var qx: array<u32, 8>;
            var qy: array<u32, 8>;
            for (var j = 0u; j < 8u; j++) {
                qx[j] = g_table_x[table_offset + j];
                qy[j] = g_table_y[table_offset + j];
            }
            result = jac_add_affine(result, qx, qy);
        }
    }
    return result;
}

// secp256k1 N/2 for low-s normalization (BIP-62)
const SECP_N_HALF = array<u32, 8>(
    0x681B20A0u, 0xDFE92F46u, 0x57A4501Du, 0x5D576E73u,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu
);

// Compress a Jacobian point to 33 bytes: prefix (0x02/0x03) + x coordinate
fn compress_pubkey(p: JacobianPoint, out: ptr<function, array<u32, 33>>) {
    let aff = jac_to_affine(p);
    // Prefix: 0x02 if y is even, 0x03 if odd
    (*out)[0] = select(0x02u, 0x03u, (aff.y[0] & 1u) != 0u);
    // X coordinate in big-endian bytes
    for (var i = 0u; i < 8u; i++) {
        let limb = aff.x[7u - i];
        let base = 1u + i * 4u;
        (*out)[base]      = (limb >> 24u) & 0xFFu;
        (*out)[base + 1u] = (limb >> 16u) & 0xFFu;
        (*out)[base + 2u] = (limb >> 8u) & 0xFFu;
        (*out)[base + 3u] = limb & 0xFFu;
    }
}

// Hash the unsigned CBOR template with base58 pubkey patched at known offsets
fn hash_unsigned_template_with_pubkey(
    base58_pk: ptr<function, array<u32, 48>>,
) -> array<u32, 8> {
    let tmpl_len = params.unsigned_template_byte_len;
    let off1 = unsigned_pubkey_offsets[0];
    let off2 = unsigned_pubkey_offsets[1];
    let len1 = unsigned_pubkey_offsets[2]; // should be 48
    let len2 = unsigned_pubkey_offsets[3]; // should be 48

    // Copy template into local buffer, patch pubkey at offsets
    var data: array<u32, 512>;
    for (var i = 0u; i < tmpl_len; i++) {
        data[i] = unsigned_template[i];
    }
    for (var i = 0u; i < len1; i++) {
        data[off1 + i] = (*base58_pk)[i];
    }
    for (var i = 0u; i < len2; i++) {
        data[off2 + i] = (*base58_pk)[i];
    }

    return sha256_hash(&data, tmpl_len);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.num_threads) { return; }

    // 1. Load scalar (private key)
    var scalar: array<u32, 8>;
    let scalar_offset = tid * 8u;
    for (var i = 0u; i < 8u; i++) {
        scalar[i] = scalars[scalar_offset + i];
    }

    // 2. Compute public key
    var pubkey: JacobianPoint;
    if (params.is_first_launch != 0u) {
        pubkey = scalar_mul_g_windowed(scalar);
    } else {
        // Incremental: pubkey += stride_G (load previous pubkey from results as affine, add stride_G)
        // For simplicity, recompute from scalar each time in initial implementation
        // TODO: optimize with incremental point addition
        pubkey = scalar_mul_g_windowed(scalar);
    }

    // 3. Compress pubkey to 33 bytes
    var compressed: array<u32, 33>;
    compress_pubkey(pubkey, &compressed);

    // 4. Prepend multicodec prefix [0xe7, 0x01] + 33 pubkey bytes = 35 bytes
    var multicodec: array<u32, 35>;
    multicodec[0] = 0xe7u;
    multicodec[1] = 0x01u;
    for (var i = 0u; i < 33u; i++) {
        multicodec[i + 2u] = compressed[i];
    }

    // 5. Base58 encode → 48 chars (did:key payload)
    var base58_pubkey: array<u32, 48>;
    enc_base58_35bytes(&multicodec, &base58_pubkey);

    // 6. SHA256 hash unsigned CBOR template (with pubkey patched in)
    let msg_hash = hash_unsigned_template_with_pubkey(&base58_pubkey);

    // 7. Convert msg_hash (BE words) to LE limbs for HMAC-DRBG
    // msg_hash is already in SHA256 state format (BE words)
    // We need it as BE bytes → LE limbs for the scalar
    var hash_bytes: array<u32, 32>;
    for (var i = 0u; i < 8u; i++) {
        let w = msg_hash[i];
        let base = i * 4u;
        hash_bytes[base]      = (w >> 24u) & 0xFFu;
        hash_bytes[base + 1u] = (w >> 16u) & 0xFFu;
        hash_bytes[base + 2u] = (w >> 8u) & 0xFFu;
        hash_bytes[base + 3u] = w & 0xFFu;
    }
    let hash_scalar = hmac_load_scalar_bytes(&hash_bytes);

    // 8. Convert privkey to bytes for HMAC-DRBG
    var privkey_bytes: array<u32, 32>;
    hmac_store_scalar_bytes(scalar, &privkey_bytes);

    // 9. RFC 6979 nonce
    let nonce_k = hmac_rfc6979_nonce(scalar, hash_scalar);

    // 10. R = nonce * G
    let R = scalar_mul_g_windowed(nonce_k);

    // 11. r = R.x mod n
    let R_aff = jac_to_affine(R);
    var r_val = R_aff.x;
    if (scalar_cmp(r_val, SECP_N) >= 0) {
        r_val = scalar_sub_internal(r_val, SECP_N);
    }

    // 12. s = k_inv * (hash + r * privkey) mod n
    let r_times_priv = scalar_mod_n_mul(r_val, scalar);
    let hash_plus_rpriv = scalar_mod_n_add(hash_scalar, r_times_priv);
    let k_inv = scalar_mod_n_inv(nonce_k);
    var s_val = scalar_mod_n_mul(k_inv, hash_plus_rpriv);

    // 13. Low-s normalization (BIP-62)
    if (scalar_cmp(s_val, SECP_N_HALF) > 0) {
        s_val = scalar_mod_n_sub(SECP_N, s_val);
    }

    // 14. Encode signature as 64 bytes (r || s, big-endian)
    var sig_bytes: array<u32, 64>;
    var r_bytes: array<u32, 32>;
    var s_bytes: array<u32, 32>;
    hmac_store_scalar_bytes(r_val, &r_bytes);
    hmac_store_scalar_bytes(s_val, &s_bytes);
    for (var i = 0u; i < 32u; i++) {
        sig_bytes[i] = r_bytes[i];
        sig_bytes[32u + i] = s_bytes[i];
    }

    // 15. Base64url encode signature → 86 chars
    var base64_sig: array<u32, 86>;
    enc_base64url_64bytes(&sig_bytes, &base64_sig);

    // 16. Write results: privkey (8 limbs) + base58_pubkey (48) + base64url_sig (86) = 142 u32s
    let out_offset = tid * EC_RESULT_STRIDE;
    for (var i = 0u; i < 8u; i++) {
        results[out_offset + i] = scalar[i];
    }
    for (var i = 0u; i < 48u; i++) {
        results[out_offset + 8u + i] = base58_pubkey[i];
    }
    for (var i = 0u; i < 86u; i++) {
        results[out_offset + 56u + i] = base64_sig[i];
    }

    // 17. Advance scalar for next launch: scalar += stride
    // (stride = total_threads, precomputed on CPU)
    let stride_scalar = scalars[params.num_threads * 8u]; // stride stored after all thread scalars
    // Actually, advance is done CPU-side for simplicity in first implementation
}
