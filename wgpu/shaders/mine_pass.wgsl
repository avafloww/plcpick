// Unified Mining Pass — single-dispatch pipeline (replaces ec_pass + hash_pass)
//
// Merged architecture matching CUDA kernel:
// - Full mining pipeline in one dispatch (no intermediate buffer)
// - Multiple iterations per thread per dispatch
// - Incremental point addition (pubkey += stride_G) after first iteration
// - GPU-side scalar advancement (no CPU roundtrip)
// - Persistent pubkey state between dispatches
//
// Composed with: field.wgsl, curve.wgsl, scalar.wgsl, sha256.wgsl,
//                hmac_drbg.wgsl, encoding.wgsl, pattern.wgsl

const MATCH_STRIDE: u32 = 32u; // u32s per match slot: privkey(8) + suffix(24)

struct MineParams {
    num_threads: u32,
    iterations_per_thread: u32,
    is_first_launch: u32,
    pattern_len: u32,
    unsigned_tmpl_offset: u32,
    unsigned_tmpl_len: u32,
    signed_tmpl_offset: u32,
    signed_tmpl_len: u32,
    pattern_offset: u32,
    max_matches: u32,
    unsigned_pk_off1: u32,
    unsigned_pk_off2: u32,
    signed_pk_off1: u32,
    signed_pk_off2: u32,
    signed_sig_offset: u32,
    _pad: u32,
}

// Group 0: EC state + mining data (8 storage + 1 uniform)
@group(0) @binding(0) var<storage, read_write> scalars: array<u32>;
@group(0) @binding(1) var<storage, read_write> pubkeys: array<u32>;      // 24 u32 per thread (JacobianPoint)
@group(0) @binding(2) var<storage, read> g_table_x: array<u32>;          // 15 * 8 = 120 u32
@group(0) @binding(3) var<storage, read> g_table_y: array<u32>;          // 15 * 8 = 120 u32
@group(0) @binding(4) var<storage, read> stride_g_xy: array<u32>;        // 16 u32: affine x[8] + y[8]
@group(0) @binding(5) var<storage, read> all_templates: array<u32>;      // packed: unsigned || signed || pattern
@group(0) @binding(6) var<storage, read_write> matches: array<u32>;      // match output slots
@group(0) @binding(7) var<storage, read_write> match_count: atomic<u32>;
@group(0) @binding(8) var<uniform> params: MineParams;

// 4-bit windowed scalar multiplication: scalar * G using precomputed table
fn scalar_mul_g_windowed(scalar: array<u32, 8>) -> JacobianPoint {
    var result = jac_infinity();
    for (var i = 63i; i >= 0; i--) {
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);
        result = jac_double(result);
        let limb_idx = u32(i) / 8u;
        let nibble_idx = u32(i) % 8u;
        let nibble = (scalar[limb_idx] >> (nibble_idx * 4u)) & 0xFu;
        if (nibble != 0u) {
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

// Compress Jacobian point to 33 bytes: prefix (0x02/0x03) + x coordinate
fn compress_pubkey(p: JacobianPoint, out: ptr<function, array<u32, 33>>) {
    let aff = jac_to_affine(p);
    (*out)[0] = select(0x02u, 0x03u, (aff.y[0] & 1u) != 0u);
    for (var i = 0u; i < 8u; i++) {
        let limb = aff.x[7u - i];
        let base = 1u + i * 4u;
        (*out)[base]      = (limb >> 24u) & 0xFFu;
        (*out)[base + 1u] = (limb >> 16u) & 0xFFu;
        (*out)[base + 2u] = (limb >> 8u) & 0xFFu;
        (*out)[base + 3u] = limb & 0xFFu;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.num_threads) { return; }

    // --- Load scalar (private key) ---
    var scalar: array<u32, 8>;
    let scalar_offset = tid * 8u;
    for (var i = 0u; i < 8u; i++) {
        scalar[i] = scalars[scalar_offset + i];
    }

    // --- Load or compute pubkey ---
    var pubkey: JacobianPoint;
    if (params.is_first_launch != 0u) {
        pubkey = scalar_mul_g_windowed(scalar);
    } else {
        let pk_offset = tid * 24u;
        for (var i = 0u; i < 8u; i++) {
            pubkey.x[i] = pubkeys[pk_offset + i];
            pubkey.y[i] = pubkeys[pk_offset + 8u + i];
            pubkey.z[i] = pubkeys[pk_offset + 16u + i];
        }
    }

    // --- Load stride_G affine coordinates (loop-invariant) ---
    var sg_x: array<u32, 8>;
    var sg_y: array<u32, 8>;
    for (var j = 0u; j < 8u; j++) {
        sg_x[j] = stride_g_xy[j];
        sg_y[j] = stride_g_xy[8u + j];
    }

    // --- Stride scalar = num_threads (for scalar advancement) ---
    var stride_scalar: array<u32, 8>;
    stride_scalar[0] = params.num_threads;
    // limbs [1..7] are zero-initialized

    // --- Load pattern (loop-invariant) ---
    var pat_local: array<u32, 24>;
    let pat_off = params.pattern_offset;
    let pat_len = params.pattern_len;
    for (var i = 0u; i < pat_len; i++) {
        pat_local[i] = all_templates[pat_off + i];
    }

    // Cache template offsets
    let u_tmpl_off = params.unsigned_tmpl_offset;
    let u_tmpl_len = params.unsigned_tmpl_len;
    let s_tmpl_off = params.signed_tmpl_offset;
    let s_tmpl_len = params.signed_tmpl_len;
    let u_pk_off1 = params.unsigned_pk_off1;
    let u_pk_off2 = params.unsigned_pk_off2;
    let s_pk_off1 = params.signed_pk_off1;
    let s_pk_off2 = params.signed_pk_off2;
    let s_sig_off = params.signed_sig_offset;

    // =====================================================================
    // Main iteration loop (matches CUDA's iterations_per_thread)
    // =====================================================================
    for (var iter = 0u; iter < params.iterations_per_thread; iter++) {

        // 1. Compress pubkey to 33 bytes
        var compressed: array<u32, 33>;
        compress_pubkey(pubkey, &compressed);

        // 2. Prepend multicodec prefix [0xe7, 0x01] + 33 pubkey bytes = 35 bytes
        var multicodec: array<u32, 35>;
        multicodec[0] = 0xe7u;
        multicodec[1] = 0x01u;
        for (var i = 0u; i < 33u; i++) {
            multicodec[i + 2u] = compressed[i];
        }

        // 3. Base58 encode -> 48 chars (did:key payload)
        var base58_pubkey: array<u32, 48>;
        enc_base58_35bytes(&multicodec, &base58_pubkey);

        // 4. Build unsigned CBOR template, patch pubkey, SHA256
        var data: array<u32, 512>;
        for (var i = 0u; i < u_tmpl_len; i++) {
            data[i] = all_templates[u_tmpl_off + i];
        }
        for (var i = 0u; i < 48u; i++) {
            data[u_pk_off1 + i] = base58_pubkey[i];
            data[u_pk_off2 + i] = base58_pubkey[i];
        }
        let msg_hash = sha256_hash(&data, u_tmpl_len);

        // 5. Convert hash to LE limbs for HMAC-DRBG
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

        // 6. RFC 6979 nonce
        let nonce_k = hmac_rfc6979_nonce(scalar, hash_scalar);

        // 7. R = nonce * G, r = R.x mod n
        let R = scalar_mul_g_windowed(nonce_k);
        let R_aff = jac_to_affine(R);
        var r_val = R_aff.x;
        if (scalar_cmp(r_val, SECP_N) >= 0) {
            r_val = scalar_sub_internal(r_val, SECP_N);
        }

        // 8. s = k_inv * (hash + r * privkey) mod n
        let r_times_priv = scalar_mod_n_mul(r_val, scalar);
        let hash_plus_rpriv = scalar_mod_n_add(hash_scalar, r_times_priv);
        let k_inv = scalar_mod_n_inv(nonce_k);
        var s_val = scalar_mod_n_mul(k_inv, hash_plus_rpriv);

        // 9. Low-s normalization (BIP-62)
        if (scalar_cmp(s_val, SECP_N_HALF) > 0) {
            s_val = scalar_mod_n_sub(SECP_N, s_val);
        }

        // 10. Encode signature (r || s) as 64 bytes, then base64url
        var sig_bytes: array<u32, 64>;
        var r_bytes: array<u32, 32>;
        var s_bytes: array<u32, 32>;
        hmac_store_scalar_bytes(r_val, &r_bytes);
        hmac_store_scalar_bytes(s_val, &s_bytes);
        for (var i = 0u; i < 32u; i++) {
            sig_bytes[i] = r_bytes[i];
            sig_bytes[32u + i] = s_bytes[i];
        }
        var base64_sig: array<u32, 86>;
        enc_base64url_64bytes(&sig_bytes, &base64_sig);

        // 11. Build signed CBOR template, patch pubkey + sig, SHA256
        for (var i = 0u; i < s_tmpl_len; i++) {
            data[i] = all_templates[s_tmpl_off + i];
        }
        for (var i = 0u; i < 48u; i++) {
            data[s_pk_off1 + i] = base58_pubkey[i];
            data[s_pk_off2 + i] = base58_pubkey[i];
        }
        for (var i = 0u; i < 86u; i++) {
            data[s_sig_off + i] = base64_sig[i];
        }
        let did_hash = sha256_hash(&data, s_tmpl_len);

        // 12. Base32 first 15 bytes -> 24-char DID suffix
        var hash_bytes_15: array<u32, 15>;
        for (var i = 0u; i < 4u; i++) {
            let w = did_hash[i];
            let base = i * 4u;
            if (base < 15u) { hash_bytes_15[base] = (w >> 24u) & 0xFFu; }
            if (base + 1u < 15u) { hash_bytes_15[base + 1u] = (w >> 16u) & 0xFFu; }
            if (base + 2u < 15u) { hash_bytes_15[base + 2u] = (w >> 8u) & 0xFFu; }
            if (base + 3u < 15u) { hash_bytes_15[base + 3u] = w & 0xFFu; }
        }
        var suffix: array<u32, 24>;
        enc_base32_15bytes(&hash_bytes_15, &suffix);

        // 13. Pattern match
        if (pattern_glob_match(&pat_local, pat_len, &suffix, 24u)) {
            let slot = atomicAdd(&match_count, 1u);
            if (slot < params.max_matches) {
                let match_offset = slot * MATCH_STRIDE;
                for (var i = 0u; i < 8u; i++) {
                    matches[match_offset + i] = scalar[i];
                }
                for (var i = 0u; i < 24u; i++) {
                    matches[match_offset + 8u + i] = suffix[i];
                }
            }
        }

        // 14. Advance scalar + pubkey for next iteration
        scalar = scalar_mod_n_add(scalar, stride_scalar);
        pubkey = jac_add_affine(pubkey, sg_x, sg_y);
    }

    // --- Save state for next dispatch ---
    let so = tid * 8u;
    for (var i = 0u; i < 8u; i++) {
        scalars[so + i] = scalar[i];
    }
    let po = tid * 24u;
    for (var i = 0u; i < 8u; i++) {
        pubkeys[po + i] = pubkey.x[i];
        pubkeys[po + 8u + i] = pubkey.y[i];
        pubkeys[po + 16u + i] = pubkey.z[i];
    }
}
