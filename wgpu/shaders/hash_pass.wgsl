// Hash Pass — Pass 2 of two-pass mining pipeline
// Lightweight: CBOR template patching, SHA256, base32, pattern match
//
// Composed with: sha256.wgsl, encoding.wgsl, pattern.wgsl
//
// Reads ec_pass output (per-thread):
//   [0..8)    privkey — 8 u32 LE limbs
//   [8..56)   base58_pubkey — 48 u32 chars
//   [56..142) base64url_sig — 86 u32 chars

const EC_RESULT_STRIDE: u32 = 142u;

struct HashPassParams {
    num_threads: u32,
    pattern_len: u32,
    signed_template_byte_len: u32,
    max_matches: u32,
    pubkey_offset_1: u32,
    pubkey_offset_2: u32,
    sig_offset: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> ec_results: array<u32>;
@group(0) @binding(1) var<storage, read> signed_template: array<u32>;   // CBOR bytes (one per u32)
@group(0) @binding(2) var<storage, read> pattern_buf: array<u32>;       // pattern chars (one per u32)
@group(0) @binding(3) var<storage, read_write> matches: array<u32>;     // match output
@group(0) @binding(4) var<storage, read_write> match_count: atomic<u32>;
@group(0) @binding(5) var<uniform> params: HashPassParams;

// Match output layout per match:
//   [0..8)   privkey — 8 u32 LE limbs
//   [8..32)  suffix — 24 u32 chars
//   Total: 32 u32s per match
const MATCH_STRIDE: u32 = 32u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.num_threads) { return; }

    let in_offset = tid * EC_RESULT_STRIDE;

    // Read base58 pubkey (48 chars) and base64url sig (86 chars) from ec_pass
    var base58_pk: array<u32, 48>;
    for (var i = 0u; i < 48u; i++) {
        base58_pk[i] = ec_results[in_offset + 8u + i];
    }
    var base64_sig: array<u32, 86>;
    for (var i = 0u; i < 86u; i++) {
        base64_sig[i] = ec_results[in_offset + 56u + i];
    }

    // Copy signed CBOR template and patch with pubkey + signature
    let tmpl_len = params.signed_template_byte_len;
    var data: array<u32, 512>;
    for (var i = 0u; i < tmpl_len; i++) {
        data[i] = signed_template[i];
    }

    // Patch base58 pubkey at two offsets (did:key encoding, 48 chars each)
    let off1 = params.pubkey_offset_1;
    let off2 = params.pubkey_offset_2;
    for (var i = 0u; i < 48u; i++) {
        data[off1 + i] = base58_pk[i];
        data[off2 + i] = base58_pk[i];
    }

    // Patch base64url signature at one offset (86 chars)
    let sig_off = params.sig_offset;
    for (var i = 0u; i < 86u; i++) {
        data[sig_off + i] = base64_sig[i];
    }

    // SHA256 hash the patched signed CBOR template
    let did_hash = sha256_hash(&data, tmpl_len);

    // Extract first 15 bytes from hash (hash is 8 BE words)
    var hash_bytes_15: array<u32, 15>;
    for (var i = 0u; i < 4u; i++) {
        let w = did_hash[i];
        let base = i * 4u;
        if (base < 15u) { hash_bytes_15[base] = (w >> 24u) & 0xFFu; }
        if (base + 1u < 15u) { hash_bytes_15[base + 1u] = (w >> 16u) & 0xFFu; }
        if (base + 2u < 15u) { hash_bytes_15[base + 2u] = (w >> 8u) & 0xFFu; }
        if (base + 3u < 15u) { hash_bytes_15[base + 3u] = w & 0xFFu; }
    }

    // Base32 encode 15 bytes → 24-char DID suffix
    var suffix: array<u32, 24>;
    enc_base32_15bytes(&hash_bytes_15, &suffix);

    // Pattern match
    if (pattern_glob_match(&pattern_buf, params.pattern_len, &suffix, 24u)) {
        let slot = atomicAdd(&match_count, 1u);
        if (slot < params.max_matches) {
            let match_offset = slot * MATCH_STRIDE;
            // Write privkey (8 u32 limbs)
            for (var i = 0u; i < 8u; i++) {
                matches[match_offset + i] = ec_results[in_offset + i];
            }
            // Write DID suffix (24 chars)
            for (var i = 0u; i < 24u; i++) {
                matches[match_offset + 8u + i] = suffix[i];
            }
        }
    }
}
