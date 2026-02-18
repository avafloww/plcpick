use std::collections::BTreeMap;

use base64::Engine as _;
use k256::ecdsa::{SigningKey, Signature};
use serde::Serialize;
use sha2::{Sha256, Digest};
use signature::Signer as _;

#[derive(Serialize, Clone)]
pub struct Service {
    #[serde(rename = "type")]
    pub service_type: String,
    pub endpoint: String,
}

#[derive(Serialize, Clone)]
pub struct PlcOperation {
    #[serde(rename = "type")]
    pub op_type: String,
    #[serde(rename = "rotationKeys")]
    pub rotation_keys: Vec<String>,
    #[serde(rename = "verificationMethods")]
    pub verification_methods: BTreeMap<String, String>,
    #[serde(rename = "alsoKnownAs")]
    pub also_known_as: Vec<String>,
    pub services: BTreeMap<String, Service>,
    pub prev: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sig: Option<String>,
}

pub fn encode_did_key(vk: &k256::ecdsa::VerifyingKey) -> String {
    let pt = vk.to_encoded_point(true);
    let mut buf = Vec::with_capacity(2 + pt.as_bytes().len());
    buf.extend_from_slice(&[0xe7, 0x01]); // multicodec secp256k1-pub varint
    buf.extend_from_slice(pt.as_bytes());
    format!("did:key:z{}", bs58::encode(buf).into_string())
}

pub fn build_signed_op(key: &SigningKey, handle: &str, pds: &str) -> PlcOperation {
    let dk = encode_did_key(key.verifying_key());

    let mut verification_methods = BTreeMap::new();
    verification_methods.insert("atproto".into(), dk.clone());

    let mut services = BTreeMap::new();
    services.insert(
        "atproto_pds".into(),
        Service {
            service_type: "AtprotoPersonalDataServer".into(),
            endpoint: pds.into(),
        },
    );

    let mut op = PlcOperation {
        op_type: "plc_operation".into(),
        rotation_keys: vec![dk],
        verification_methods,
        also_known_as: vec![format!("at://{handle}")],
        services,
        prev: None,
        sig: None,
    };

    let unsigned_cbor = serde_ipld_dagcbor::to_vec(&op).expect("cbor encode failed");
    let sig: Signature = key.sign(&unsigned_cbor);
    op.sig = Some(
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sig.to_bytes()),
    );

    op
}

pub fn did_suffix(op: &PlcOperation) -> String {
    let cbor = serde_ipld_dagcbor::to_vec(op).expect("cbor encode failed");
    let hash = Sha256::digest(&cbor);
    data_encoding::BASE32_NOPAD
        .encode(&hash[..15])
        .to_ascii_lowercase()
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
/// CBOR template with byte offsets for GPU patching.
/// The template bytes are fixed for a given handle/PDS — only the
/// base58-encoded pubkey and base64url signature vary.
pub struct CborTemplate {
    pub unsigned_bytes: Vec<u8>,
    /// Offsets of the 48-char base58 pubkey payload within unsigned_bytes
    /// (i.e., the bytes right after "did:key:z" in each location)
    pub unsigned_pubkey_offsets: [usize; 2],

    pub signed_bytes: Vec<u8>,
    /// Offsets of the 48-char base58 pubkey payload within signed_bytes
    pub signed_pubkey_offsets: [usize; 2],
    /// Offset of the 86-char base64url signature payload within signed_bytes
    pub signed_sig_offset: usize,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
/// Find all occurrences of `needle` in `haystack`, return their byte offsets.
fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut start = 0;
    while let Some(pos) = haystack[start..].windows(needle.len()).position(|w| w == needle) {
        offsets.push(start + pos);
        start += pos + 1;
    }
    offsets
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl CborTemplate {
    /// Build CBOR templates for the given handle/PDS.
    /// Uses a dummy key to generate the template, then locates the byte offsets
    /// of the variable parts (pubkey base58 payload and signature base64url payload).
    pub fn new(handle: &str, pds: &str) -> Self {
        let key = SigningKey::random(&mut rand::thread_rng());
        let op = build_signed_op(&key, handle, pds);

        // Get the base58 pubkey payload (48 chars after "did:key:z")
        let did_key_str = &op.rotation_keys[0]; // "did:key:zXXX...XXX"
        let base58_payload = &did_key_str[9..]; // skip "did:key:z"
        assert_eq!(base58_payload.len(), 48, "base58 payload must be 48 chars");

        // Get the signature payload (86 chars)
        let sig_payload = op.sig.as_ref().expect("signed op must have sig");
        assert_eq!(sig_payload.len(), 86, "base64url sig must be 86 chars");

        // Build unsigned CBOR (sig = None)
        let mut unsigned_op = op.clone();
        unsigned_op.sig = None;
        let unsigned_bytes = serde_ipld_dagcbor::to_vec(&unsigned_op).expect("cbor");

        // Build signed CBOR
        let signed_bytes = serde_ipld_dagcbor::to_vec(&op).expect("cbor");

        // Find base58 payload offsets in unsigned template
        let unsigned_pubkey_positions = find_all(&unsigned_bytes, base58_payload.as_bytes());
        assert_eq!(
            unsigned_pubkey_positions.len(), 2,
            "expected 2 pubkey locations in unsigned CBOR, found {}",
            unsigned_pubkey_positions.len()
        );

        // Find base58 payload offsets in signed template
        let signed_pubkey_positions = find_all(&signed_bytes, base58_payload.as_bytes());
        assert_eq!(
            signed_pubkey_positions.len(), 2,
            "expected 2 pubkey locations in signed CBOR, found {}",
            signed_pubkey_positions.len()
        );

        // Find signature payload offset in signed template
        let signed_sig_positions = find_all(&signed_bytes, sig_payload.as_bytes());
        assert_eq!(
            signed_sig_positions.len(), 1,
            "expected 1 sig location in signed CBOR, found {}",
            signed_sig_positions.len()
        );

        CborTemplate {
            unsigned_bytes,
            unsigned_pubkey_offsets: [unsigned_pubkey_positions[0], unsigned_pubkey_positions[1]],
            signed_bytes,
            signed_pubkey_offsets: [signed_pubkey_positions[0], signed_pubkey_positions[1]],
            signed_sig_offset: signed_sig_positions[0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn did_key_encoding_length() {
        // did:key:z + 48 base58 chars = 57 chars always
        let key = SigningKey::random(&mut rand::thread_rng());
        let dk = encode_did_key(key.verifying_key());
        assert!(dk.starts_with("did:key:z"));
        assert_eq!(dk.len(), 57, "did:key string should always be 57 chars");
    }

    #[test]
    fn did_key_encoding_consistent_length() {
        // verify across 100 random keys
        for _ in 0..100 {
            let key = SigningKey::random(&mut rand::thread_rng());
            let dk = encode_did_key(key.verifying_key());
            assert_eq!(dk.len(), 57);
        }
    }

    #[test]
    fn build_signed_op_produces_signature() {
        let key = SigningKey::random(&mut rand::thread_rng());
        let op = build_signed_op(&key, "test.bsky.social", "https://bsky.social");
        assert!(op.sig.is_some());
        // base64url_nopad of 64 bytes = 86 chars
        assert_eq!(op.sig.as_ref().unwrap().len(), 86);
    }

    #[test]
    fn did_suffix_is_24_base32_chars() {
        let key = SigningKey::random(&mut rand::thread_rng());
        let op = build_signed_op(&key, "test.bsky.social", "https://bsky.social");
        let suffix = did_suffix(&op);
        assert_eq!(suffix.len(), 24);
        // all chars should be base32 lowercase
        assert!(suffix.bytes().all(|b| matches!(b, b'a'..=b'z' | b'2'..=b'7')));
    }

    #[test]
    fn cbor_template_construction() {
        let tmpl = CborTemplate::new("test.bsky.social", "https://bsky.social");

        // Unsigned template should not contain sig field
        assert!(tmpl.unsigned_bytes.len() < tmpl.signed_bytes.len());

        // Offsets should be within bounds
        for &off in &tmpl.unsigned_pubkey_offsets {
            assert!(off + 48 <= tmpl.unsigned_bytes.len());
        }
        for &off in &tmpl.signed_pubkey_offsets {
            assert!(off + 48 <= tmpl.signed_bytes.len());
        }
        assert!(tmpl.signed_sig_offset + 86 <= tmpl.signed_bytes.len());

        // Two pubkey offsets should be different
        assert_ne!(tmpl.unsigned_pubkey_offsets[0], tmpl.unsigned_pubkey_offsets[1]);
        assert_ne!(tmpl.signed_pubkey_offsets[0], tmpl.signed_pubkey_offsets[1]);
    }

    #[test]
    fn cbor_template_patching_produces_valid_did() {
        let tmpl = CborTemplate::new("test.bsky.social", "https://bsky.social");

        // Generate a new key and manually patch the template
        let key = SigningKey::random(&mut rand::thread_rng());
        let op = build_signed_op(&key, "test.bsky.social", "https://bsky.social");
        let expected_suffix = did_suffix(&op);

        // Get the base58 and sig payloads from this key
        let did_key_str = &op.rotation_keys[0];
        let base58_payload = &did_key_str[9..];
        let sig_payload = op.sig.as_ref().unwrap();

        // Patch signed template
        let mut patched = tmpl.signed_bytes.clone();
        for &off in &tmpl.signed_pubkey_offsets {
            patched[off..off + 48].copy_from_slice(base58_payload.as_bytes());
        }
        patched[tmpl.signed_sig_offset..tmpl.signed_sig_offset + 86]
            .copy_from_slice(sig_payload.as_bytes());

        // Hash and check
        let hash = Sha256::digest(&patched);
        let suffix = data_encoding::BASE32_NOPAD
            .encode(&hash[..15])
            .to_ascii_lowercase();

        assert_eq!(suffix, expected_suffix, "patched template should produce same DID suffix");
    }

    #[test]
    fn cbor_template_length_stable() {
        // The core assumption: CBOR byte layout is fixed for a given handle/PDS
        let handle = "test.bsky.social";
        let pds = "https://bsky.social";

        let mut unsigned_len = None;
        let mut signed_len = None;

        for _ in 0..1000 {
            let key = SigningKey::random(&mut rand::thread_rng());
            let mut op = build_signed_op(&key, handle, pds);

            let signed_cbor = serde_ipld_dagcbor::to_vec(&op).expect("cbor");
            op.sig = None;
            let unsigned_cbor = serde_ipld_dagcbor::to_vec(&op).expect("cbor");

            match unsigned_len {
                None => unsigned_len = Some(unsigned_cbor.len()),
                Some(l) => assert_eq!(unsigned_cbor.len(), l, "unsigned CBOR length changed!"),
            }
            match signed_len {
                None => signed_len = Some(signed_cbor.len()),
                Some(l) => assert_eq!(signed_cbor.len(), l, "signed CBOR length changed!"),
            }
        }
    }
}
