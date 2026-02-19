// =============================================================================
// secp256k1 Scalar Arithmetic (mod n) — 256-bit using u32 limbs
// =============================================================================
// Group order: n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Representation: 8 x u32 limbs, little-endian
//
// Unlike field.wgsl (mod p where 2^256 - p fits in 33 bits), the n-complement
// 2^256 - n = 0x14551231950B75FC4402DA1732FC9BEBF is 129 bits (5 limbs).
// This requires a different reduction strategy for multiplication.

// Group order n in limbs (little-endian)
const N0: u32 = 0xD0364141u;
const N1: u32 = 0xBFD25E8Cu;
const N2: u32 = 0xAF48A03Bu;
const N3: u32 = 0xBAAEDCE6u;
const N4: u32 = 0xFFFFFFFEu;
const N5: u32 = 0xFFFFFFFFu;
const N6: u32 = 0xFFFFFFFFu;
const N7: u32 = 0xFFFFFFFFu;

const SECP_N = array<u32, 8>(N0, N1, N2, N3, N4, N5, N6, N7);

// 2^256 mod n = 2^256 - n (little-endian limbs), 129 bits
const NC0: u32 = 0x2FC9BEBFu;
const NC1: u32 = 0x402DA173u;
const NC2: u32 = 0x50B75FC4u;
const NC3: u32 = 0x45512319u;
const NC4: u32 = 0x00000001u;

// -----------------------------------------------------------------------------
// Check if scalar is zero
// -----------------------------------------------------------------------------
fn scalar_is_zero(a: array<u32, 8>) -> bool {
    return (a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7]) == 0u;
}

// -----------------------------------------------------------------------------
// Compare two 256-bit values: returns -1 if a < b, 0 if equal, 1 if a > b
// Compares from high limb to low
// -----------------------------------------------------------------------------
fn scalar_cmp(a: array<u32, 8>, b: array<u32, 8>) -> i32 {
    if (a[7] > b[7]) { return 1; } if (a[7] < b[7]) { return -1; }
    if (a[6] > b[6]) { return 1; } if (a[6] < b[6]) { return -1; }
    if (a[5] > b[5]) { return 1; } if (a[5] < b[5]) { return -1; }
    if (a[4] > b[4]) { return 1; } if (a[4] < b[4]) { return -1; }
    if (a[3] > b[3]) { return 1; } if (a[3] < b[3]) { return -1; }
    if (a[2] > b[2]) { return 1; } if (a[2] < b[2]) { return -1; }
    if (a[1] > b[1]) { return 1; } if (a[1] < b[1]) { return -1; }
    if (a[0] > b[0]) { return 1; } if (a[0] < b[0]) { return -1; }
    return 0;
}

// -----------------------------------------------------------------------------
// Subtract: result = a - b (assumes a >= b, no underflow handling)
// Used internally for conditional reduction
// -----------------------------------------------------------------------------
fn scalar_sub_internal(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var c: array<u32, 8>;
    var borrow: u32 = 0u;
    var diff: u32;

    diff = a[0] - b[0]; borrow = select(0u, 1u, a[0] < b[0]); c[0] = diff;
    diff = a[1] - b[1] - borrow; borrow = select(0u, 1u, a[1] < b[1] + borrow || (borrow == 1u && b[1] == 0xFFFFFFFFu)); c[1] = diff;
    diff = a[2] - b[2] - borrow; borrow = select(0u, 1u, a[2] < b[2] + borrow || (borrow == 1u && b[2] == 0xFFFFFFFFu)); c[2] = diff;
    diff = a[3] - b[3] - borrow; borrow = select(0u, 1u, a[3] < b[3] + borrow || (borrow == 1u && b[3] == 0xFFFFFFFFu)); c[3] = diff;
    diff = a[4] - b[4] - borrow; borrow = select(0u, 1u, a[4] < b[4] + borrow || (borrow == 1u && b[4] == 0xFFFFFFFFu)); c[4] = diff;
    diff = a[5] - b[5] - borrow; borrow = select(0u, 1u, a[5] < b[5] + borrow || (borrow == 1u && b[5] == 0xFFFFFFFFu)); c[5] = diff;
    diff = a[6] - b[6] - borrow; borrow = select(0u, 1u, a[6] < b[6] + borrow || (borrow == 1u && b[6] == 0xFFFFFFFFu)); c[6] = diff;
    diff = a[7] - b[7] - borrow; c[7] = diff;

    return c;
}

// -----------------------------------------------------------------------------
// Reduce: if a >= n, return a - n, else return a
// Only handles single reduction (assumes a < 2n)
// -----------------------------------------------------------------------------
fn scalar_mod_n_reduce(a: array<u32, 8>) -> array<u32, 8> {
    if (scalar_cmp(a, SECP_N) >= 0) {
        return scalar_sub_internal(a, SECP_N);
    }
    return a;
}

// -----------------------------------------------------------------------------
// Addition: (a + b) mod n
// -----------------------------------------------------------------------------
fn scalar_mod_n_add(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var c: array<u32, 8>;
    var carry: u32 = 0u;
    var sum: u32;

    sum = a[0] + b[0]; carry = select(0u, 1u, sum < a[0]); c[0] = sum;
    sum = a[1] + b[1] + carry; carry = select(0u, 1u, sum < a[1] || (carry == 1u && sum == a[1])); c[1] = sum;
    sum = a[2] + b[2] + carry; carry = select(0u, 1u, sum < a[2] || (carry == 1u && sum == a[2])); c[2] = sum;
    sum = a[3] + b[3] + carry; carry = select(0u, 1u, sum < a[3] || (carry == 1u && sum == a[3])); c[3] = sum;
    sum = a[4] + b[4] + carry; carry = select(0u, 1u, sum < a[4] || (carry == 1u && sum == a[4])); c[4] = sum;
    sum = a[5] + b[5] + carry; carry = select(0u, 1u, sum < a[5] || (carry == 1u && sum == a[5])); c[5] = sum;
    sum = a[6] + b[6] + carry; carry = select(0u, 1u, sum < a[6] || (carry == 1u && sum == a[6])); c[6] = sum;
    sum = a[7] + b[7] + carry; carry = select(0u, 1u, sum < a[7] || (carry == 1u && sum == a[7])); c[7] = sum;

    // If overflow (carry=1), result >= 2^256, which is definitely >= n, so subtract n.
    // Otherwise, result might still be >= n, so conditionally reduce.
    if (carry == 1u) {
        // a + b >= 2^256. Since a,b < n < 2^256, we have a+b < 2n < 2^257.
        // So a+b - n < 2^256, guaranteed to fit. But a+b wrapped mod 2^256 to c,
        // so c = a+b - 2^256. We need (a+b) - n = c + (2^256 - n) = c + NC.
        var carry2: u32 = 0u;
        var s2: u32;
        s2 = c[0] + NC0; carry2 = select(0u, 1u, s2 < c[0]); c[0] = s2;
        s2 = c[1] + NC1 + carry2; carry2 = select(0u, 1u, s2 < c[1] || (carry2 == 1u && s2 == c[1])); c[1] = s2;
        s2 = c[2] + NC2 + carry2; carry2 = select(0u, 1u, s2 < c[2] || (carry2 == 1u && s2 == c[2])); c[2] = s2;
        s2 = c[3] + NC3 + carry2; carry2 = select(0u, 1u, s2 < c[3] || (carry2 == 1u && s2 == c[3])); c[3] = s2;
        s2 = c[4] + NC4 + carry2; carry2 = select(0u, 1u, s2 < c[4] || (carry2 == 1u && s2 == c[4])); c[4] = s2;
        // NC5..NC7 are 0, just propagate carry
        if (carry2 > 0u) { s2 = c[5] + carry2; carry2 = select(0u, 1u, s2 < c[5]); c[5] = s2; }
        if (carry2 > 0u) { s2 = c[6] + carry2; carry2 = select(0u, 1u, s2 < c[6]); c[6] = s2; }
        if (carry2 > 0u) { c[7] = c[7] + carry2; }
        return c;
    }

    return scalar_mod_n_reduce(c);
}

// -----------------------------------------------------------------------------
// Subtraction: (a - b) mod n
// If a < b, adds n to result
// -----------------------------------------------------------------------------
fn scalar_mod_n_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var c: array<u32, 8>;
    var borrow: u32 = 0u;
    var diff: u32;

    diff = a[0] - b[0]; borrow = select(0u, 1u, a[0] < b[0]); c[0] = diff;
    diff = a[1] - b[1] - borrow; borrow = select(0u, 1u, a[1] < b[1] + borrow || (borrow == 1u && b[1] == 0xFFFFFFFFu)); c[1] = diff;
    diff = a[2] - b[2] - borrow; borrow = select(0u, 1u, a[2] < b[2] + borrow || (borrow == 1u && b[2] == 0xFFFFFFFFu)); c[2] = diff;
    diff = a[3] - b[3] - borrow; borrow = select(0u, 1u, a[3] < b[3] + borrow || (borrow == 1u && b[3] == 0xFFFFFFFFu)); c[3] = diff;
    diff = a[4] - b[4] - borrow; borrow = select(0u, 1u, a[4] < b[4] + borrow || (borrow == 1u && b[4] == 0xFFFFFFFFu)); c[4] = diff;
    diff = a[5] - b[5] - borrow; borrow = select(0u, 1u, a[5] < b[5] + borrow || (borrow == 1u && b[5] == 0xFFFFFFFFu)); c[5] = diff;
    diff = a[6] - b[6] - borrow; borrow = select(0u, 1u, a[6] < b[6] + borrow || (borrow == 1u && b[6] == 0xFFFFFFFFu)); c[6] = diff;
    diff = a[7] - b[7] - borrow; c[7] = diff; borrow = select(0u, 1u, a[7] < b[7] + borrow || (borrow == 1u && b[7] == 0xFFFFFFFFu));

    if (borrow == 1u) {
        var carry2: u32 = 0u;
        var sum2: u32;
        sum2 = c[0] + N0; carry2 = select(0u, 1u, sum2 < c[0]); c[0] = sum2;
        sum2 = c[1] + N1 + carry2; carry2 = select(0u, 1u, sum2 < c[1] || (carry2 == 1u && sum2 == c[1])); c[1] = sum2;
        sum2 = c[2] + N2 + carry2; carry2 = select(0u, 1u, sum2 < c[2] || (carry2 == 1u && sum2 == c[2])); c[2] = sum2;
        sum2 = c[3] + N3 + carry2; carry2 = select(0u, 1u, sum2 < c[3] || (carry2 == 1u && sum2 == c[3])); c[3] = sum2;
        sum2 = c[4] + N4 + carry2; carry2 = select(0u, 1u, sum2 < c[4] || (carry2 == 1u && sum2 == c[4])); c[4] = sum2;
        sum2 = c[5] + N5 + carry2; carry2 = select(0u, 1u, sum2 < c[5] || (carry2 == 1u && sum2 == c[5])); c[5] = sum2;
        sum2 = c[6] + N6 + carry2; carry2 = select(0u, 1u, sum2 < c[6] || (carry2 == 1u && sum2 == c[6])); c[6] = sum2;
        sum2 = c[7] + N7 + carry2; c[7] = sum2;
    }

    return c;
}

// -----------------------------------------------------------------------------
// Multiplication: (a * b) mod n
//
// Strategy: schoolbook 8x8 -> 16 limbs (same as fe_mul), then reduce upper
// limbs mod n. For each upper limb p[k] (k=8..15), we replace it with
// p[k] * (2^256 - n) added to the lower limbs, since 2^256 ≡ (2^256 - n) mod n.
//
// Unlike field reduction where 2^256 - p = 2^32 + 977 (tiny), here
// 2^256 - n is 129 bits, so each reduction step can produce overflow that
// spills ~5 limbs upward. We reduce from p15 down to p8, then do a second
// pass on p8 if needed, plus final conditional subtraction.
// -----------------------------------------------------------------------------
fn scalar_mod_n_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    // =========================================================================
    // STEP 1: Full schoolbook multiplication -> 16 limbs
    // =========================================================================
    var p0: u32 = 0u; var p1: u32 = 0u; var p2: u32 = 0u; var p3: u32 = 0u;
    var p4: u32 = 0u; var p5: u32 = 0u; var p6: u32 = 0u; var p7: u32 = 0u;
    var p8: u32 = 0u; var p9: u32 = 0u; var p10: u32 = 0u; var p11: u32 = 0u;
    var p12: u32 = 0u; var p13: u32 = 0u; var p14: u32 = 0u; var p15: u32 = 0u;

    var t: vec2<u32>;
    var carry: u32;
    var s: u32;
    var hi: u32;

    // Row 0: a[0] * b[j]
    carry = 0u;
    t = mul32(a[0], b[0]); s = t.x; carry = t.y; p0 = s;
    t = mul32(a[0], b[1]); s = t.x + carry; carry = t.y + select(0u, 1u, s < carry); p1 = s;
    t = mul32(a[0], b[2]); s = t.x + carry; carry = t.y + select(0u, 1u, s < carry); p2 = s;
    t = mul32(a[0], b[3]); s = t.x + carry; carry = t.y + select(0u, 1u, s < carry); p3 = s;
    t = mul32(a[0], b[4]); s = t.x + carry; carry = t.y + select(0u, 1u, s < carry); p4 = s;
    t = mul32(a[0], b[5]); s = t.x + carry; carry = t.y + select(0u, 1u, s < carry); p5 = s;
    t = mul32(a[0], b[6]); s = t.x + carry; carry = t.y + select(0u, 1u, s < carry); p6 = s;
    t = mul32(a[0], b[7]); s = t.x + carry; carry = t.y + select(0u, 1u, s < carry); p7 = s;
    p8 = carry;

    // Row 1
    carry = 0u;
    t = mul32(a[1], b[0]); s = p1 + t.x; hi = select(0u, 1u, s < p1); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p1 = s;
    t = mul32(a[1], b[1]); s = p2 + t.x; hi = select(0u, 1u, s < p2); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p2 = s;
    t = mul32(a[1], b[2]); s = p3 + t.x; hi = select(0u, 1u, s < p3); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p3 = s;
    t = mul32(a[1], b[3]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[1], b[4]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[1], b[5]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[1], b[6]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[1], b[7]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    p9 = carry;

    // Row 2
    carry = 0u;
    t = mul32(a[2], b[0]); s = p2 + t.x; hi = select(0u, 1u, s < p2); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p2 = s;
    t = mul32(a[2], b[1]); s = p3 + t.x; hi = select(0u, 1u, s < p3); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p3 = s;
    t = mul32(a[2], b[2]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[2], b[3]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[2], b[4]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[2], b[5]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[2], b[6]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[2], b[7]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    p10 = carry;

    // Row 3
    carry = 0u;
    t = mul32(a[3], b[0]); s = p3 + t.x; hi = select(0u, 1u, s < p3); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p3 = s;
    t = mul32(a[3], b[1]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[3], b[2]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[3], b[3]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[3], b[4]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[3], b[5]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[3], b[6]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[3], b[7]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    p11 = carry;

    // Row 4
    carry = 0u;
    t = mul32(a[4], b[0]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[4], b[1]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[4], b[2]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[4], b[3]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[4], b[4]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[4], b[5]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[4], b[6]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    t = mul32(a[4], b[7]); s = p11 + t.x; hi = select(0u, 1u, s < p11); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p11 = s;
    p12 = carry;

    // Row 5
    carry = 0u;
    t = mul32(a[5], b[0]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[5], b[1]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[5], b[2]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[5], b[3]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[5], b[4]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[5], b[5]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    t = mul32(a[5], b[6]); s = p11 + t.x; hi = select(0u, 1u, s < p11); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p11 = s;
    t = mul32(a[5], b[7]); s = p12 + t.x; hi = select(0u, 1u, s < p12); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p12 = s;
    p13 = carry;

    // Row 6
    carry = 0u;
    t = mul32(a[6], b[0]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[6], b[1]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[6], b[2]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[6], b[3]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[6], b[4]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    t = mul32(a[6], b[5]); s = p11 + t.x; hi = select(0u, 1u, s < p11); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p11 = s;
    t = mul32(a[6], b[6]); s = p12 + t.x; hi = select(0u, 1u, s < p12); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p12 = s;
    t = mul32(a[6], b[7]); s = p13 + t.x; hi = select(0u, 1u, s < p13); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p13 = s;
    p14 = carry;

    // Row 7
    carry = 0u;
    t = mul32(a[7], b[0]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[7], b[1]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[7], b[2]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[7], b[3]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    t = mul32(a[7], b[4]); s = p11 + t.x; hi = select(0u, 1u, s < p11); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p11 = s;
    t = mul32(a[7], b[5]); s = p12 + t.x; hi = select(0u, 1u, s < p12); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p12 = s;
    t = mul32(a[7], b[6]); s = p13 + t.x; hi = select(0u, 1u, s < p13); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p13 = s;
    t = mul32(a[7], b[7]); s = p14 + t.x; hi = select(0u, 1u, s < p14); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p14 = s;
    p15 = carry;

    // =========================================================================
    // STEP 2: Reduce upper limbs mod n
    //
    // For each upper limb h at position k (k=8..15), we have:
    //   h * 2^(32*k) ≡ h * NC * 2^(32*(k-8)) (mod n)
    //
    // We process from p15 down to p8. Each h * NC is at most 161 bits,
    // so it spans 6 limbs from position (k-8). The multiplication by NC
    // (5 limbs) can cause overflow into the next upper limbs, which is
    // why we process top-down and may need a second pass on p8.
    // =========================================================================
    var h: u32;

    // Reduce p15: h * NC added at offset 7 (limbs p7..p12+)
    // p15 * 2^(32*15) ≡ p15 * NC * 2^(32*7) mod n
    h = p15; p15 = 0u;
    if (h != 0u) {
        var c: u32;
        var old: u32;
        t = mul32(h, NC0);
        old = p7; p7 = p7 + t.x; c = select(0u, 1u, p7 < old);
        s = p8 + t.y; hi = select(0u, 1u, s < p8); s = s + c; hi = hi + select(0u, 1u, s < c); p8 = s; c = hi;
        t = mul32(h, NC1);
        s = p8 + t.x; hi = select(0u, 1u, s < p8); p8 = s;
        s = p9 + t.y + hi; hi = select(0u, 1u, s < p9 || (hi == 1u && s == p9)); s = s + c; hi = hi + select(0u, 1u, s < c); p9 = s; c = hi;
        t = mul32(h, NC2);
        s = p9 + t.x; hi = select(0u, 1u, s < p9); p9 = s;
        s = p10 + t.y + hi; hi = select(0u, 1u, s < p10 || (hi == 1u && s == p10)); s = s + c; hi = hi + select(0u, 1u, s < c); p10 = s; c = hi;
        t = mul32(h, NC3);
        s = p10 + t.x; hi = select(0u, 1u, s < p10); p10 = s;
        s = p11 + t.y + hi; hi = select(0u, 1u, s < p11 || (hi == 1u && s == p11)); s = s + c; hi = hi + select(0u, 1u, s < c); p11 = s; c = hi;
        // NC4 = 1, so h * NC4 = h
        s = p11 + h; hi = select(0u, 1u, s < p11); p11 = s;
        c = c + hi;
        if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
        if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
        if (c > 0u) { p14 = p14 + c; }
    }

    // Reduce p14: h * NC added at offset 6
    h = p14; p14 = 0u;
    if (h != 0u) {
        var c: u32;
        var old: u32;
        t = mul32(h, NC0);
        old = p6; p6 = p6 + t.x; c = select(0u, 1u, p6 < old);
        s = p7 + t.y; hi = select(0u, 1u, s < p7); s = s + c; hi = hi + select(0u, 1u, s < c); p7 = s; c = hi;
        t = mul32(h, NC1);
        s = p7 + t.x; hi = select(0u, 1u, s < p7); p7 = s;
        s = p8 + t.y + hi; hi = select(0u, 1u, s < p8 || (hi == 1u && s == p8)); s = s + c; hi = hi + select(0u, 1u, s < c); p8 = s; c = hi;
        t = mul32(h, NC2);
        s = p8 + t.x; hi = select(0u, 1u, s < p8); p8 = s;
        s = p9 + t.y + hi; hi = select(0u, 1u, s < p9 || (hi == 1u && s == p9)); s = s + c; hi = hi + select(0u, 1u, s < c); p9 = s; c = hi;
        t = mul32(h, NC3);
        s = p9 + t.x; hi = select(0u, 1u, s < p9); p9 = s;
        s = p10 + t.y + hi; hi = select(0u, 1u, s < p10 || (hi == 1u && s == p10)); s = s + c; hi = hi + select(0u, 1u, s < c); p10 = s; c = hi;
        s = p10 + h; hi = select(0u, 1u, s < p10); p10 = s;
        c = c + hi;
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
        if (c > 0u) { p13 = p13 + c; }
    }

    // Reduce p13: h * NC added at offset 5
    h = p13; p13 = 0u;
    if (h != 0u) {
        var c: u32;
        var old: u32;
        t = mul32(h, NC0);
        old = p5; p5 = p5 + t.x; c = select(0u, 1u, p5 < old);
        s = p6 + t.y; hi = select(0u, 1u, s < p6); s = s + c; hi = hi + select(0u, 1u, s < c); p6 = s; c = hi;
        t = mul32(h, NC1);
        s = p6 + t.x; hi = select(0u, 1u, s < p6); p6 = s;
        s = p7 + t.y + hi; hi = select(0u, 1u, s < p7 || (hi == 1u && s == p7)); s = s + c; hi = hi + select(0u, 1u, s < c); p7 = s; c = hi;
        t = mul32(h, NC2);
        s = p7 + t.x; hi = select(0u, 1u, s < p7); p7 = s;
        s = p8 + t.y + hi; hi = select(0u, 1u, s < p8 || (hi == 1u && s == p8)); s = s + c; hi = hi + select(0u, 1u, s < c); p8 = s; c = hi;
        t = mul32(h, NC3);
        s = p8 + t.x; hi = select(0u, 1u, s < p8); p8 = s;
        s = p9 + t.y + hi; hi = select(0u, 1u, s < p9 || (hi == 1u && s == p9)); s = s + c; hi = hi + select(0u, 1u, s < c); p9 = s; c = hi;
        s = p9 + h; hi = select(0u, 1u, s < p9); p9 = s;
        c = c + hi;
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { p12 = p12 + c; }
    }

    // Reduce p12: h * NC added at offset 4
    h = p12; p12 = 0u;
    if (h != 0u) {
        var c: u32;
        var old: u32;
        t = mul32(h, NC0);
        old = p4; p4 = p4 + t.x; c = select(0u, 1u, p4 < old);
        s = p5 + t.y; hi = select(0u, 1u, s < p5); s = s + c; hi = hi + select(0u, 1u, s < c); p5 = s; c = hi;
        t = mul32(h, NC1);
        s = p5 + t.x; hi = select(0u, 1u, s < p5); p5 = s;
        s = p6 + t.y + hi; hi = select(0u, 1u, s < p6 || (hi == 1u && s == p6)); s = s + c; hi = hi + select(0u, 1u, s < c); p6 = s; c = hi;
        t = mul32(h, NC2);
        s = p6 + t.x; hi = select(0u, 1u, s < p6); p6 = s;
        s = p7 + t.y + hi; hi = select(0u, 1u, s < p7 || (hi == 1u && s == p7)); s = s + c; hi = hi + select(0u, 1u, s < c); p7 = s; c = hi;
        t = mul32(h, NC3);
        s = p7 + t.x; hi = select(0u, 1u, s < p7); p7 = s;
        s = p8 + t.y + hi; hi = select(0u, 1u, s < p8 || (hi == 1u && s == p8)); s = s + c; hi = hi + select(0u, 1u, s < c); p8 = s; c = hi;
        s = p8 + h; hi = select(0u, 1u, s < p8); p8 = s;
        c = c + hi;
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { p11 = p11 + c; }
    }

    // Reduce p11: h * NC added at offset 3
    h = p11; p11 = 0u;
    if (h != 0u) {
        var c: u32;
        var old: u32;
        t = mul32(h, NC0);
        old = p3; p3 = p3 + t.x; c = select(0u, 1u, p3 < old);
        s = p4 + t.y; hi = select(0u, 1u, s < p4); s = s + c; hi = hi + select(0u, 1u, s < c); p4 = s; c = hi;
        t = mul32(h, NC1);
        s = p4 + t.x; hi = select(0u, 1u, s < p4); p4 = s;
        s = p5 + t.y + hi; hi = select(0u, 1u, s < p5 || (hi == 1u && s == p5)); s = s + c; hi = hi + select(0u, 1u, s < c); p5 = s; c = hi;
        t = mul32(h, NC2);
        s = p5 + t.x; hi = select(0u, 1u, s < p5); p5 = s;
        s = p6 + t.y + hi; hi = select(0u, 1u, s < p6 || (hi == 1u && s == p6)); s = s + c; hi = hi + select(0u, 1u, s < c); p6 = s; c = hi;
        t = mul32(h, NC3);
        s = p6 + t.x; hi = select(0u, 1u, s < p6); p6 = s;
        s = p7 + t.y + hi; hi = select(0u, 1u, s < p7 || (hi == 1u && s == p7)); s = s + c; hi = hi + select(0u, 1u, s < c); p7 = s; c = hi;
        s = p7 + h; hi = select(0u, 1u, s < p7); p7 = s;
        c = c + hi;
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { p10 = p10 + c; }
    }

    // Reduce p10: h * NC added at offset 2
    h = p10; p10 = 0u;
    if (h != 0u) {
        var c: u32;
        var old: u32;
        t = mul32(h, NC0);
        old = p2; p2 = p2 + t.x; c = select(0u, 1u, p2 < old);
        s = p3 + t.y; hi = select(0u, 1u, s < p3); s = s + c; hi = hi + select(0u, 1u, s < c); p3 = s; c = hi;
        t = mul32(h, NC1);
        s = p3 + t.x; hi = select(0u, 1u, s < p3); p3 = s;
        s = p4 + t.y + hi; hi = select(0u, 1u, s < p4 || (hi == 1u && s == p4)); s = s + c; hi = hi + select(0u, 1u, s < c); p4 = s; c = hi;
        t = mul32(h, NC2);
        s = p4 + t.x; hi = select(0u, 1u, s < p4); p4 = s;
        s = p5 + t.y + hi; hi = select(0u, 1u, s < p5 || (hi == 1u && s == p5)); s = s + c; hi = hi + select(0u, 1u, s < c); p5 = s; c = hi;
        t = mul32(h, NC3);
        s = p5 + t.x; hi = select(0u, 1u, s < p5); p5 = s;
        s = p6 + t.y + hi; hi = select(0u, 1u, s < p6 || (hi == 1u && s == p6)); s = s + c; hi = hi + select(0u, 1u, s < c); p6 = s; c = hi;
        s = p6 + h; hi = select(0u, 1u, s < p6); p6 = s;
        c = c + hi;
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { p9 = p9 + c; }
    }

    // Reduce p9: h * NC added at offset 1
    h = p9; p9 = 0u;
    if (h != 0u) {
        var c: u32;
        var old: u32;
        t = mul32(h, NC0);
        old = p1; p1 = p1 + t.x; c = select(0u, 1u, p1 < old);
        s = p2 + t.y; hi = select(0u, 1u, s < p2); s = s + c; hi = hi + select(0u, 1u, s < c); p2 = s; c = hi;
        t = mul32(h, NC1);
        s = p2 + t.x; hi = select(0u, 1u, s < p2); p2 = s;
        s = p3 + t.y + hi; hi = select(0u, 1u, s < p3 || (hi == 1u && s == p3)); s = s + c; hi = hi + select(0u, 1u, s < c); p3 = s; c = hi;
        t = mul32(h, NC2);
        s = p3 + t.x; hi = select(0u, 1u, s < p3); p3 = s;
        s = p4 + t.y + hi; hi = select(0u, 1u, s < p4 || (hi == 1u && s == p4)); s = s + c; hi = hi + select(0u, 1u, s < c); p4 = s; c = hi;
        t = mul32(h, NC3);
        s = p4 + t.x; hi = select(0u, 1u, s < p4); p4 = s;
        s = p5 + t.y + hi; hi = select(0u, 1u, s < p5 || (hi == 1u && s == p5)); s = s + c; hi = hi + select(0u, 1u, s < c); p5 = s; c = hi;
        s = p5 + h; hi = select(0u, 1u, s < p5); p5 = s;
        c = c + hi;
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { p8 = p8 + c; }
    }

    // Reduce p8 (first pass): h * NC added at offset 0
    h = p8; p8 = 0u;
    if (h != 0u) {
        var c: u32;
        var old: u32;
        t = mul32(h, NC0);
        old = p0; p0 = p0 + t.x; c = select(0u, 1u, p0 < old);
        s = p1 + t.y; hi = select(0u, 1u, s < p1); s = s + c; hi = hi + select(0u, 1u, s < c); p1 = s; c = hi;
        t = mul32(h, NC1);
        s = p1 + t.x; hi = select(0u, 1u, s < p1); p1 = s;
        s = p2 + t.y + hi; hi = select(0u, 1u, s < p2 || (hi == 1u && s == p2)); s = s + c; hi = hi + select(0u, 1u, s < c); p2 = s; c = hi;
        t = mul32(h, NC2);
        s = p2 + t.x; hi = select(0u, 1u, s < p2); p2 = s;
        s = p3 + t.y + hi; hi = select(0u, 1u, s < p3 || (hi == 1u && s == p3)); s = s + c; hi = hi + select(0u, 1u, s < c); p3 = s; c = hi;
        t = mul32(h, NC3);
        s = p3 + t.x; hi = select(0u, 1u, s < p3); p3 = s;
        s = p4 + t.y + hi; hi = select(0u, 1u, s < p4 || (hi == 1u && s == p4)); s = s + c; hi = hi + select(0u, 1u, s < c); p4 = s; c = hi;
        s = p4 + h; hi = select(0u, 1u, s < p4); p4 = s;
        c = c + hi;
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { p8 = p8 + c; }
    }

    // Reduce p8 (second pass — overflow from first pass can produce small p8)
    h = p8; p8 = 0u;
    if (h != 0u) {
        var c: u32;
        var old: u32;
        t = mul32(h, NC0);
        old = p0; p0 = p0 + t.x; c = select(0u, 1u, p0 < old);
        s = p1 + t.y; hi = select(0u, 1u, s < p1); s = s + c; hi = hi + select(0u, 1u, s < c); p1 = s; c = hi;
        t = mul32(h, NC1);
        s = p1 + t.x; hi = select(0u, 1u, s < p1); p1 = s;
        s = p2 + t.y + hi; hi = select(0u, 1u, s < p2 || (hi == 1u && s == p2)); s = s + c; hi = hi + select(0u, 1u, s < c); p2 = s; c = hi;
        t = mul32(h, NC2);
        s = p2 + t.x; hi = select(0u, 1u, s < p2); p2 = s;
        s = p3 + t.y + hi; hi = select(0u, 1u, s < p3 || (hi == 1u && s == p3)); s = s + c; hi = hi + select(0u, 1u, s < c); p3 = s; c = hi;
        t = mul32(h, NC3);
        s = p3 + t.x; hi = select(0u, 1u, s < p3); p3 = s;
        s = p4 + t.y + hi; hi = select(0u, 1u, s < p4 || (hi == 1u && s == p4)); s = s + c; hi = hi + select(0u, 1u, s < c); p4 = s; c = hi;
        s = p4 + h; hi = select(0u, 1u, s < p4); p4 = s;
        c = c + hi;
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { p7 = p7 + c; }
    }

    // =========================================================================
    // STEP 3: Final conditional subtraction (result may still be >= n)
    // At most 2 subtractions needed
    // =========================================================================
    var result: array<u32, 8>;
    result[0] = p0; result[1] = p1; result[2] = p2; result[3] = p3;
    result[4] = p4; result[5] = p5; result[6] = p6; result[7] = p7;

    result = scalar_mod_n_reduce(result);
    result = scalar_mod_n_reduce(result);

    return result;
}

// -----------------------------------------------------------------------------
// Squaring: a^2 mod n
// Delegates to scalar_mod_n_mul. Could be optimized with cross-term symmetry
// like fe_square, but correctness first — this is only called in the
// scalar_mod_n_inv chain.
// -----------------------------------------------------------------------------
fn scalar_mod_n_sqr(a: array<u32, 8>) -> array<u32, 8> {
    return scalar_mod_n_mul(a, a);
}

// -----------------------------------------------------------------------------
// Modular inverse: a^(-1) mod n via Fermat's little theorem
// a^(-1) = a^(n-2) mod n
//
// n-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD036413F
//
// Uses an addition chain similar to fe_inv but adapted for n-2.
// n-2 bit structure (256 bits, MSB first):
//   bits 255-129: same as n (all 1s for bits 255-129 except bit 128 which is 0)
//   Actually, let's use a simple square-and-multiply with cached powers.
//
// We use a windowed approach: precompute a^1..a^15, then process 4 bits at
// a time from MSB. 256 bits = 64 nybbles. This costs 252 squarings + ~60
// multiplies, acceptable since scalar_mod_n_inv is called once per signature.
// -----------------------------------------------------------------------------
fn scalar_mod_n_inv(a: array<u32, 8>) -> array<u32, 8> {
    // n - 2 in little-endian u32 limbs
    let nm2 = array<u32, 8>(
        0xD036413Fu, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
        0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    );

    // Precompute table: tab[i] = a^(i+1) for i=0..14
    // tab[0] = a, tab[1] = a^2, ..., tab[14] = a^15
    var tab0 = a;
    let a2 = scalar_mod_n_sqr(a);
    var tab1 = a2;
    var tab2 = scalar_mod_n_mul(a2, a);           // a^3
    var tab3 = scalar_mod_n_mul(a2, a2);           // a^4
    var tab4 = scalar_mod_n_mul(tab3, a);          // a^5
    var tab5 = scalar_mod_n_mul(tab3, a2);         // a^6
    var tab6 = scalar_mod_n_mul(tab4, a2);         // a^7
    var tab7 = scalar_mod_n_mul(tab3, tab3);       // a^8
    var tab8 = scalar_mod_n_mul(tab7, a);          // a^9
    var tab9 = scalar_mod_n_mul(tab7, a2);         // a^10
    var tab10 = scalar_mod_n_mul(tab8, a2);        // a^11
    var tab11 = scalar_mod_n_mul(tab9, a2);        // a^12
    var tab12 = scalar_mod_n_mul(tab10, a2);       // a^13
    var tab13 = scalar_mod_n_mul(tab11, a2);       // a^14
    var tab14 = scalar_mod_n_mul(tab12, a2);       // a^15

    // Process 4 bits at a time from MSB (bit 255) down to bit 0
    // Start with the top nybble (bits 255-252)
    // nm2[7] = 0xFFFFFFFF, top nybble = 0xF -> tab[14] = a^15
    var result = tab14;

    // Process remaining 63 nybbles (bits 251 down to 0)
    // We iterate over limbs 7..0, and within each limb over nybbles 6..0
    // (limb 7 nybble 7 already processed above)

    // Limb 7: nybbles 6..0 (all 0xF)
    // Loop runs for j = 6, 5, 4, 3, 2, 1, 0 (7 iterations).
    // When j=0, 0 < 7 is true so body executes; then j wraps to 0xFFFFFFFF, exits.
    for (var j = 6u; j < 7u; j = j - 1u) {
        result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
        result = scalar_mod_n_mul(result, tab14); // nybble = F
    }

    // Limb 6: 0xFFFFFFFF — all nybbles are F
    for (var j = 0u; j < 8u; j = j + 1u) {
        result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
        result = scalar_mod_n_mul(result, tab14);
    }

    // Limb 5: 0xFFFFFFFF — all nybbles are F
    for (var j = 0u; j < 8u; j = j + 1u) {
        result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
        result = scalar_mod_n_mul(result, tab14);
    }

    // Limb 4: 0xFFFFFFFE — nybbles: F F F F F F F E
    for (var j = 0u; j < 7u; j = j + 1u) {
        result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
        result = scalar_mod_n_mul(result, tab14);
    }
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab13); // E = 14 -> tab[13]

    // Limb 3: 0xBAAEDCE6 — nybbles: B A A E D C E 6
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab10); // B = 11 -> tab[10]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab9);  // A = 10 -> tab[9]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab9);  // A
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab13); // E = 14 -> tab[13]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab12); // D = 13 -> tab[12]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab11); // C = 12 -> tab[11]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab13); // E = 14 -> tab[13]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab5);  // 6 -> tab[5]

    // Limb 2: 0xAF48A03B — nybbles: A F 4 8 A 0 3 B
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab9);  // A
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab14); // F
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab3);  // 4 -> tab[3]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab7);  // 8 -> tab[7]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab9);  // A
    // nybble 0: 4 squarings, no multiply (but we still need to handle it)
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    // 0 nybble — no multiply
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab2);  // 3 -> tab[2]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab10); // B = 11 -> tab[10]

    // Limb 1: 0xBFD25E8C — nybbles: B F D 2 5 E 8 C
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab10); // B
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab14); // F
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab12); // D
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab1);  // 2 -> tab[1]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab4);  // 5 -> tab[4]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab13); // E
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab7);  // 8 -> tab[7]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab11); // C

    // Limb 0: 0xD036413F — nybbles: D 0 3 6 4 1 3 F
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab12); // D
    // nybble 0
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    // 0 nybble — no multiply
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab2);  // 3
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab5);  // 6
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab3);  // 4
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab0);  // 1 -> tab[0]
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab2);  // 3
    result = scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(scalar_mod_n_sqr(result))));
    result = scalar_mod_n_mul(result, tab14); // F

    return result;
}
