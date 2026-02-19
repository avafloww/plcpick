// secp256k1 elliptic curve operations for Vulkan/GLSL
//
// Field prime p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// Group order n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Generator G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
//                0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
//
// 256-bit integers stored as 8 x uint in LITTLE-ENDIAN limb order
// (limb[0] = least significant 32 bits)
//
// This file is #included by .comp files. Do NOT put #version or #extension here.

// --- 256-bit integer type ---

struct U256 {
    uint d[8]; // little-endian limbs
};

// --- EC Point types ---

struct JacobianPoint {
    U256 x, y, z;
};

struct AffinePoint {
    U256 x, y;
};

#ifdef USE_SHARED_G_TABLE
shared AffinePoint shared_g_table[15];
#endif

// --- Constants ---

// Field prime p
const U256 SECP_P = U256(uint[8](
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
));

// Group order n
const U256 SECP_N = U256(uint[8](
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
));

// N / 2 (for low-s check)
const U256 SECP_N_HALF = U256(uint[8](
    0x681B20A0u, 0xDFE92F46u, 0x57A4501Du, 0x5D576E73u,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu
));

// Generator point G (affine x, y)
const U256 SECP_GX = U256(uint[8](
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
));

const U256 SECP_GY = U256(uint[8](
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
));

// --- Helper: create U256 from a single uint ---

U256 secp_u256_from_u32(uint v) {
    U256 r;
    r.d[0] = v;
    for (int i = 1; i < 8; i++) r.d[i] = 0u;
    return r;
}

// --- Basic 256-bit arithmetic ---

bool secp_is_zero(in U256 a) {
    return (a.d[0] | a.d[1] | a.d[2] | a.d[3] |
            a.d[4] | a.d[5] | a.d[6] | a.d[7]) == 0u;
}

// Compare: returns -1, 0, or 1
int secp_cmp(in U256 a, in U256 b) {
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] < b.d[i]) return -1;
        if (a.d[i] > b.d[i]) return 1;
    }
    return 0;
}

// a + b, returns carry
uint secp_add256(inout U256 r, in U256 a, in U256 b) {
    uint64_t carry = 0ul;
    for (int i = 0; i < 8; i++) {
        carry += uint64_t(a.d[i]) + uint64_t(b.d[i]);
        r.d[i] = uint(carry);
        carry >>= 32;
    }
    return uint(carry);
}

// a - b, returns borrow
uint secp_sub256(inout U256 r, in U256 a, in U256 b) {
    int64_t borrow = 0l;
    for (int i = 0; i < 8; i++) {
        borrow += int64_t(a.d[i]) - int64_t(b.d[i]);
        r.d[i] = uint(borrow);
        borrow >>= 32;
    }
    return uint(borrow & 1l);
}

// --- Field arithmetic (mod p) ---

void secp_field_add(inout U256 r, in U256 a, in U256 b) {
    uint carry = secp_add256(r, a, b);
    if (carry != 0u || secp_cmp(r, SECP_P) >= 0) {
        secp_sub256(r, r, SECP_P);
    }
}

void secp_field_sub(inout U256 r, in U256 a, in U256 b) {
    uint borrow = secp_sub256(r, a, b);
    if (borrow != 0u) {
        secp_add256(r, r, SECP_P);
    }
}

void secp_field_negate(inout U256 r, in U256 a) {
    if (secp_is_zero(a)) {
        r = a;
    } else {
        secp_sub256(r, SECP_P, a);
    }
}

// Multiplication mod p using schoolbook with secp256k1-specific reduction
// secp256k1 has p = 2^256 - 2^32 - 977, so reduction is efficient
void secp_field_mul(inout U256 r, in U256 a, in U256 b) {
    // Full 512-bit product stored in 16 uint64_t limbs (only low 32 bits used per slot)
    uint64_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0ul;

    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0ul;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = uint64_t(a.d[i]) * uint64_t(b.d[j]) + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        t[i + 8] = carry;
    }

    // Reduce mod p using: 2^256 = 2^32 + 977 (mod p)
    {
        uint64_t carry = 0ul;
        for (int i = 0; i < 8; i++) {
            uint64_t hi = t[i + 8];
            carry += t[i] + hi * 977ul;
            t[i] = carry & 0xFFFFFFFFul;
            carry >>= 32;
            carry += hi;
        }

        // carry is at most ~2^33. Fold again.
        uint64_t c_lo = carry * 977ul;
        uint64_t c_hi = carry;
        c_lo += t[0];
        t[0] = c_lo & 0xFFFFFFFFul;
        c_lo >>= 32;
        c_lo += t[1] + c_hi;
        t[1] = c_lo & 0xFFFFFFFFul;
        c_lo >>= 32;
        for (int i = 2; i < 8 && c_lo != 0ul; i++) {
            c_lo += t[i];
            t[i] = c_lo & 0xFFFFFFFFul;
            c_lo >>= 32;
        }
    }

    for (int i = 0; i < 8; i++) {
        r.d[i] = uint(t[i]);
    }

    // Final reduction: if r >= p, subtract p
    if (secp_cmp(r, SECP_P) >= 0) {
        secp_sub256(r, r, SECP_P);
    }
}

void secp_field_sqr(inout U256 r, in U256 a) {
    // Optimized squaring: 36 multiplies vs 64 in full mul (44% fewer)
    // Exploits a[i]*a[j] == a[j]*a[i] symmetry
    uint64_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0ul;

    // 1. Cross terms (28 multiplies): accumulate a[i]*a[j] for i<j
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0ul;
        for (int j = i + 1; j < 8; j++) {
            uint64_t prod = uint64_t(a.d[i]) * uint64_t(a.d[j]) + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        t[i + 8] += carry;
    }

    // 2. Double all cross terms (t = 2 * sum(a[i]*a[j] for i<j))
    uint64_t dc = 0ul;
    for (int i = 0; i < 16; i++) {
        dc += t[i] + t[i];
        t[i] = dc & 0xFFFFFFFFul;
        dc >>= 32;
    }

    // 3. Add diagonal squares (8 multiplies): a[i]^2 into t[2*i]
    dc = 0ul;
    for (int i = 0; i < 8; i++) {
        uint64_t sq = uint64_t(a.d[i]) * uint64_t(a.d[i]);
        dc += t[2*i] + (sq & 0xFFFFFFFFul);
        t[2*i] = dc & 0xFFFFFFFFul;
        dc >>= 32;
        dc += t[2*i+1] + (sq >> 32);
        t[2*i+1] = dc & 0xFFFFFFFFul;
        dc >>= 32;
    }

    // 4. Reduce mod p using: 2^256 = 2^32 + 977 (mod p)
    {
        uint64_t carry = 0ul;
        for (int i = 0; i < 8; i++) {
            uint64_t hi = t[i + 8];
            carry += t[i] + hi * 977ul;
            t[i] = carry & 0xFFFFFFFFul;
            carry >>= 32;
            carry += hi;
        }

        // carry is at most ~2^33. Fold again.
        uint64_t c_lo = carry * 977ul;
        uint64_t c_hi = carry;
        c_lo += t[0];
        t[0] = c_lo & 0xFFFFFFFFul;
        c_lo >>= 32;
        c_lo += t[1] + c_hi;
        t[1] = c_lo & 0xFFFFFFFFul;
        c_lo >>= 32;
        for (int i = 2; i < 8 && c_lo != 0ul; i++) {
            c_lo += t[i];
            t[i] = c_lo & 0xFFFFFFFFul;
            c_lo >>= 32;
        }
    }

    for (int i = 0; i < 8; i++) r.d[i] = uint(t[i]);

    // Final reduction: if r >= p, subtract p
    if (secp_cmp(r, SECP_P) >= 0) {
        secp_sub256(r, r, SECP_P);
    }
}

// Modular inverse via Fermat's little theorem: a^(-1) = a^(p-2) mod p
void secp_field_inv(inout U256 r, in U256 a) {
    U256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;

    // x2 = a^(2^2 - 1)
    secp_field_sqr(t, a);
    secp_field_mul(x2, t, a);

    // x3 = a^(2^3 - 1)
    secp_field_sqr(t, x2);
    secp_field_mul(x3, t, a);

    // x6 = a^(2^6 - 1)
    t = x3;
    for (int i = 0; i < 3; i++) secp_field_sqr(t, t);
    secp_field_mul(x6, t, x3);

    // x9 = a^(2^9 - 1)
    t = x6;
    for (int i = 0; i < 3; i++) secp_field_sqr(t, t);
    secp_field_mul(x9, t, x3);

    // x11 = a^(2^11 - 1)
    t = x9;
    for (int i = 0; i < 2; i++) secp_field_sqr(t, t);
    secp_field_mul(x11, t, x2);

    // x22 = a^(2^22 - 1)
    t = x11;
    for (int i = 0; i < 11; i++) secp_field_sqr(t, t);
    secp_field_mul(x22, t, x11);

    // x44 = a^(2^44 - 1)
    t = x22;
    for (int i = 0; i < 22; i++) secp_field_sqr(t, t);
    secp_field_mul(x44, t, x22);

    // x88 = a^(2^88 - 1)
    t = x44;
    for (int i = 0; i < 44; i++) secp_field_sqr(t, t);
    secp_field_mul(x88, t, x44);

    // x176 = a^(2^176 - 1)
    t = x88;
    for (int i = 0; i < 88; i++) secp_field_sqr(t, t);
    secp_field_mul(x176, t, x88);

    // x220 = a^(2^220 - 1)
    t = x176;
    for (int i = 0; i < 44; i++) secp_field_sqr(t, t);
    secp_field_mul(x220, t, x44);

    // x223 = a^(2^223 - 1)
    t = x220;
    for (int i = 0; i < 3; i++) secp_field_sqr(t, t);
    secp_field_mul(x223, t, x3);

    // t = x223^2 = a^(2^224 - 2)
    secp_field_sqr(t, x223);

    // t = t^(2^32) = a^((2^224 - 2) * 2^32)
    for (int i = 0; i < 32; i++) secp_field_sqr(t, t);

    // Now multiply by a^(0xFFFFFC2D) via square-and-multiply
    U256 acc = a; // a^1
    const uint exp = 0xFFFFFC2Du;
    for (int i = 30; i >= 0; i--) {
        secp_field_sqr(acc, acc);
        if (((exp >> i) & 1u) != 0u) {
            secp_field_mul(acc, acc, a);
        }
    }

    secp_field_mul(r, t, acc);
}

// --- Scalar arithmetic (mod n) ---

void secp_scalar_add(inout U256 r, in U256 a, in U256 b) {
    uint carry = secp_add256(r, a, b);
    if (carry != 0u || secp_cmp(r, SECP_N) >= 0) {
        secp_sub256(r, r, SECP_N);
    }
}

void secp_scalar_mul(inout U256 r, in U256 a, in U256 b) {
    // Full 512-bit product then reduce mod N
    uint64_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0ul;

    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0ul;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = uint64_t(a.d[i]) * uint64_t(b.d[j]) + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFFul;
            carry = prod >> 32;
        }
        t[i + 8] = carry;
    }

    // Reduce 512-bit t mod n using: 2^256 = R (mod n)
    // R = 2^256 - n, fits in 5 limbs (129 bits)
    const uint R_LIMBS[5] = uint[5](0x2FC9BEBFu, 0x402DA173u, 0x50B75FC4u, 0x45512319u, 0x00000001u);

    // Fold 1: hi=t[8..15] (8 limbs) * R (5 limbs) -> 13 limbs, + lo=t[0..7]
    uint64_t f[13];
    for (int i = 0; i < 13; i++) f[i] = 0ul;

    for (int i = 0; i < 8; i++) {
        uint64_t c = 0ul;
        for (int j = 0; j < 5; j++) {
            c += f[i + j] + uint64_t(uint(t[i + 8])) * uint64_t(R_LIMBS[j]);
            f[i + j] = c & 0xFFFFFFFFul;
            c >>= 32;
        }
        f[i + 5] = c;
    }
    {
        uint64_t c = 0ul;
        for (int i = 0; i < 8; i++) {
            c += f[i] + (t[i] & 0xFFFFFFFFul);
            f[i] = c & 0xFFFFFFFFul;
            c >>= 32;
        }
        for (int i = 8; i < 13 && c != 0ul; i++) {
            c += f[i];
            f[i] = c & 0xFFFFFFFFul;
            c >>= 32;
        }
    }

    // Fold 2: hi=f[8..12] (5 limbs) * R (5 limbs) -> 10 limbs, + lo=f[0..7]
    bool has_high = false;
    for (int i = 8; i < 13; i++) { if (f[i] != 0ul) { has_high = true; break; } }
    if (has_high) {
        uint64_t g[10];
        for (int i = 0; i < 10; i++) g[i] = 0ul;

        for (int i = 0; i < 5; i++) {
            uint64_t c = 0ul;
            for (int j = 0; j < 5; j++) {
                c += g[i + j] + uint64_t(uint(f[i + 8])) * uint64_t(R_LIMBS[j]);
                g[i + j] = c & 0xFFFFFFFFul;
                c >>= 32;
            }
            g[i + 5] = c;
        }
        uint64_t c = 0ul;
        for (int i = 0; i < 8; i++) {
            c += g[i] + (f[i] & 0xFFFFFFFFul);
            f[i] = c & 0xFFFFFFFFul;
            c >>= 32;
        }
        for (int i = 8; i < 10; i++) {
            c += g[i];
            f[i] = c & 0xFFFFFFFFul;
            c >>= 32;
        }
        f[10] = c; f[11] = 0ul; f[12] = 0ul;

        // Fold 3 if needed
        has_high = false;
        for (int i = 8; i < 13; i++) { if (f[i] != 0ul) { has_high = true; break; } }
        if (has_high) {
            uint64_t h[8];
            for (int i = 0; i < 8; i++) h[i] = 0ul;

            for (int i = 0; i < 3; i++) {
                uint64_t c2 = 0ul;
                for (int j = 0; j < 5; j++) {
                    c2 += h[i + j] + uint64_t(uint(f[i + 8])) * uint64_t(R_LIMBS[j]);
                    h[i + j] = c2 & 0xFFFFFFFFul;
                    c2 >>= 32;
                }
                h[i + 5] = c2;
            }
            c = 0ul;
            for (int i = 0; i < 8; i++) {
                c += h[i] + (f[i] & 0xFFFFFFFFul);
                f[i] = c & 0xFFFFFFFFul;
                c >>= 32;
            }
        }
    }

    // Extract final 256-bit result and reduce mod n
    for (int i = 0; i < 8; i++) r.d[i] = uint(f[i]);
    while (secp_cmp(r, SECP_N) >= 0) secp_sub256(r, r, SECP_N);
}

// Scalar inverse via Fermat: a^(n-2) mod n
void secp_scalar_inv(inout U256 r, in U256 a) {
    U256 result;
    result.d[0] = 1u;
    for (int i = 1; i < 8; i++) result.d[i] = 0u;

    // n-2 limbs (little-endian)
    const uint nm2[8] = uint[8](
        0xD036413Fu, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
        0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    );

    for (int limb = 7; limb >= 0; limb--) {
        for (int bit = 31; bit >= 0; bit--) {
            if (limb == 7 && bit == 31) {
                result = a;
                continue;
            }
            secp_scalar_mul(result, result, result);
            if (((nm2[limb] >> bit) & 1u) != 0u) {
                secp_scalar_mul(result, result, a);
            }
        }
    }

    r = result;
}

// --- EC Point operations (Jacobian coordinates) ---
// Point (X, Y, Z) represents affine (X/Z^2, Y/Z^3)
// Point at infinity: Z = 0

bool secp_point_is_infinity(in JacobianPoint p) {
    return secp_is_zero(p.z);
}

// Point doubling: R = 2*P
// Using formulas from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
void secp_point_double(inout JacobianPoint r, in JacobianPoint p) {
    if (secp_point_is_infinity(p)) {
        r = p;
        return;
    }

    // Save values needed after r.x/r.y are written (in case r aliases p)
    U256 py = p.y;
    U256 pz = p.z;

    U256 aa, bb, cc, dd, ee, ff;

    // A = X1^2
    secp_field_sqr(aa, p.x);
    // B = Y1^2
    secp_field_sqr(bb, py);
    // C = B^2
    secp_field_sqr(cc, bb);

    // D = 2*((X1+B)^2 - A - C)
    U256 t;
    secp_field_add(t, p.x, bb);
    secp_field_sqr(dd, t);
    secp_field_sub(dd, dd, aa);
    secp_field_sub(dd, dd, cc);
    secp_field_add(dd, dd, dd);

    // E = 3*A
    secp_field_add(ee, aa, aa);
    secp_field_add(ee, ee, aa);

    // F = E^2
    secp_field_sqr(ff, ee);

    // X3 = F - 2*D
    U256 d2;
    secp_field_add(d2, dd, dd);
    secp_field_sub(r.x, ff, d2);

    // Y3 = E*(D - X3) - 8*C
    secp_field_sub(t, dd, r.x);
    secp_field_mul(r.y, ee, t);
    U256 c8;
    secp_field_add(c8, cc, cc);   // 2C
    secp_field_add(c8, c8, c8);   // 4C
    secp_field_add(c8, c8, c8);   // 8C
    secp_field_sub(r.y, r.y, c8);

    // Z3 = 2*Y1*Z1
    secp_field_mul(r.z, py, pz);
    secp_field_add(r.z, r.z, r.z);
}

// Point addition: R = P + Q (general case)
void secp_point_add(inout JacobianPoint r, in JacobianPoint p, in JacobianPoint q) {
    if (secp_point_is_infinity(p)) { r = q; return; }
    if (secp_point_is_infinity(q)) { r = p; return; }

    U256 z1z1, z2z2, u1, u2, s1, s2, h, ii, j, rr, v;

    secp_field_sqr(z1z1, p.z);
    secp_field_sqr(z2z2, q.z);
    secp_field_mul(u1, p.x, z2z2);
    secp_field_mul(u2, q.x, z1z1);
    secp_field_mul(s1, p.y, q.z);
    secp_field_mul(s1, s1, z2z2);
    secp_field_mul(s2, q.y, p.z);
    secp_field_mul(s2, s2, z1z1);

    secp_field_sub(h, u2, u1);

    if (secp_is_zero(h)) {
        U256 sdiff;
        secp_field_sub(sdiff, s2, s1);
        if (secp_is_zero(sdiff)) {
            secp_point_double(r, p);
            return;
        } else {
            r.x = secp_u256_from_u32(0u);
            r.y = secp_u256_from_u32(1u);
            r.z = secp_u256_from_u32(0u);
            return;
        }
    }

    // I = (2*H)^2
    U256 h2;
    secp_field_add(h2, h, h);
    secp_field_sqr(ii, h2);

    // J = H*I
    secp_field_mul(j, h, ii);

    // rr = 2*(S2 - S1)
    secp_field_sub(rr, s2, s1);
    secp_field_add(rr, rr, rr);

    // V = U1*I
    secp_field_mul(v, u1, ii);

    // X3 = rr^2 - J - 2*V
    secp_field_sqr(r.x, rr);
    secp_field_sub(r.x, r.x, j);
    U256 v2;
    secp_field_add(v2, v, v);
    secp_field_sub(r.x, r.x, v2);

    // Y3 = rr*(V - X3) - 2*S1*J
    U256 t;
    secp_field_sub(t, v, r.x);
    secp_field_mul(r.y, rr, t);
    secp_field_mul(t, s1, j);
    secp_field_add(t, t, t);
    secp_field_sub(r.y, r.y, t);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    secp_field_add(t, p.z, q.z);
    secp_field_sqr(t, t);
    secp_field_sub(t, t, z1z1);
    secp_field_sub(t, t, z2z2);
    secp_field_mul(r.z, t, h);
}

// Mixed addition: R = P + Q where Q is affine (Z=1)
void secp_point_add_affine(inout JacobianPoint r, in JacobianPoint p, in U256 qx, in U256 qy) {
    if (secp_point_is_infinity(p)) {
        r.x = qx;
        r.y = qy;
        r.z = secp_u256_from_u32(1u);
        return;
    }

    // Save p.y before r.y is written (in case r aliases p)
    U256 py = p.y;

    U256 z1z1, u2, s2, h, hh, ii, j, rr, v;

    secp_field_sqr(z1z1, p.z);
    secp_field_mul(u2, qx, z1z1);
    U256 z1_cubed;
    secp_field_mul(z1_cubed, p.z, z1z1);
    secp_field_mul(s2, qy, z1_cubed);

    secp_field_sub(h, u2, p.x);

    if (secp_is_zero(h)) {
        U256 sdiff;
        secp_field_sub(sdiff, s2, py);
        if (secp_is_zero(sdiff)) {
            secp_point_double(r, p);
            return;
        } else {
            r.x = secp_u256_from_u32(0u);
            r.y = secp_u256_from_u32(1u);
            r.z = secp_u256_from_u32(0u);
            return;
        }
    }

    secp_field_sqr(hh, h);
    secp_field_add(ii, hh, hh);
    secp_field_add(ii, ii, ii); // I = 4*H^2

    secp_field_mul(j, h, ii);

    secp_field_sub(rr, s2, py);
    secp_field_add(rr, rr, rr);

    secp_field_mul(v, p.x, ii);

    secp_field_sqr(r.x, rr);
    secp_field_sub(r.x, r.x, j);
    U256 v2;
    secp_field_add(v2, v, v);
    secp_field_sub(r.x, r.x, v2);

    U256 t;
    secp_field_sub(t, v, r.x);
    secp_field_mul(r.y, rr, t);
    secp_field_mul(t, py, j);
    secp_field_add(t, t, t);
    secp_field_sub(r.y, r.y, t);

    secp_field_add(r.z, p.z, h);
    secp_field_sqr(r.z, r.z);
    secp_field_sub(r.z, r.z, z1z1);
    secp_field_sub(r.z, r.z, hh);
}

// Convert Jacobian to affine (must come before init_g_table which calls it)
void secp_jacobian_to_affine(inout U256 ax, inout U256 ay, in JacobianPoint p) {
    U256 z_inv, z_inv2, z_inv3;
    secp_field_inv(z_inv, p.z);
    secp_field_sqr(z_inv2, z_inv);
    secp_field_mul(z_inv3, z_inv2, z_inv);
    secp_field_mul(ax, p.x, z_inv2);
    secp_field_mul(ay, p.y, z_inv3);
}

// --- Scalar multiplication: k * G using 4-bit windowed method ---

// Compute G_TABLE inline (called once before scalar_mul_G)
void secp_init_g_table(inout AffinePoint table[15]) {
    // table[0] = G
    table[0].x = SECP_GX;
    table[0].y = SECP_GY;

    // Compute 2*G, 3*G, ..., 15*G
    JacobianPoint acc;
    acc.x = SECP_GX;
    acc.y = SECP_GY;
    acc.z = secp_u256_from_u32(1u);

    for (int i = 1; i < 15; i++) {
        JacobianPoint next;
        secp_point_add_affine(next, acc, SECP_GX, SECP_GY);
        acc = next;
        // Convert to affine
        U256 ax, ay;
        secp_jacobian_to_affine(ax, ay, acc);
        table[i].x = ax;
        table[i].y = ay;
    }
}

// k * G using windowed method (4-bit window)
// g_table must be pre-populated with [1*G, 2*G, ..., 15*G] in affine
void secp_scalar_mul_G(inout JacobianPoint r, in U256 k, in AffinePoint g_table[15]) {
    // Initialize result to point at infinity
    for (int i = 0; i < 8; i++) { r.x.d[i] = 0u; r.y.d[i] = 0u; r.z.d[i] = 0u; }

    // Process 4 bits at a time from MSB
    for (int i = 63; i >= 0; i--) {
        // Double 4 times (skip on first iteration)
        if (i < 63) {
            secp_point_double(r, r);
            secp_point_double(r, r);
            secp_point_double(r, r);
            secp_point_double(r, r);
        }

        // Extract 4-bit window
        int limb_idx = i / 8;
        int bit_offset = (i % 8) * 4;
        uint window = (k.d[limb_idx] >> bit_offset) & 0xFu;

        if (window != 0u) {
            if (secp_point_is_infinity(r)) {
                r.x = g_table[window - 1u].x;
                r.y = g_table[window - 1u].y;
                r.z = secp_u256_from_u32(1u);
            } else {
                secp_point_add_affine(r, r, g_table[window - 1u].x, g_table[window - 1u].y);
            }
        }
    }
}

#ifdef USE_SHARED_G_TABLE
// k * G using windowed method, reading from workgroup shared memory
void secp_scalar_mul_G(inout JacobianPoint r, in U256 k) {
    // Initialize result to point at infinity
    for (int i = 0; i < 8; i++) { r.x.d[i] = 0u; r.y.d[i] = 0u; r.z.d[i] = 0u; }

    // Process 4 bits at a time from MSB
    for (int i = 63; i >= 0; i--) {
        // Double 4 times (skip on first iteration)
        if (i < 63) {
            secp_point_double(r, r);
            secp_point_double(r, r);
            secp_point_double(r, r);
            secp_point_double(r, r);
        }

        // Extract 4-bit window
        int limb_idx = i / 8;
        int bit_offset = (i % 8) * 4;
        uint window = (k.d[limb_idx] >> bit_offset) & 0xFu;

        if (window != 0u) {
            if (secp_point_is_infinity(r)) {
                r.x = shared_g_table[window - 1u].x;
                r.y = shared_g_table[window - 1u].y;
                r.z = secp_u256_from_u32(1u);
            } else {
                secp_point_add_affine(r, r, shared_g_table[window - 1u].x, shared_g_table[window - 1u].y);
            }
        }
    }
}
#endif

// Get compressed public key as uint array (33 uint values, each holding one byte)
void secp_get_compressed_pubkey_uint(inout uint out33[33], in JacobianPoint pub_key) {
    U256 ax, ay;
    secp_jacobian_to_affine(ax, ay, pub_key);

    out33[0] = (ay.d[0] & 1u) != 0u ? 0x03u : 0x02u;

    for (int i = 7; i >= 0; i--) {
        int offset = 1 + (7 - i) * 4;
        out33[offset]     = (ax.d[i] >> 24) & 0xFFu;
        out33[offset + 1] = (ax.d[i] >> 16) & 0xFFu;
        out33[offset + 2] = (ax.d[i] >> 8) & 0xFFu;
        out33[offset + 3] = ax.d[i] & 0xFFu;
    }
}

// Load a 32-byte big-endian scalar into U256 (little-endian limbs)
// Input: 32 uint values, each holding one byte (big-endian order)
void secp_load_scalar(inout U256 r, in uint bytes32[32]) {
    for (int i = 0; i < 8; i++) {
        int offset = (7 - i) * 4;
        r.d[i] = (bytes32[offset] << 24) |
                 (bytes32[offset + 1] << 16) |
                 (bytes32[offset + 2] << 8) |
                 bytes32[offset + 3];
    }
}

// Store U256 as 32-byte big-endian
// Output: 32 uint values, each holding one byte
void secp_store_scalar(inout uint bytes32[32], in U256 a) {
    for (int i = 0; i < 8; i++) {
        int offset = (7 - i) * 4;
        bytes32[offset]     = (a.d[i] >> 24) & 0xFFu;
        bytes32[offset + 1] = (a.d[i] >> 16) & 0xFFu;
        bytes32[offset + 2] = (a.d[i] >> 8) & 0xFFu;
        bytes32[offset + 3] = a.d[i] & 0xFFu;
    }
}
