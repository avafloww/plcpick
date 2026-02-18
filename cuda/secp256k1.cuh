#pragma once
#include <cstdint>

// secp256k1 elliptic curve operations for CUDA
//
// Field prime p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// Group order n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Generator G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
//                0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
//
// 256-bit integers stored as 8 x uint32_t in LITTLE-ENDIAN limb order
// (limb[0] = least significant 32 bits)

namespace secp256k1 {

// --- 256-bit integer type ---

struct U256 {
    uint32_t d[8]; // little-endian limbs
};

// --- Constants ---

// Field prime p
__constant__ static const U256 P = {{
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
}};

// Group order n
__constant__ static const U256 N = {{
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
}};

// N / 2 (for low-s check)
__constant__ static const U256 N_HALF = {{
    0x681B20A0, 0xDFE92F46, 0x57A4501D, 0x5D576E73,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF
}};

// Generator point G (affine x, y)
__constant__ static const U256 GX = {{
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
}};

__constant__ static const U256 GY = {{
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
}};

// --- Basic 256-bit arithmetic ---

__device__ __forceinline__ bool is_zero(const U256 &a) {
    return (a.d[0] | a.d[1] | a.d[2] | a.d[3] |
            a.d[4] | a.d[5] | a.d[6] | a.d[7]) == 0;
}

// Compare: returns -1, 0, or 1
__device__ __forceinline__ int cmp(const U256 &a, const U256 &b) {
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] < b.d[i]) return -1;
        if (a.d[i] > b.d[i]) return 1;
    }
    return 0;
}

// a + b, returns carry
__device__ __forceinline__ uint32_t add256(U256 &r, const U256 &a, const U256 &b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a.d[i] + b.d[i];
        r.d[i] = (uint32_t)carry;
        carry >>= 32;
    }
    return (uint32_t)carry;
}

// a - b, returns borrow
__device__ __forceinline__ uint32_t sub256(U256 &r, const U256 &a, const U256 &b) {
    int64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        borrow += (int64_t)a.d[i] - b.d[i];
        r.d[i] = (uint32_t)borrow;
        borrow >>= 32;
    }
    return (uint32_t)(borrow & 1);
}

// --- Field arithmetic (mod p) ---

__device__ void field_add(U256 &r, const U256 &a, const U256 &b) {
    uint32_t carry = add256(r, a, b);
    if (carry || cmp(r, P) >= 0) {
        sub256(r, r, P);
    }
}

__device__ void field_sub(U256 &r, const U256 &a, const U256 &b) {
    uint32_t borrow = sub256(r, a, b);
    if (borrow) {
        add256(r, r, P);
    }
}

__device__ void field_negate(U256 &r, const U256 &a) {
    if (is_zero(a)) {
        r = a;
    } else {
        sub256(r, P, a);
    }
}

// Multiplication mod p using schoolbook with secp256k1-specific reduction
// secp256k1 has p = 2^256 - 2^32 - 977, so reduction is efficient
__device__ void field_mul(U256 &r, const U256 &a, const U256 &b) {
    // Full 512-bit product
    uint64_t t[16] = {0};

    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a.d[i] * b.d[j] + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        t[i + 8] = carry;
    }

    // Reduce mod p using: 2^256 ≡ 2^32 + 977 (mod p)
    // Process high limbs t[8..15] — each represents t[i] * 2^(32*i)
    // 2^(32*i) for i >= 8: t[i] * 2^(32*i) = t[i] * 2^256 * 2^(32*(i-8))
    //                     ≡ t[i] * (2^32 + 977) * 2^(32*(i-8))
    // We do this iteratively, folding high limbs into low limbs

    // First fold: t[8..15] → t[0..7] using 2^256 ≡ 2^32 + 977 (mod p)
    // Split C = 2^32 + 977 to avoid uint64 overflow:
    //   t[i+8] * C = t[i+8] * 977  (add to current limb)
    //              + t[i+8] * 2^32  (add to carry for next limb)
    {
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t hi = t[i + 8];
            carry += t[i] + hi * 977ULL;  // hi*977 fits in ~42 bits
            t[i] = carry & 0xFFFFFFFF;
            carry >>= 32;
            carry += hi;  // the * 2^32 part, added to next limb's carry
        }

        // carry is at most ~2^33. Fold again: carry * 2^256 ≡ carry * (2^32 + 977)
        // Split again to avoid overflow.
        uint64_t c_lo = carry * 977ULL;  // ~2^43, fits in uint64
        uint64_t c_hi = carry;           // the * 2^32 part
        c_lo += t[0];
        t[0] = c_lo & 0xFFFFFFFF;
        c_lo >>= 32;
        c_lo += t[1] + c_hi;
        t[1] = c_lo & 0xFFFFFFFF;
        c_lo >>= 32;
        for (int i = 2; i < 8 && c_lo; i++) {
            c_lo += t[i];
            t[i] = c_lo & 0xFFFFFFFF;
            c_lo >>= 32;
        }
    }

    for (int i = 0; i < 8; i++) {
        r.d[i] = (uint32_t)t[i];
    }

    // Final reduction: if r >= p, subtract p
    if (cmp(r, P) >= 0) {
        sub256(r, r, P);
    }
}

__device__ void field_sqr(U256 &r, const U256 &a) {
    field_mul(r, a, a);
}

// Modular inverse via Fermat's little theorem: a^(-1) = a^(p-2) mod p
__device__ void field_inv(U256 &r, const U256 &a) {
    // p - 2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    // Use addition chain optimized for secp256k1
    // Based on: https://briansmith.org/ecc-inversion-addition-chains-01

    U256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;

    // x2 = a^(2^2 - 1)
    field_sqr(t, a);
    field_mul(x2, t, a);

    // x3 = a^(2^3 - 1)
    field_sqr(t, x2);
    field_mul(x3, t, a);

    // x6 = a^(2^6 - 1)
    t = x3;
    for (int i = 0; i < 3; i++) field_sqr(t, t);
    field_mul(x6, t, x3);

    // x9 = a^(2^9 - 1)
    t = x6;
    for (int i = 0; i < 3; i++) field_sqr(t, t);
    field_mul(x9, t, x3);

    // x11 = a^(2^11 - 1)
    t = x9;
    for (int i = 0; i < 2; i++) field_sqr(t, t);
    field_mul(x11, t, x2);

    // x22 = a^(2^22 - 1)
    t = x11;
    for (int i = 0; i < 11; i++) field_sqr(t, t);
    field_mul(x22, t, x11);

    // x44 = a^(2^44 - 1)
    t = x22;
    for (int i = 0; i < 22; i++) field_sqr(t, t);
    field_mul(x44, t, x22);

    // x88 = a^(2^88 - 1)
    t = x44;
    for (int i = 0; i < 44; i++) field_sqr(t, t);
    field_mul(x88, t, x44);

    // x176 = a^(2^176 - 1)
    t = x88;
    for (int i = 0; i < 88; i++) field_sqr(t, t);
    field_mul(x176, t, x88);

    // x220 = a^(2^220 - 1)
    t = x176;
    for (int i = 0; i < 44; i++) field_sqr(t, t);
    field_mul(x220, t, x44);

    // x223 = a^(2^223 - 1)
    t = x220;
    for (int i = 0; i < 3; i++) field_sqr(t, t);
    field_mul(x223, t, x3);

    // Final: a^(p-2) = a^(2^256 - 2^32 - 979)
    // = x223 * 2^33 + 2^32 - 979 ... actually let me just compute directly
    // p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    // = (2^223 - 1) * 2^33 ... no, let me use the standard approach

    // t = x223 << 23 = a^((2^223 - 1) * 2^23)
    t = x223;
    for (int i = 0; i < 23; i++) field_sqr(t, t);
    // t *= x22 = a^((2^223 - 1) * 2^23 + 2^22 - 1) = a^(2^246 - 2^23 + 2^22 - 1)
    field_mul(t, t, x22);
    // Actually this is getting complex. Let me just do the last bits directly.
    // p - 2 last 10 bits: ...FC2D = 1111 1100 0010 1101
    // After x223: we need to square 33 times and multiply by the right thing

    // Restart the tail: from x223, we need exponent:
    // p-2 = 0xFFFF...FEFFFFFC2D
    // The top 223 bits are all 1s. Then: 0_0000_0000_1111_1111_1111_1100_0010_1101
    // Wait, let me recount. p = 2^256 - 2^32 - 977
    // p-2 = 2^256 - 2^32 - 979
    // In hex: FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    // Bits: 256 ones... no. Let me think differently.
    // p in hex: FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    // p-2:      FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    //
    // Top 224 bits (bits 255..32): FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE
    //   = 2^224 - 2 (since FE at the end)
    // Bottom 32 bits: FFFFFC2D = 4294966317
    //
    // So p-2 = (2^224 - 2) * 2^32 + 4294966317
    //
    // Using x223 = a^(2^223 - 1):
    // a^(2^224 - 2) = (a^(2^223 - 1))^2 = x223^2
    // Then a^((2^224 - 2) * 2^32) = square 32 times
    // Then multiply by a^(FFFFFC2D)

    // t = x223^2 = a^(2^224 - 2)
    field_sqr(t, x223);

    // t = t^(2^32) = a^((2^224 - 2) * 2^32)
    for (int i = 0; i < 32; i++) field_sqr(t, t);

    // Now multiply by a^(0xFFFFFC2D)
    // 0xFFFFFC2D = 4294966317
    // Binary: 11111111 11111111 11111100 00101101
    // = 2^32 - 979
    // We need a^(2^32 - 979)
    // Decompose: bits set at positions 31..10 (all ones), 8 is 0, then 0010 1101
    // Actually easier: 0xFFFFFC2D in binary:
    // FF = 11111111, FF = 11111111, FC = 11111100, 2D = 00101101
    // So: 1111_1111_1111_1111_1111_1100_0010_1101

    // Build a^(0xFFFFFC2D) via square-and-multiply
    U256 acc = a; // a^1
    // We process bits from MSB to LSB (bit 31 down to bit 0)
    // bit 31 = 1 (already in acc)
    const uint32_t exp = 0xFFFFFC2D;
    for (int i = 30; i >= 0; i--) {
        field_sqr(acc, acc);
        if ((exp >> i) & 1) {
            field_mul(acc, acc, a);
        }
    }

    field_mul(r, t, acc);
}

// --- Scalar arithmetic (mod n) ---

__device__ void scalar_add(U256 &r, const U256 &a, const U256 &b) {
    uint32_t carry = add256(r, a, b);
    if (carry || cmp(r, N) >= 0) {
        sub256(r, r, N);
    }
}

__device__ void scalar_mul(U256 &r, const U256 &a, const U256 &b) {
    // Full 512-bit product then reduce mod N
    uint64_t t[16] = {0};
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a.d[i] * b.d[j] + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        t[i + 8] = carry;
    }

    // Reduce 512-bit t mod n using: 2^256 ≡ R (mod n)
    // R = 2^256 - n, fits in 5 limbs (129 bits)
    const uint32_t R_LIMBS[5] = { 0x2FC9BEBF, 0x402DA173, 0x50B75FC4, 0x45512319, 0x00000001 };

    // Fold 1: hi=t[8..15] (8 limbs) * R (5 limbs) → 13 limbs, + lo=t[0..7]
    uint64_t f[13] = {0};
    for (int i = 0; i < 8; i++) {
        uint64_t c = 0;
        for (int j = 0; j < 5; j++) {
            c += f[i + j] + (uint64_t)(uint32_t)t[i + 8] * R_LIMBS[j];
            f[i + j] = c & 0xFFFFFFFF;
            c >>= 32;
        }
        f[i + 5] = c;
    }
    {
        uint64_t c = 0;
        for (int i = 0; i < 8; i++) {
            c += f[i] + (t[i] & 0xFFFFFFFF);
            f[i] = c & 0xFFFFFFFF;
            c >>= 32;
        }
        for (int i = 8; i < 13 && c; i++) {
            c += f[i];
            f[i] = c & 0xFFFFFFFF;
            c >>= 32;
        }
    }

    // Fold 2: hi=f[8..12] (5 limbs) * R (5 limbs) → 10 limbs, + lo=f[0..7]
    bool has_high = false;
    for (int i = 8; i < 13; i++) { if (f[i]) { has_high = true; break; } }
    if (has_high) {
        uint64_t g[10] = {0};
        for (int i = 0; i < 5; i++) {
            uint64_t c = 0;
            for (int j = 0; j < 5; j++) {
                c += g[i + j] + (uint64_t)(uint32_t)f[i + 8] * R_LIMBS[j];
                g[i + j] = c & 0xFFFFFFFF;
                c >>= 32;
            }
            g[i + 5] = c;
        }
        uint64_t c = 0;
        for (int i = 0; i < 8; i++) {
            c += g[i] + (f[i] & 0xFFFFFFFF);
            f[i] = c & 0xFFFFFFFF;
            c >>= 32;
        }
        // g[8..9] might be nonzero; they fold into f[8..9] with carry
        for (int i = 8; i < 10; i++) {
            c += g[i];
            f[i] = c & 0xFFFFFFFF;
            c >>= 32;
        }
        f[10] = c; f[11] = 0; f[12] = 0;

        // Fold 3 if needed: f[8..10] (3 limbs) * R + f[0..7]
        has_high = false;
        for (int i = 8; i < 13; i++) { if (f[i]) { has_high = true; break; } }
        if (has_high) {
            uint64_t h[8] = {0};
            for (int i = 0; i < 3; i++) {
                uint64_t c2 = 0;
                for (int j = 0; j < 5; j++) {
                    c2 += h[i + j] + (uint64_t)(uint32_t)f[i + 8] * R_LIMBS[j];
                    h[i + j] = c2 & 0xFFFFFFFF;
                    c2 >>= 32;
                }
                h[i + 5] = c2;
            }
            c = 0;
            for (int i = 0; i < 8; i++) {
                c += h[i] + (f[i] & 0xFFFFFFFF);
                f[i] = c & 0xFFFFFFFF;
                c >>= 32;
            }
            // At this point, any carry means we need at most a few subtractions of n
        }
    }

    // Extract final 256-bit result and reduce mod n
    for (int i = 0; i < 8; i++) r.d[i] = (uint32_t)f[i];
    while (cmp(r, N) >= 0) sub256(r, r, N);
}

// Scalar inverse via Fermat: a^(n-2) mod n
// n-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD036413F
__device__ void scalar_inv(U256 &r, const U256 &a) {
    // Simple square-and-multiply for n-2
    // n-2 in binary: lots of 1s then ...413F at the end
    // For simplicity, use generic binary method

    U256 base = a;
    U256 result;
    // Set result = 1
    result.d[0] = 1;
    for (int i = 1; i < 8; i++) result.d[i] = 0;

    // n-2 bytes (big-endian): FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FE
    //                          BA AE DC E6 AF 48 A0 3B BF D2 5E 8C D0 36 41 3F
    // Process bit by bit from MSB
    // n-2 limbs (little-endian): d[7]=0xFFFFFFFF ... d[4]=0xFFFFFFFE d[3]=0xBAAEDCE6 ...
    const uint32_t nm2[8] = {
        0xD036413F, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
        0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };

    for (int limb = 7; limb >= 0; limb--) {
        for (int bit = 31; bit >= 0; bit--) {
            // Skip the very first 1-bit (we start with result=1, base=a)
            // Actually, start from MSB. First bit is 1 (bit 255 of n-2).
            // Standard binary method: for each bit, square result, then multiply if bit=1
            if (limb == 7 && bit == 31) {
                // First bit is 1, result = a
                result = a;
                continue;
            }
            scalar_mul(result, result, result);
            if ((nm2[limb] >> bit) & 1) {
                scalar_mul(result, result, a);
            }
        }
    }

    r = result;
}

// --- EC Point operations (Jacobian coordinates) ---
// Point (X, Y, Z) represents affine (X/Z^2, Y/Z^3)
// Point at infinity: Z = 0

struct JacobianPoint {
    U256 x, y, z;
};

__device__ bool point_is_infinity(const JacobianPoint &p) {
    return is_zero(p.z);
}

// Point doubling: R = 2*P
// Using formulas from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
// (a=0 for secp256k1)
// Note: r and p may alias (r == p is safe).
__device__ void point_double(JacobianPoint &r, const JacobianPoint &p) {
    if (point_is_infinity(p)) {
        r = p;
        return;
    }

    // Save values needed after r.x/r.y are written (in case r aliases p)
    U256 py = p.y;
    U256 pz = p.z;

    U256 a, b, c, d, e, f;

    // A = X1^2
    field_sqr(a, p.x);
    // B = Y1^2
    field_sqr(b, py);
    // C = B^2
    field_sqr(c, b);

    // D = 2*((X1+B)^2 - A - C)
    U256 t;
    field_add(t, p.x, b);
    field_sqr(d, t);
    field_sub(d, d, a);
    field_sub(d, d, c);
    field_add(d, d, d); // D = 2 * d

    // E = 3*A
    field_add(e, a, a);
    field_add(e, e, a);

    // F = E^2
    field_sqr(f, e);

    // X3 = F - 2*D
    U256 d2;
    field_add(d2, d, d);
    field_sub(r.x, f, d2);

    // Y3 = E*(D - X3) - 8*C
    field_sub(t, d, r.x);
    field_mul(r.y, e, t);
    U256 c8;
    field_add(c8, c, c);   // 2C
    field_add(c8, c8, c8); // 4C
    field_add(c8, c8, c8); // 8C
    field_sub(r.y, r.y, c8);

    // Z3 = 2*Y1*Z1 (uses saved py, pz to avoid aliasing)
    field_mul(r.z, py, pz);
    field_add(r.z, r.z, r.z);
}

// Point addition: R = P + Q (P != Q, neither infinity)
// Using add-2007-bl formulas
__device__ void point_add(JacobianPoint &r, const JacobianPoint &p, const JacobianPoint &q) {
    if (point_is_infinity(p)) { r = q; return; }
    if (point_is_infinity(q)) { r = p; return; }

    U256 z1z1, z2z2, u1, u2, s1, s2, h, i, j, rr, v;

    // Z1Z1 = Z1^2
    field_sqr(z1z1, p.z);
    // Z2Z2 = Z2^2
    field_sqr(z2z2, q.z);
    // U1 = X1*Z2Z2
    field_mul(u1, p.x, z2z2);
    // U2 = X2*Z1Z1
    field_mul(u2, q.x, z1z1);
    // S1 = Y1*Z2*Z2Z2
    field_mul(s1, p.y, q.z);
    field_mul(s1, s1, z2z2);
    // S2 = Y2*Z1*Z1Z1
    field_mul(s2, q.y, p.z);
    field_mul(s2, s2, z1z1);

    // H = U2 - U1
    field_sub(h, u2, u1);

    if (is_zero(h)) {
        // U1 == U2, check if S1 == S2 (same point -> double) or S1 == -S2 (point at infinity)
        U256 sdiff;
        field_sub(sdiff, s2, s1);
        if (is_zero(sdiff)) {
            point_double(r, p);
            return;
        } else {
            // P = -Q, result is infinity
            r.x.d[0] = 0; r.y.d[0] = 1; r.z.d[0] = 0;
            for (int k = 1; k < 8; k++) { r.x.d[k] = 0; r.y.d[k] = 0; r.z.d[k] = 0; }
            return;
        }
    }

    // I = (2*H)^2
    U256 h2;
    field_add(h2, h, h);
    field_sqr(i, h2);

    // J = H*I
    field_mul(j, h, i);

    // rr = 2*(S2 - S1)
    field_sub(rr, s2, s1);
    field_add(rr, rr, rr);

    // V = U1*I
    field_mul(v, u1, i);

    // X3 = rr^2 - J - 2*V
    field_sqr(r.x, rr);
    field_sub(r.x, r.x, j);
    U256 v2;
    field_add(v2, v, v);
    field_sub(r.x, r.x, v2);

    // Y3 = rr*(V - X3) - 2*S1*J
    U256 t;
    field_sub(t, v, r.x);
    field_mul(r.y, rr, t);
    field_mul(t, s1, j);
    field_add(t, t, t);
    field_sub(r.y, r.y, t);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    field_add(t, p.z, q.z);
    field_sqr(t, t);
    field_sub(t, t, z1z1);
    field_sub(t, t, z2z2);
    field_mul(r.z, t, h);
}

// Mixed addition: R = P + Q where Q is in affine (Z=1)
// Note: r and p may alias (r == p is safe).
__device__ void point_add_affine(JacobianPoint &r, const JacobianPoint &p, const U256 &qx, const U256 &qy) {
    if (point_is_infinity(p)) {
        r.x = qx;
        r.y = qy;
        r.z.d[0] = 1;
        for (int i = 1; i < 8; i++) r.z.d[i] = 0;
        return;
    }

    // Save p.y before r.y is written (in case r aliases p)
    U256 py = p.y;

    U256 z1z1, u2, s2, h, hh, i, j, rr, v;

    field_sqr(z1z1, p.z);
    field_mul(u2, qx, z1z1);
    U256 z1_cubed;
    field_mul(z1_cubed, p.z, z1z1);
    field_mul(s2, qy, z1_cubed);

    // U1 = X1 (p.x), U2 = qx * Z1^2, S1 = Y1 (p.y), S2 = qy * Z1^3
    field_sub(h, u2, p.x);

    if (is_zero(h)) {
        U256 sdiff;
        field_sub(sdiff, s2, py);
        if (is_zero(sdiff)) {
            point_double(r, p);
            return;
        } else {
            r.x.d[0] = 0; r.y.d[0] = 1; r.z.d[0] = 0;
            for (int k = 1; k < 8; k++) { r.x.d[k] = 0; r.y.d[k] = 0; r.z.d[k] = 0; }
            return;
        }
    }

    field_sqr(hh, h);
    field_add(i, hh, hh);
    field_add(i, i, i); // I = 4*H^2

    field_mul(j, h, i);

    field_sub(rr, s2, py);
    field_add(rr, rr, rr);

    field_mul(v, p.x, i);

    field_sqr(r.x, rr);
    field_sub(r.x, r.x, j);
    U256 v2;
    field_add(v2, v, v);
    field_sub(r.x, r.x, v2);

    U256 t;
    field_sub(t, v, r.x);
    field_mul(r.y, rr, t);
    field_mul(t, py, j);      // uses saved py (r.y was already written)
    field_add(t, t, t);
    field_sub(r.y, r.y, t);

    field_add(r.z, p.z, h);   // p.z == r.z not yet written, safe
    field_sqr(r.z, r.z);
    field_sub(r.z, r.z, z1z1);
    field_sub(r.z, r.z, hh);
}

// --- Scalar multiplication: k * G using 4-bit windowed method ---

// Precomputed table of [1*G, 2*G, ..., 15*G] in affine coordinates
// Stored as pairs of (x, y) U256 values
// This will be populated at kernel launch time and stored in constant memory
struct AffinePoint {
    U256 x, y;
};

// G table — filled by init_g_table kernel at startup
// Using __device__ (not __constant__) because constant memory is read-only from device,
// and because threads in a warp access different indices (different scalars),
// which would serialize constant memory reads anyway.
__device__ AffinePoint G_TABLE[15];

// k * G using windowed method (4-bit window)
__device__ void scalar_mul_G(JacobianPoint &r, const U256 &k) {
    // Initialize result to point at infinity
    for (int i = 0; i < 8; i++) { r.x.d[i] = 0; r.y.d[i] = 0; r.z.d[i] = 0; }

    // Process 4 bits at a time from MSB
    for (int i = 63; i >= 0; i--) {
        // Double 4 times
        if (i < 63) { // Skip doubling on first iteration
            point_double(r, r);
            point_double(r, r);
            point_double(r, r);
            point_double(r, r);
        } else {
            // First window: find MSB window
        }

        // Extract 4-bit window
        int limb_idx = i / 8;     // which uint32 limb
        int bit_offset = (i % 8) * 4; // bit offset within limb
        uint32_t window = (k.d[limb_idx] >> bit_offset) & 0xF;

        if (window != 0) {
            if (point_is_infinity(r)) {
                // First nonzero window: set result directly
                r.x = G_TABLE[window - 1].x;
                r.y = G_TABLE[window - 1].y;
                r.z.d[0] = 1;
                for (int j = 1; j < 8; j++) r.z.d[j] = 0;
            } else {
                point_add_affine(r, r, G_TABLE[window - 1].x, G_TABLE[window - 1].y);
            }
        }
    }
}

// Convert Jacobian to affine
__device__ void jacobian_to_affine(U256 &ax, U256 &ay, const JacobianPoint &p) {
    U256 z_inv, z_inv2, z_inv3;
    field_inv(z_inv, p.z);
    field_sqr(z_inv2, z_inv);
    field_mul(z_inv3, z_inv2, z_inv);
    field_mul(ax, p.x, z_inv2);
    field_mul(ay, p.y, z_inv3);
}

// Get compressed public key (33 bytes): 0x02/0x03 prefix + 32 bytes x
__device__ void get_compressed_pubkey(uint8_t *out33, const JacobianPoint &pub_key) {
    U256 ax, ay;
    jacobian_to_affine(ax, ay, pub_key);

    // Prefix: 0x02 if y is even, 0x03 if y is odd
    out33[0] = (ay.d[0] & 1) ? 0x03 : 0x02;

    // X coordinate in big-endian
    for (int i = 7; i >= 0; i--) {
        int offset = 1 + (7 - i) * 4;
        out33[offset]     = (uint8_t)(ax.d[i] >> 24);
        out33[offset + 1] = (uint8_t)(ax.d[i] >> 16);
        out33[offset + 2] = (uint8_t)(ax.d[i] >> 8);
        out33[offset + 3] = (uint8_t)(ax.d[i]);
    }
}

// Load a 32-byte big-endian scalar into U256 (little-endian limbs)
__device__ void load_scalar(U256 &r, const uint8_t *bytes32) {
    for (int i = 0; i < 8; i++) {
        int offset = (7 - i) * 4;
        r.d[i] = ((uint32_t)bytes32[offset] << 24) |
                 ((uint32_t)bytes32[offset + 1] << 16) |
                 ((uint32_t)bytes32[offset + 2] << 8) |
                 ((uint32_t)bytes32[offset + 3]);
    }
}

// Store U256 as 32-byte big-endian
__device__ void store_scalar(uint8_t *bytes32, const U256 &a) {
    for (int i = 0; i < 8; i++) {
        int offset = (7 - i) * 4;
        bytes32[offset]     = (uint8_t)(a.d[i] >> 24);
        bytes32[offset + 1] = (uint8_t)(a.d[i] >> 16);
        bytes32[offset + 2] = (uint8_t)(a.d[i] >> 8);
        bytes32[offset + 3] = (uint8_t)(a.d[i]);
    }
}

} // namespace secp256k1
