// Vendored from oritwoen/kangaroo (MIT License)
// https://github.com/oritwoen/kangaroo
// secp256k1 field arithmetic — 8 x u32 limbs, pure u32 (no uint64)

// =============================================================================
// secp256k1 Elliptic Curve Operations (Jacobian Coordinates)
// =============================================================================
// Curve: y² = x³ + 7 over F_p
// p = 2^256 - 2^32 - 977
//
// Jacobian coordinates: (X, Y, Z) represents affine point (X/Z², Y/Z³)
// Advantage: No modular inverse needed during point operations!
// Only one inverse needed at the very end to convert back to affine.

// Affine point structure
struct AffinePoint {
    x: array<u32, 8>,
    y: array<u32, 8>
}

// Jacobian point structure
struct JacobianPoint {
    x: array<u32, 8>,
    y: array<u32, 8>,
    z: array<u32, 8>
}

// -----------------------------------------------------------------------------
// Field element constants
// -----------------------------------------------------------------------------

fn fe_zero() -> array<u32, 8> {
    return array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn fe_one() -> array<u32, 8> {
    return array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

// Check if field element is zero
fn fe_is_zero(a: array<u32, 8>) -> bool {
    return a[0] == 0u && a[1] == 0u && a[2] == 0u && a[3] == 0u &&
           a[4] == 0u && a[5] == 0u && a[6] == 0u && a[7] == 0u;
}

// -----------------------------------------------------------------------------
// Point at infinity check (Z = 0 in Jacobian)
// -----------------------------------------------------------------------------

fn jac_is_infinity(p: JacobianPoint) -> bool {
    return fe_is_zero(p.z);
}

fn jac_infinity() -> JacobianPoint {
    var p: JacobianPoint;
    p.x = fe_one();  // Convention: (1, 1, 0) for infinity
    p.y = fe_one();
    p.z = fe_zero();
    return p;
}

// -----------------------------------------------------------------------------
// Point doubling: R = 2*P (Jacobian coordinates)
// Cost: 4M + 4S (M = multiplication, S = squaring)
// Formula from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
// For a = 0 (secp256k1): y² = x³ + 7
// -----------------------------------------------------------------------------

fn jac_double(p: JacobianPoint) -> JacobianPoint {
    // OPTIMIZATION: Skip infinity/edge case checks
    // In Kangaroo algorithm, doubling is only called in edge cases
    // which are astronomically unlikely in practice

    // A = X1²
    let a = fe_square(p.x);

    // B = Y1²
    let b = fe_square(p.y);

    // C = B² = Y1⁴
    let c = fe_square(b);

    // D = 2*((X1+B)² - A - C)
    let xpb = fe_add(p.x, b);
    let xpb2 = fe_square(xpb);
    let d_inner = fe_sub(fe_sub(xpb2, a), c);
    let d = fe_double(d_inner);

    // E = 3*A (since a = 0 for secp256k1)
    let e = fe_add(fe_add(a, a), a);

    // F = E²
    let f = fe_square(e);

    // X3 = F - 2*D
    let x3 = fe_sub(f, fe_double(d));

    // Y3 = E*(D - X3) - 8*C
    let d_minus_x3 = fe_sub(d, x3);
    let e_times_diff = fe_mul(e, d_minus_x3);
    let eight_c = fe_double(fe_double(fe_double(c)));
    let y3 = fe_sub(e_times_diff, eight_c);

    // Z3 = 2*Y1*Z1
    let y1z1 = fe_mul(p.y, p.z);
    let z3 = fe_double(y1z1);

    var result: JacobianPoint;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// -----------------------------------------------------------------------------
// Mixed addition: R = P + Q where P is Jacobian and Q is affine
// Handles P == Q case (doubling) and P == -Q case (infinity)
// Cost: 8M + 4S (or 4M + 4S for doubling)
// -----------------------------------------------------------------------------

fn jac_add_affine(
    p: JacobianPoint,
    qx: array<u32, 8>,
    qy: array<u32, 8>
) -> JacobianPoint {
    // Z1² and Z1³
    let z1z1 = fe_square(p.z);         // Z1²
    let z1z1z1 = fe_mul(z1z1, p.z);    // Z1³

    // U2 = X2 * Z1² (X2 is qx since Q is affine)
    let u2 = fe_mul(qx, z1z1);

    // S2 = Y2 * Z1³ (Y2 is qy since Q is affine)
    let s2 = fe_mul(qy, z1z1z1);

    // H = U2 - X1
    let h = fe_sub(u2, p.x);

    // Check if H == 0 (same x-coordinate)
    if (fe_is_zero(h)) {
        // Either P == Q (doubling) or P == -Q (infinity)
        let s_diff = fe_sub(s2, p.y);
        if (fe_is_zero(s_diff)) {
            // P == Q: need to double
            return jac_double(p);
        } else {
            // P == -Q: result is infinity
            return jac_infinity();
        }
    }

    // r = 2 * (S2 - Y1)
    let r = fe_double(fe_sub(s2, p.y));

    // HH = H²
    let hh = fe_square(h);

    // I = 4 * HH
    let i = fe_double(fe_double(hh));

    // J = H * I
    let j = fe_mul(h, i);

    // V = X1 * I
    let v = fe_mul(p.x, i);

    // X3 = r² - J - 2*V
    let r2 = fe_square(r);
    let x3 = fe_sub(fe_sub(r2, j), fe_double(v));

    // Y3 = r * (V - X3) - 2 * Y1 * J
    let v_minus_x3 = fe_sub(v, x3);
    let r_times_diff = fe_mul(r, v_minus_x3);
    let y1j = fe_mul(p.y, j);
    let y3 = fe_sub(r_times_diff, fe_double(y1j));

    // Z3 = (Z1 + H)² - Z1² - H²
    let z1_plus_h = fe_add(p.z, h);
    let z1_plus_h_sq = fe_square(z1_plus_h);
    let z3 = fe_sub(fe_sub(z1_plus_h_sq, z1z1), hh);

    var result: JacobianPoint;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// Convert Jacobian (X, Y, Z) to affine (X/Z², Y/Z³)
fn jac_to_affine(p: JacobianPoint) -> AffinePoint {
    let z_inv = fe_inv(p.z);
    let z_inv2 = fe_square(z_inv);
    let z_inv3 = fe_mul(z_inv2, z_inv);
    var result: AffinePoint;
    result.x = fe_mul(p.x, z_inv2);
    result.y = fe_mul(p.y, z_inv3);
    return result;
}

// 4-bit windowed scalar multiplication: scalar * G using precomputed table
// g_table_x/g_table_y: 15 affine points, each 8 u32 limbs = 120 u32s total per coordinate
fn scalar_mul_g_windowed(
    scalar: array<u32, 8>,
    g_table_x: ptr<storage, array<u32>, read>,
    g_table_y: ptr<storage, array<u32>, read>
) -> JacobianPoint {
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
                qx[j] = (*g_table_x)[table_offset + j];
                qy[j] = (*g_table_y)[table_offset + j];
            }
            result = jac_add_affine(result, qx, qy);
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// 256-bit scalar addition (for distance tracking)
// NOTE: Unrolled manually to avoid dynamic indexing (crashes RADV)
// -----------------------------------------------------------------------------

fn scalar_add_256(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var c: array<u32, 8>;
    var carry: u32 = 0u;
    var sum: u32;

    sum = a[0] + b[0] + carry; carry = select(0u, 1u, sum < a[0] || (carry == 1u && sum <= a[0])); c[0] = sum;
    sum = a[1] + b[1] + carry; carry = select(0u, 1u, sum < a[1] || (carry == 1u && sum <= a[1])); c[1] = sum;
    sum = a[2] + b[2] + carry; carry = select(0u, 1u, sum < a[2] || (carry == 1u && sum <= a[2])); c[2] = sum;
    sum = a[3] + b[3] + carry; carry = select(0u, 1u, sum < a[3] || (carry == 1u && sum <= a[3])); c[3] = sum;
    sum = a[4] + b[4] + carry; carry = select(0u, 1u, sum < a[4] || (carry == 1u && sum <= a[4])); c[4] = sum;
    sum = a[5] + b[5] + carry; carry = select(0u, 1u, sum < a[5] || (carry == 1u && sum <= a[5])); c[5] = sum;
    sum = a[6] + b[6] + carry; carry = select(0u, 1u, sum < a[6] || (carry == 1u && sum <= a[6])); c[6] = sum;
    sum = a[7] + b[7] + carry; c[7] = sum;

    return c;
}

fn scalar_sub_256(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var c: array<u32, 8>;
    var borrow: u32 = 0u;
    for (var i = 0u; i < 8u; i++) {
        let diff = a[i] - b[i] - borrow;
        borrow = select(0u, 1u, a[i] < b[i] + borrow || (borrow == 1u && b[i] == 0xFFFFFFFFu));
        c[i] = diff;
    }
    return c;
}

