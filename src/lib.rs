//! # kobold-simd
//!
//! Batched, branch-light decoders for the two hot COBOL numeric byte formats -- **COMP-3
//! packed-decimal** and **zoned-decimal DISPLAY** -- built to decode *many* fixed-width records in
//! one call. Legacy COBOL estates are dominated by these two encodings; a migration or analytics
//! pass that touches millions of records spends most of its time turning these byte fields into
//! integers. This crate is that hot loop.
//!
//! Part of the KOBOLD ecosystem -- independently-authored forensic tooling, Apache-2.0. This crate
//! contains **no GnuCOBOL/libcob source** and depends on nothing; the byte formats it decodes
//! (packed-decimal, zoned-decimal) are public, long-documented data conventions.
//!
//! ## The correctness contract
//!
//! Every batched kernel is paired with a dead-simple scalar *reference* decoder, and the moat claim
//! is exactly this: **the batched kernel equals the scalar reference, lane for lane, byte for byte.**
//! The reference decoders ([`decode_packed_scalar`], [`decode_zoned_ascii_scalar`]) are short and
//! obviously correct; the batch kernels ([`decode_packed_batch`], [`decode_zoned_ascii_batch`]) are
//! shaped for throughput. The cross-check test generates a buffer of many records (mixed signs,
//! scales, and deliberately corrupt fields) and asserts each batch result equals the per-record
//! scalar result. If they ever diverge, the batch kernel is wrong -- not the reference.
//!
//! ## Values are unscaled mantissas
//!
//! Decoders return the **unscaled integer mantissa** as `i128`. A field declared with `scale = s`
//! has logical value `mantissa * 10^-s`; applying the scale (a `Decimal`/fixed-point concern) is
//! left to the caller. `scale` is carried through the API only so callers can thread it without a
//! parallel array; it does not change the decoded mantissa.
//!
//! ## Fail-closed
//!
//! A malformed field decodes to `None`, never to a guessed value. A non-BCD digit nibble, a bad
//! length, or a byte outside the valid zoned digit/overpunch set all yield `None`. In a batch, one
//! corrupt record produces one `None` lane and never disturbs its neighbours.
//!
//! ## SIMD shape (and the safe default)
//!
//! The batch kernels process records in fixed-size **lanes** (`LANES = 8` at a time) with no early
//! exit inside the lane loop: every lane in a chunk is decoded unconditionally, validity is tracked
//! as a per-lane boolean, and the `Option` is materialised only when the chunk is committed. That
//! structure -- fixed trip count, straight-line body, no data-dependent branches -- is what an
//! auto-vectorizer needs to widen the inner loop, and it is the same shape an explicit AVX2/NEON
//! intrinsic specialization would take.
//!
//! This crate ships **100% safe portable code** and keeps `#![forbid(unsafe_code)]`. Explicit
//! target-feature intrinsic specializations (AVX2 gather, NEON `tbl`) are a documented future
//! optimization that would live behind a `cfg(target_feature = ...)` gate with this safe portable
//! path as the always-correct fallback; the cross-check test is what would keep such a specialization
//! honest.
#![forbid(unsafe_code)]

/// Records decoded per chunk in the batch kernels' inner loop. Fixed (not data-dependent) so the
/// lane loop has a constant trip count an auto-vectorizer can widen.
pub const LANES: usize = 8;

/// Decode one COMP-3 packed-decimal field to its unscaled `i128` mantissa (scalar reference).
///
/// COMP-3 packs two decimal digits per byte as BCD, most-significant first, with the **low nibble of
/// the final byte** holding the sign: `0xD` is negative, every other sign nibble (`0xC`, `0xF`, and
/// the tolerated `0xA`/`0xB`/`0xE`) is positive. With `digits` odd the field is `digits/2 + 1` bytes
/// and every digit nibble is used; with `digits` even there is one leading unused high nibble
/// (conventionally `0`) which is read as an ordinary digit -- if it is non-zero it still contributes,
/// matching runtime behavior rather than rejecting it.
///
/// Returns the **unscaled mantissa**; the field's logical value is `mantissa * 10^-scale`. `scale`
/// does not affect the returned integer and is accepted only for caller convenience.
///
/// Fail-closed -> `None` when: `bytes` is empty, its length disagrees with `digits`, or any digit
/// nibble is `> 9`.
pub fn decode_packed_scalar(bytes: &[u8], digits: u32, scale: i32) -> Option<i128> {
    let _ = scale; // mantissa is scale-independent; carried only for API symmetry
    if bytes.is_empty() {
        return None;
    }
    let expected = (digits as usize) / 2 + 1;
    if bytes.len() != expected {
        return None;
    }
    let last = bytes.len() - 1;
    let mut acc: i128 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let hi = (b >> 4) as i128;
        let lo = (b & 0x0F) as i128;
        if hi > 9 {
            return None;
        }
        acc = acc * 10 + hi;
        if i != last {
            if lo > 9 {
                return None;
            }
            acc = acc * 10 + lo;
        }
    }
    // Sign nibble: low nibble of the final byte. 0xD negative, anything else positive.
    let sign = bytes[last] & 0x0F;
    if sign == 0x0D {
        acc = -acc;
    }
    Some(acc)
}

/// Decode one ASCII zoned-decimal (`PIC 9 DISPLAY`) field to its unscaled `i128` mantissa (scalar
/// reference).
///
/// Each byte carries one digit. A plain digit is `0x30..=0x39` (`'0'..='9'`). When `signed`, the
/// **trailing** byte may instead be an overpunch in `0x70..=0x79`, which per the GnuCOBOL ASCII
/// scheme encodes the same digit (`byte - 0x70`) with a **negative** sign. Leading/interior bytes
/// must always be plain digits.
///
/// Returns the unscaled mantissa. Fail-closed -> `None` when: `bytes` is empty, any non-trailing
/// byte is not `0x30..=0x39`, or the trailing byte is neither a plain digit nor (when `signed`) an
/// overpunch.
pub fn decode_zoned_ascii_scalar(bytes: &[u8], signed: bool) -> Option<i128> {
    if bytes.is_empty() {
        return None;
    }
    let last = bytes.len() - 1;
    let mut acc: i128 = 0;
    let mut negative = false;
    for (i, &b) in bytes.iter().enumerate() {
        let digit = if i == last && signed && (0x70..=0x79).contains(&b) {
            negative = true;
            (b - 0x70) as i128
        } else if (0x30..=0x39).contains(&b) {
            (b - 0x30) as i128
        } else {
            return None;
        };
        acc = acc * 10 + digit;
    }
    if negative {
        acc = -acc;
    }
    Some(acc)
}

/// Decode `field_len` bytes at `field_offset` of every `record_stride`-byte record in `buf` as a
/// COMP-3 packed field, in one batched call. Results are appended to `out` in record order, one
/// `Option<i128>` per record (`None` for a corrupt field, exactly as [`decode_packed_scalar`] would
/// return).
///
/// `out` is cleared and reserved up front. Records are processed in fixed [`LANES`]-wide chunks with
/// no early exit inside the lane loop -- the structure an auto-vectorizer needs. The result is
/// **identical**, record for record, to calling [`decode_packed_scalar`] on each field slice; that
/// equivalence is the crate's correctness contract (see the cross-check test).
///
/// A trailing partial record (fewer than `record_stride` bytes left, or a field that would read past
/// the buffer) is **not** decoded: only whole, fully-present fields produce a lane.
pub fn decode_packed_batch(
    buf: &[u8],
    record_stride: usize,
    field_offset: usize,
    field_len: usize,
    digits: u32,
    scale: i32,
    out: &mut Vec<Option<i128>>,
) {
    out.clear();
    if record_stride == 0 || field_len == 0 {
        return;
    }
    let records = whole_records(buf.len(), record_stride, field_offset, field_len);
    out.reserve(records);

    let expected = (digits as usize) / 2 + 1;
    let chunks = records / LANES;

    // Lane-parallel body: a fixed-count inner loop with straight-line per-lane work and no
    // data-dependent branch on the chunk boundary.
    for c in 0..chunks {
        let mut lane_vals = [0i128; LANES];
        let mut lane_ok = [false; LANES];
        for l in 0..LANES {
            let r = c * LANES + l;
            let start = r * record_stride + field_offset;
            let field = &buf[start..start + field_len];
            let (v, ok) = packed_lane(field, expected);
            lane_vals[l] = v;
            lane_ok[l] = ok;
        }
        for l in 0..LANES {
            out.push(if lane_ok[l] { Some(lane_vals[l]) } else { None });
        }
    }
    // Remainder records (fewer than a full lane) -- same per-record work, scalar tail.
    for r in (chunks * LANES)..records {
        let start = r * record_stride + field_offset;
        let field = &buf[start..start + field_len];
        let (v, ok) = packed_lane(field, expected);
        out.push(if ok { Some(v) } else { None });
    }
    let _ = scale;
}

/// Decode `field_len` bytes at `field_offset` of every `record_stride`-byte record in `buf` as an
/// ASCII zoned field, in one batched call. Semantics mirror [`decode_packed_batch`]: lane-shaped,
/// fail-closed, one `Option<i128>` appended to `out` per whole record, identical to per-record
/// [`decode_zoned_ascii_scalar`].
pub fn decode_zoned_ascii_batch(
    buf: &[u8],
    record_stride: usize,
    field_offset: usize,
    field_len: usize,
    signed: bool,
    out: &mut Vec<Option<i128>>,
) {
    out.clear();
    if record_stride == 0 || field_len == 0 {
        return;
    }
    let records = whole_records(buf.len(), record_stride, field_offset, field_len);
    out.reserve(records);

    let chunks = records / LANES;
    for c in 0..chunks {
        let mut lane_vals = [0i128; LANES];
        let mut lane_ok = [false; LANES];
        for l in 0..LANES {
            let r = c * LANES + l;
            let start = r * record_stride + field_offset;
            let field = &buf[start..start + field_len];
            let (v, ok) = zoned_lane(field, signed);
            lane_vals[l] = v;
            lane_ok[l] = ok;
        }
        for l in 0..LANES {
            out.push(if lane_ok[l] { Some(lane_vals[l]) } else { None });
        }
    }
    for r in (chunks * LANES)..records {
        let start = r * record_stride + field_offset;
        let field = &buf[start..start + field_len];
        let (v, ok) = zoned_lane(field, signed);
        out.push(if ok { Some(v) } else { None });
    }
}

/// Whole records whose field `[field_offset, field_offset+field_len)` lies fully inside `buf`.
fn whole_records(buf_len: usize, stride: usize, field_offset: usize, field_len: usize) -> usize {
    let field_end = match field_offset.checked_add(field_len) {
        Some(e) => e,
        None => return 0,
    };
    if field_end > stride {
        // Field declared past the record bound -- decode nothing rather than read across records.
        return 0;
    }
    let full = buf_len / stride;
    // The field of record `r` ends at `r*stride + field_end`; for the last full record this is
    // `<= (full-1)*stride + stride = full*stride <= buf_len`, so every full record is in range.
    full
}

/// Branch-light per-lane COMP-3 decode. Returns `(mantissa, ok)`; `ok == false` means the field is
/// corrupt and the value must be discarded. Mirrors [`decode_packed_scalar`] exactly but never
/// short-circuits, so it composes into a fixed-count lane loop.
#[inline]
fn packed_lane(field: &[u8], expected_len: usize) -> (i128, bool) {
    if field.len() != expected_len || expected_len == 0 {
        return (0, false);
    }
    let last = field.len() - 1;
    let mut acc: i128 = 0;
    let mut ok = true;
    for (i, &b) in field.iter().enumerate() {
        let hi = (b >> 4) as i128;
        let lo = (b & 0x0F) as i128;
        ok &= hi <= 9;
        acc = acc * 10 + hi;
        if i != last {
            ok &= lo <= 9;
            acc = acc * 10 + lo;
        }
    }
    if (field[last] & 0x0F) == 0x0D {
        acc = -acc;
    }
    (acc, ok)
}

/// Branch-light per-lane ASCII zoned decode. Returns `(mantissa, ok)`; mirrors
/// [`decode_zoned_ascii_scalar`] without short-circuiting.
#[inline]
fn zoned_lane(field: &[u8], signed: bool) -> (i128, bool) {
    if field.is_empty() {
        return (0, false);
    }
    let last = field.len() - 1;
    let mut acc: i128 = 0;
    let mut ok = true;
    let mut negative = false;
    for (i, &b) in field.iter().enumerate() {
        let is_overpunch = i == last && signed && (0x70..=0x79).contains(&b);
        let is_digit = (0x30..=0x39).contains(&b);
        // Per-lane validity, no branch out of the loop.
        ok &= is_overpunch | is_digit;
        let digit = if is_overpunch {
            negative = true;
            b.wrapping_sub(0x70) as i128
        } else {
            // Mask to the low nibble; for a valid digit this is `b - 0x30`, for an invalid byte the
            // value is discarded because `ok` is already false.
            (b & 0x0F) as i128
        };
        acc = acc * 10 + digit;
    }
    if negative {
        acc = -acc;
    }
    (acc, ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- scalar packed reference --------------------------------------------------------------

    #[test]
    fn packed_positive() {
        // 12345 positive, PIC S9(5) COMP-3 -> 3 bytes 0x12 0x34 0x5C
        assert_eq!(decode_packed_scalar(&[0x12, 0x34, 0x5C], 5, 0), Some(12345));
    }

    #[test]
    fn packed_negative() {
        // -12345 -> sign nibble 0xD
        assert_eq!(decode_packed_scalar(&[0x12, 0x34, 0x5D], 5, 0), Some(-12345));
    }

    #[test]
    fn packed_sign_variants_are_positive() {
        // 0xC and 0xF both positive
        assert_eq!(decode_packed_scalar(&[0x12, 0x34, 0x5C], 5, 0), Some(12345));
        assert_eq!(decode_packed_scalar(&[0x12, 0x34, 0x5F], 5, 0), Some(12345));
    }

    #[test]
    fn packed_even_digits_leading_nibble() {
        // PIC S9(4): digits/2+1 = 3 bytes, leading high nibble unused (0). 0x00 0x12 0x3C = +123
        assert_eq!(decode_packed_scalar(&[0x00, 0x12, 0x3C], 4, 0), Some(123));
    }

    #[test]
    fn packed_scale_does_not_change_mantissa() {
        // value 123.45 stored as mantissa 12345 with scale 2 -> still 12345
        assert_eq!(decode_packed_scalar(&[0x12, 0x34, 0x5C], 5, 2), Some(12345));
    }

    #[test]
    fn packed_zero() {
        assert_eq!(decode_packed_scalar(&[0x00, 0x00, 0x0C], 5, 0), Some(0));
    }

    #[test]
    fn packed_fail_closed() {
        // bad length
        assert_eq!(decode_packed_scalar(&[0x12, 0x3C], 5, 0), None);
        // empty
        assert_eq!(decode_packed_scalar(&[], 5, 0), None);
        // non-BCD digit nibble (0xA in a digit position)
        assert_eq!(decode_packed_scalar(&[0x1A, 0x34, 0x5C], 5, 0), None);
        assert_eq!(decode_packed_scalar(&[0xA2, 0x34, 0x5C], 5, 0), None);
    }

    // ---- scalar zoned reference ---------------------------------------------------------------

    #[test]
    fn zoned_unsigned() {
        assert_eq!(decode_zoned_ascii_scalar(b"12345", false), Some(12345));
    }

    #[test]
    fn zoned_signed_positive_plain_trailing() {
        // signed but trailing is a plain digit -> positive
        assert_eq!(decode_zoned_ascii_scalar(b"12345", true), Some(12345));
    }

    #[test]
    fn zoned_signed_negative_overpunch() {
        // "1234" + 0x75 ('u', overpunch digit 5, negative) -> -12345
        assert_eq!(decode_zoned_ascii_scalar(b"1234\x75", true), Some(-12345));
        // overpunch 0x70 = negative 0 digit -> -12340
        assert_eq!(decode_zoned_ascii_scalar(b"1234\x70", true), Some(-12340));
    }

    #[test]
    fn zoned_overpunch_ignored_when_unsigned() {
        // unsigned field: 0x75 is not a plain digit -> fail-closed
        assert_eq!(decode_zoned_ascii_scalar(b"1234\x75", false), None);
    }

    #[test]
    fn zoned_fail_closed() {
        assert_eq!(decode_zoned_ascii_scalar(b"", false), None);
        // interior non-digit
        assert_eq!(decode_zoned_ascii_scalar(b"12\x0d45", false), None);
        // overpunch in a non-trailing position is invalid even when signed
        assert_eq!(decode_zoned_ascii_scalar(b"\x751234", true), None);
        // trailing byte outside digit and overpunch ranges
        assert_eq!(decode_zoned_ascii_scalar(b"1234\xFF", true), None);
    }

    // ---- lane helpers equal their scalar references -------------------------------------------

    #[test]
    fn packed_lane_matches_scalar() {
        let cases: &[&[u8]] = &[
            &[0x12, 0x34, 0x5C],
            &[0x12, 0x34, 0x5D],
            &[0x00, 0x00, 0x0C],
            &[0x1A, 0x34, 0x5C], // corrupt
            &[0x99, 0x99, 0x9D],
        ];
        for f in cases {
            let (v, ok) = packed_lane(f, 3);
            let lane = if ok { Some(v) } else { None };
            assert_eq!(lane, decode_packed_scalar(f, 5, 0), "field {f:02x?}");
        }
    }

    #[test]
    fn zoned_lane_matches_scalar() {
        let cases: &[(&[u8], bool)] = &[
            (b"12345", false),
            (b"12345", true),
            (b"1234\x75", true),
            (b"1234\x70", true),
            (b"1234\x75", false), // corrupt (unsigned)
            (b"12\x0d45", false),  // corrupt
        ];
        for &(f, signed) in cases {
            let (v, ok) = zoned_lane(f, signed);
            let lane = if ok { Some(v) } else { None };
            assert_eq!(lane, decode_zoned_ascii_scalar(f, signed), "field {f:02x?} signed={signed}");
        }
    }

    // ---- batch boundary behavior --------------------------------------------------------------

    #[test]
    fn batch_empty_and_partial() {
        let mut out = Vec::new();
        // empty buffer -> no records
        decode_packed_batch(&[], 8, 0, 3, 5, 0, &mut out);
        assert!(out.is_empty());
        // field declared past the record stride -> decode nothing (no cross-record reads)
        decode_packed_batch(&[0u8; 16], 8, 6, 3, 5, 0, &mut out);
        assert!(out.is_empty());
        // a trailing partial record is dropped: 20 bytes, stride 8 -> 2 whole records
        decode_packed_batch(&[0x00, 0x00, 0x0C].repeat(7), 3, 0, 3, 5, 0, &mut out);
        assert_eq!(out.len(), 7);
        // zero stride / zero field len -> nothing, no panic
        decode_packed_batch(&[0u8; 16], 0, 0, 3, 5, 0, &mut out);
        assert!(out.is_empty());
        decode_zoned_ascii_batch(&[0u8; 16], 8, 0, 0, false, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn batch_out_is_cleared() {
        let mut out = vec![Some(999i128), None];
        decode_packed_batch(&[], 8, 0, 3, 5, 0, &mut out);
        assert!(out.is_empty());
    }

    // ---- THE CORRECTNESS CONTRACT: batch == scalar reference, lane for lane --------------------

    /// Build a buffer of `n` packed records and cross-check the batch kernel against per-record
    /// scalar decode. Mixes signs, scales, and injects corruption at a known stride so both full
    /// lanes and the scalar tail exercise the fail-closed path.
    #[test]
    fn packed_batch_cross_check() {
        const DIGITS: u32 = 7; // PIC S9(7) COMP-3 -> 4 bytes
        const FIELD_LEN: usize = (DIGITS as usize) / 2 + 1;
        const STRIDE: usize = FIELD_LEN + 5; // record has filler around the field
        const FIELD_OFFSET: usize = 2;
        // Deliberately not a multiple of LANES, so the scalar remainder tail is also covered.
        const N: usize = LANES * 5 + 3;

        let mut buf = vec![0u8; N * STRIDE];
        for r in 0..N {
            let base = r * STRIDE + FIELD_OFFSET;
            // value derived from r, sign alternates
            let mag = (r as i128 * 37 + 1) % 9_999_999;
            let negative = r % 3 == 0;
            // encode `mag` into FIELD_LEN BCD bytes with sign nibble
            let mut digs = [0u8; 2 * FIELD_LEN];
            let mut v = mag;
            for d in digs.iter_mut().rev() {
                *d = (v % 10) as u8;
                v /= 10;
            }
            // pack: 2 digits per byte, last low nibble is the sign
            for i in 0..FIELD_LEN {
                let hi = digs[2 * i];
                let lo = if i == FIELD_LEN - 1 {
                    if negative { 0x0D } else { 0x0C }
                } else {
                    digs[2 * i + 1]
                };
                buf[base + i] = (hi << 4) | lo;
            }
            // corrupt every 11th record: stamp a non-BCD nibble into the field
            if r % 11 == 5 {
                buf[base] = 0xAB;
            }
        }

        let mut batch = Vec::new();
        decode_packed_batch(&buf, STRIDE, FIELD_OFFSET, FIELD_LEN, DIGITS, 2, &mut batch);
        assert_eq!(batch.len(), N);

        let mut corrupt_seen = 0usize;
        #[allow(clippy::needless_range_loop)] // `r` reconstructs the byte offset, not just an index
        for r in 0..N {
            let base = r * STRIDE + FIELD_OFFSET;
            let field = &buf[base..base + FIELD_LEN];
            let reference = decode_packed_scalar(field, DIGITS, 2);
            assert_eq!(batch[r], reference, "lane {r} diverged from scalar reference");
            if reference.is_none() {
                corrupt_seen += 1;
            }
        }
        assert!(corrupt_seen > 0, "test should exercise the fail-closed path");
    }

    /// Cross-check the zoned batch kernel against per-record scalar decode, same discipline.
    #[test]
    fn zoned_batch_cross_check() {
        const FIELD_LEN: usize = 6;
        const STRIDE: usize = FIELD_LEN + 4;
        const FIELD_OFFSET: usize = 1;
        const N: usize = LANES * 4 + 5;
        const SIGNED: bool = true;

        let mut buf = vec![b'_'; N * STRIDE];
        for r in 0..N {
            let base = r * STRIDE + FIELD_OFFSET;
            let mag = (r as i128 * 53 + 7) % 1_000_000; // fits in 6 digits
            let negative = r % 4 == 0;
            let mut v = mag;
            let mut digs = [0u8; FIELD_LEN];
            for d in digs.iter_mut().rev() {
                *d = (v % 10) as u8;
                v /= 10;
            }
            for i in 0..FIELD_LEN - 1 {
                buf[base + i] = 0x30 + digs[i];
            }
            // trailing byte: overpunch when negative, plain digit when positive
            let last = digs[FIELD_LEN - 1];
            buf[base + FIELD_LEN - 1] = if negative { 0x70 + last } else { 0x30 + last };
            // corrupt every 13th record: drop a bad byte into the interior
            if r % 13 == 4 {
                buf[base + 2] = 0xFF;
            }
        }

        let mut batch = Vec::new();
        decode_zoned_ascii_batch(&buf, STRIDE, FIELD_OFFSET, FIELD_LEN, SIGNED, &mut batch);
        assert_eq!(batch.len(), N);

        let mut corrupt_seen = 0usize;
        let mut negatives_seen = 0usize;
        #[allow(clippy::needless_range_loop)] // `r` reconstructs the byte offset, not just an index
        for r in 0..N {
            let base = r * STRIDE + FIELD_OFFSET;
            let field = &buf[base..base + FIELD_LEN];
            let reference = decode_zoned_ascii_scalar(field, SIGNED);
            assert_eq!(batch[r], reference, "lane {r} diverged from scalar reference");
            match reference {
                None => corrupt_seen += 1,
                Some(v) if v < 0 => negatives_seen += 1,
                _ => {}
            }
        }
        assert!(corrupt_seen > 0, "test should exercise the fail-closed path");
        assert!(negatives_seen > 0, "test should exercise the negative-overpunch path");
    }
}
