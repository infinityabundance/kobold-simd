# kobold-simd

Batched, branch-light decoders for the two hot COBOL numeric byte formats -- **COMP-3 packed-decimal**
and **zoned-decimal DISPLAY** -- built to decode *many* fixed-width records in one call. Legacy COBOL
estates are dominated by these two encodings; turning their byte fields into integers is the hot loop
of any migration or analytics pass, and this crate is that loop.

**Part of KOBOLD** -- independently-authored forensic tooling for legacy COBOL estates. Apache-2.0.
Standalone: no runtime dependencies, and **no GnuCOBOL source** -- the byte formats it decodes are
public, long-documented data conventions.

## The correctness contract

Each batched kernel is paired with a dead-simple scalar *reference* decoder, and the moat claim is
exactly this: **the batched kernel equals the scalar reference, lane for lane, byte for byte.** The
cross-check test generates a buffer of many records (mixed signs, scales, and deliberately corrupt
fields) and asserts each batch result equals the per-record scalar result.

## API

- `decode_packed_scalar(bytes, digits, scale) -> Option<i128>` -- scalar COMP-3 reference.
- `decode_zoned_ascii_scalar(bytes, signed) -> Option<i128>` -- scalar ASCII zoned reference.
- `decode_packed_batch(buf, record_stride, field_offset, field_len, digits, scale, out)` -- batched.
- `decode_zoned_ascii_batch(buf, record_stride, field_offset, field_len, signed, out)` -- batched.
- `LANES` -- records processed per chunk in the lane loop.

Decoders return the **unscaled integer mantissa** (`i128`); the field's logical value is
`mantissa * 10^-scale`. Decoding is **fail-closed**: a malformed field yields `None`, and in a batch
one corrupt record produces one `None` lane without disturbing its neighbours.

## SIMD shape

The batch kernels process records in fixed-size `LANES`-wide chunks with no early exit inside the lane
loop -- a constant trip count and a straight-line body, the structure an auto-vectorizer needs to
widen the inner loop. The crate ships **100% safe portable code** (`#![forbid(unsafe_code)]`).
Explicit target-feature intrinsic specializations (AVX2/NEON) are a documented future optimization
that would live behind a `cfg(target_feature = ...)` gate with this safe path as the always-correct
fallback, kept honest by the same cross-check test.

## License

Apache-2.0 (see LICENSE).
