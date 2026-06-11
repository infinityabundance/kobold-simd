//! # kobold-simd
//!
//! SIMD-accelerated packed-decimal and zoned-decimal decode and batch record scanning (AVX2/AVX-512/NEON). // SIMD intrinsics: unsafe permitted in this crate, audited per-function.
//!
//! Part of the KOBOLD ecosystem -- independently-authored forensic tooling. Apache-2.0. This crate
//! contains no GnuCOBOL/libcob source; any interaction with COBOL semantics goes through the separate
//! gnucobol-rs crate.
//!
//! Architecture: kobold-* MAY depend on gnucobol-rs; gnucobol-rs MUST NOT depend on kobold-*.
//!
//! Status: SCAFFOLD. Implementation extracted from the gnucobol-rs lab tooling + lineage engine later.
// unsafe permitted (SIMD intrinsics); audited per-function -- NO crate-level forbid.

/// Crate scaffold marker; replace with the real public API as the implementation lands.
pub const KOBOLD_CRATE: &str = "kobold-simd";
