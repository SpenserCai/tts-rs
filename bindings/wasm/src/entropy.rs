//! getrandom 0.3's documented custom backend for memory64 JS hosts.
//!
//! Its built-in wasm_js backend currently targets wasm32 only. Keep randomness
//! in the host's cryptographic API instead of silently using a deterministic seed.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(catch, js_namespace = ["globalThis", "crypto"], js_name = getRandomValues)]
    fn random_bytes(bytes: &js_sys::Uint8Array) -> Result<JsValue, JsValue>;
}

/// Fulfil getrandom's custom backend contract.
///
/// # Safety
/// getrandom supplies a valid, exclusive writable allocation of `length` bytes.
/// This function initializes each byte before returning success and never retains
/// the pointer. Empty requests return without constructing a pointer-backed slice.
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    destination: *mut u8,
    length: usize,
) -> Result<(), getrandom::Error> {
    if length == 0 {
        return Ok(());
    }
    // SAFETY: guaranteed by the getrandom backend contract above.
    let destination = unsafe { std::slice::from_raw_parts_mut(destination, length) };
    for chunk in destination.chunks_mut(65_536) {
        let bytes = js_sys::Uint8Array::new_with_length(chunk.len() as u32);
        random_bytes(&bytes).map_err(|_| getrandom::Error::UNSUPPORTED)?;
        bytes.copy_to(chunk);
    }
    Ok(())
}
