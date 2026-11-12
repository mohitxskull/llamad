#![no_main]

use libfuzzer_sys::fuzz_target;
use llamad::protocol::Request;

fuzz_target!(|data: &[u8]| {
    // Fuzz newline-delimited framing and size limit handling
    if data.len() > 1024 * 1024 {
        return;
    }
    for chunk in data.split(|&b| b == b'\n') {
        if let Ok(s) = std::str::from_utf8(chunk) {
            let _ = serde_json::from_str::<Request>(s);
        }
    }
});
