fn main() {
    // Only Windows MSVC needs special linker search paths for llama.cpp's internal vendored libraries.
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=crypt32");

        let mut found_httplib = false;

        // 1. Check Windows MAX_PATH short_dir (%LOCALAPPDATA%\llcb, %TEMP%\llcb, %TMP%\llcb, C:\Temp\llcb)
        // llama-cpp-sys-4 creates build dirs here on Windows.
        let base_dirs = [
            std::env::var("LOCALAPPDATA")
                .ok()
                .map(std::path::PathBuf::from),
            std::env::var("TEMP").ok().map(std::path::PathBuf::from),
            std::env::var("TMP").ok().map(std::path::PathBuf::from),
            Some(std::path::PathBuf::from("C:\\Temp")),
        ];

        for base in base_dirs.into_iter().flatten() {
            let llcb = base.join("llcb");
            if llcb.exists() {
                if let Ok(entries) = std::fs::read_dir(&llcb) {
                    for entry in entries.flatten() {
                        let candidate_base = entry
                            .path()
                            .join("build")
                            .join("vendor")
                            .join("cpp-httplib");
                        for sub in &["Release", "Debug", "RelWithDebInfo", ""] {
                            let dir = if sub.is_empty() {
                                candidate_base.clone()
                            } else {
                                candidate_base.join(sub)
                            };
                            if dir.join("cpp-httplib.lib").exists() {
                                println!("cargo:rustc-link-search=native={}", dir.display());
                                found_httplib = true;
                            }
                        }
                    }
                }
            }
        }

        // 2. Check target/llama-cmake-cache and target/*/build/llama-cpp-sys-4-*/
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let target_root = std::path::PathBuf::from(manifest_dir).join("target");
            let cmake_cache = target_root.join("llama-cmake-cache");
            if cmake_cache.exists() {
                if let Ok(entries) = std::fs::read_dir(&cmake_cache) {
                    for entry in entries.flatten() {
                        let candidate_base = entry
                            .path()
                            .join("build")
                            .join("vendor")
                            .join("cpp-httplib");
                        for sub in &["Release", "Debug", "RelWithDebInfo", ""] {
                            let dir = if sub.is_empty() {
                                candidate_base.clone()
                            } else {
                                candidate_base.join(sub)
                            };
                            if dir.join("cpp-httplib.lib").exists() {
                                println!("cargo:rustc-link-search=native={}", dir.display());
                                found_httplib = true;
                            }
                        }
                    }
                }
            }
        }

        // 3. Check relative to OUT_DIR in case target dir is customized
        if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let out_path = std::path::PathBuf::from(out_dir);
            // Walk up to target/debug/build or target/release/build
            if let Some(build_dir) = out_path.parent().and_then(|p| p.parent()) {
                if let Ok(entries) = std::fs::read_dir(build_dir) {
                    for entry in entries.flatten() {
                        if entry
                            .file_name()
                            .to_string_lossy()
                            .starts_with("llama-cpp-sys-4-")
                        {
                            let candidate_base = entry
                                .path()
                                .join("out")
                                .join("build")
                                .join("vendor")
                                .join("cpp-httplib");
                            for sub in &["Release", "Debug", "RelWithDebInfo", ""] {
                                let dir = if sub.is_empty() {
                                    candidate_base.clone()
                                } else {
                                    candidate_base.join(sub)
                                };
                                if dir.join("cpp-httplib.lib").exists() {
                                    println!("cargo:rustc-link-search=native={}", dir.display());
                                    found_httplib = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        if found_httplib {
            println!("cargo:rustc-link-lib=static=cpp-httplib");
        }

        // 4. OpenSSL search paths (in case cpp-httplib or llama.cpp linked against OpenSSL)
        let openssl_roots = [
            std::env::var("OPENSSL_ROOT_DIR")
                .ok()
                .map(std::path::PathBuf::from),
            std::env::var("OPENSSL_DIR")
                .ok()
                .map(std::path::PathBuf::from),
            Some(std::path::PathBuf::from("C:\\Program Files\\OpenSSL")),
            Some(std::path::PathBuf::from("C:\\Program Files\\OpenSSL-Win64")),
            Some(std::path::PathBuf::from(
                "C:\\Program Files (x86)\\OpenSSL-Win32",
            )),
        ];

        let mut linked_ssl = false;
        for root in openssl_roots.into_iter().flatten() {
            if root.exists() {
                let subdirs = [
                    root.join("lib").join("VC").join("x64").join("MD"),
                    root.join("lib").join("VC").join("x64").join("MT"),
                    root.join("lib").join("VC"),
                    root.join("lib"),
                    root.clone(),
                ];
                for dir in subdirs {
                    let has_libssl = dir.join("libssl.lib").exists();
                    let has_ssl = dir.join("ssl.lib").exists();
                    let has_libcrypto = dir.join("libcrypto.lib").exists();
                    let has_crypto = dir.join("crypto.lib").exists();

                    if (has_libssl || has_ssl) && (has_libcrypto || has_crypto) {
                        println!("cargo:rustc-link-search=native={}", dir.display());
                        if has_libssl {
                            println!("cargo:rustc-link-lib=libssl");
                            println!("cargo:rustc-link-lib=libcrypto");
                        } else {
                            println!("cargo:rustc-link-lib=ssl");
                            println!("cargo:rustc-link-lib=crypto");
                        }
                        linked_ssl = true;
                        break;
                    }
                }
                if linked_ssl {
                    break;
                }
            }
        }
    }
}
