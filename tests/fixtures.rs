//! Keeps the test suite's fixture list and `models/download.sh` in agreement.
//!
//! These two have to name the same files, and nothing connected them: the
//! filenames were duplicated across the download script, `tests/common`, and
//! (in one case) an individual test binary. Drift showed up as a confusing
//! runtime failure — a "file not found" model load, or worse, a silently
//! skipped test that still reported as a pass.
//!
//! The checks run in both directions on purpose. Script-to-fixtures catches a
//! renamed or re-quantized model; fixtures-to-script catches a download that
//! nothing uses, which is how ~470 MB of `qwen2.5-0.5b` sat in the fixture set
//! and in every CI cache without a single test referencing it.
//!
//! Needs no GGUF of its own — it reads the script as text, so it runs in the
//! ordinary unit-test job rather than the real-model one.

mod common;
use common::ALL;

/// Filenames `download.sh` will fetch, parsed from its `MODELS` map.
///
/// Deliberately a dumb line parser rather than anything clever: if the script's
/// shape changes enough to break this, the assertion below fails loudly and
/// someone looks at both files, which is the outcome we want anyway.
fn declared_in_download_script() -> Vec<String> {
    let script = include_str!("../models/download.sh");
    script
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("[\""))
        .filter_map(|line| {
            let rest = line.strip_prefix("[\"")?;
            let (name, _) = rest.split_once("\"]")?;
            Some(name.to_owned())
        })
        .collect()
}

#[test]
fn download_script_is_parseable_at_all() {
    // Guards the two tests below from passing vacuously: if the parser stops
    // matching the script's syntax it returns an empty list, and "every
    // fixture is in the empty set" would otherwise fail in a way that looks
    // like a missing model rather than a broken parser.
    let declared = declared_in_download_script();
    assert!(
        !declared.is_empty(),
        "parsed no model names out of models/download.sh — the MODELS map syntax probably changed, \
         so the drift checks below would be meaningless"
    );
    assert!(
        declared.iter().all(|n| n.ends_with(".gguf")),
        "parsed non-GGUF entries, parser is matching the wrong lines: {declared:?}"
    );
}

#[test]
fn every_fixture_is_downloadable() {
    let declared = declared_in_download_script();
    for fixture in ALL {
        assert!(
            declared.contains(&fixture.file.to_string()),
            "fixture {} ({}) is not fetched by models/download.sh.\n  \
             download.sh provides: {declared:?}\n  \
             fix: add it to the MODELS map, or point the fixture at a file the script fetches",
            fixture.file,
            fixture.capability
        );
    }
}

#[test]
fn every_download_is_used_by_a_fixture() {
    // The direction that catches waste rather than breakage. A model in the
    // script that no fixture claims is downloaded by every contributor and
    // cached by CI for nothing.
    let declared = declared_in_download_script();
    let used: Vec<&str> = ALL.iter().map(|f| f.file).collect();
    for name in &declared {
        assert!(
            used.contains(&name.as_str()),
            "models/download.sh fetches {name}, but no fixture in tests/common uses it.\n  \
             fixtures in use: {used:?}\n  \
             fix: drop it from the MODELS map, or add a Fixture that needs it",
        );
    }
}

#[test]
fn fixture_env_overrides_are_distinct() {
    // Two fixtures sharing an override variable would make one silently
    // unreachable — setting it would redirect both.
    for (i, a) in ALL.iter().enumerate() {
        for b in &ALL[i + 1..] {
            assert_ne!(
                a.env, b.env,
                "fixtures {} and {} share the override variable {}",
                a.file, b.file, a.env
            );
            assert_ne!(a.file, b.file, "two fixtures both point at {}", a.file);
        }
    }
}

#[test]
fn fixture_override_redirects_the_path() {
    // The override is the documented escape hatch for running the suite
    // against models kept outside the repo; if it stopped being read, every
    // such run would silently fall back to the bundled path.
    let fixture = &common::HYBRID;
    // SAFETY: `set_var` is unsafe in edition 2024 because a concurrent reader
    // of the environment is UB. This is the only test in this binary that
    // touches this variable, and the binary is single-threaded here.
    unsafe { std::env::set_var(fixture.env, "/tmp/somewhere-else.gguf") };
    let path = fixture.path();
    unsafe { std::env::remove_var(fixture.env) };
    assert_eq!(path, "/tmp/somewhere-else.gguf");
    // And falls back once the override is gone.
    assert!(fixture.path().ends_with(fixture.file));
}
