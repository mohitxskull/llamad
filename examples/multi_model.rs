//! Running several models in one process: a small router plus a large thinker.
//!
//! Usage:
//!   cargo run --release --example multi_model -- <router.gguf> <thinker.gguf> [prompt]
//!
//! Shows the three things that are not obvious from the API alone:
//!
//! 1. `Client::with_config`, not `Client::new` — the `LLAMAD_*` variables are
//!    process-global, so `new` would give both models the same context and
//!    thread budget.
//! 2. Thread counts chosen for how the models are used. Here they take turns
//!    (route, then answer), so each gets all physical cores: a model is
//!    fastest when it is the only one running. Split the cores instead only if
//!    two models generate simultaneously and sustained.
//! 3. Lazy loading with a policy you control, rather than a registry with a
//!    policy you do not. The thinker is 5x the size of the router and is only
//!    needed for some prompts, so it loads on first use and then stays.

use std::sync::Mutex;

use llamad::client::Client;
use llamad::config::InferenceConfig;
use llamad::protocol::{LlamaError, Request};

/// A model that loads on first use and stays resident afterwards.
///
/// This is the whole of what a model registry would do for you, minus the
/// eviction policy — which is the part worth owning yourself, since an
/// eviction at the wrong moment stalls a user request for as long as it takes
/// to read the GGUF back off disk.
struct Lazy {
    path: String,
    config: InferenceConfig,
    slot: Mutex<Option<Client>>,
}

impl Lazy {
    fn new(path: impl Into<String>, config: InferenceConfig) -> Self {
        Lazy {
            path: path.into(),
            config,
            slot: Mutex::new(None),
        }
    }

    /// Run `f` against the model, loading it first if this is the first call.
    fn with<R>(&self, f: impl FnOnce(&Client) -> R) -> Result<R, LlamaError> {
        let mut guard = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_none() {
            eprintln!("[loading {}]", self.path);
            *guard = Some(Client::with_config(&self.path, self.config.clone())?);
        }
        Ok(f(guard.as_ref().expect("just loaded")))
    }

    /// Unload the model and release its memory. The next `with` reloads it.
    fn unload(&self) {
        drop(self.slot.lock().unwrap_or_else(|p| p.into_inner()).take());
    }
}

/// A grammar admitting exactly one of the two routing labels.
///
/// Leading whitespace is allowed on purpose: chat models like to open with a
/// space or newline, and a grammar that forbids it can leave llama.cpp with no
/// legal token.
const ROUTE_GRAMMAR: &str = r#"root ::= [ \n]* ("simple" | "hard")"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: {} <router.gguf> <thinker.gguf> [prompt]", args[0]);
        std::process::exit(2);
    }
    let prompt = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "What is 17 * 23?".to_owned());

    // The router is small and answers constantly: short context is plenty.
    let router = Client::with_config(
        &args[1],
        InferenceConfig {
            n_ctx: 1024,
            ..Default::default()
        },
    )?;

    // The thinker needs room for a reasoning chain, and is not loaded until a
    // prompt actually needs it.
    let thinker = Lazy::new(
        &args[2],
        InferenceConfig {
            n_ctx: 4096,
            ..Default::default()
        },
    );

    // Classify. The grammar makes the answer parseable without post-processing
    // — the model cannot emit anything but one of the two labels.
    let route = router.complete(
        Request::new(format!("Task: {prompt}\nIs this simple or hard?"))
            .with_system("Answer with exactly one word.")
            .with_grammar(ROUTE_GRAMMAR)
            .with_temperature(0.0)
            .with_max_tokens(8),
    )?;
    let hard = route.text.trim() == "hard";
    println!(
        "route: {} ({} prompt tokens)",
        route.text.trim(),
        route.prompt_tokens
    );

    // Answer on whichever model the route chose. Only the "hard" path pays for
    // loading the larger model.
    let answer = if hard {
        thinker.with(|c| c.complete(Request::new(&prompt).with_max_tokens(512)))??
    } else {
        router.complete(Request::new(&prompt).with_max_tokens(128))?
    };

    println!("---\n{}", answer.text.trim());
    println!(
        "---\n{} tokens generated on the {} model",
        answer.generated_tokens,
        if hard { "thinker" } else { "router" }
    );

    // Both models stay resident until dropped. An idle engine costs memory but
    // no CPU — its inference thread parks on the command channel — so holding
    // the thinker is only worth undoing if you need the memory back.
    thinker.unload();
    Ok(())
}
