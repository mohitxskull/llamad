use llama_cpp_4::prelude::*;

use crate::protocol::{Grammar, LlamaError, SamplingParams};

type Result<T> = std::result::Result<T, LlamaError>;

/// Parameterized description of one sampler in the decoding chain.
///
/// Data rather than opaque [`LlamaSampler`] handles so the chain's content
/// and order are unit-testable (the FFI handle cannot be introspected).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SamplerKind {
    Penalties {
        penalty_last_n: i32,
        penalty_repeat: f32,
    },
    TopK {
        k: i32,
    },
    TopP {
        p: f32,
        min_keep: usize,
    },
    Temp {
        t: f32,
    },
    Dist {
        seed: u32,
    },
}

impl SamplerKind {
    pub(crate) fn into_sampler(self, n_vocab: i32) -> LlamaSampler {
        match self {
            SamplerKind::Penalties {
                penalty_last_n,
                penalty_repeat,
            } => LlamaSampler::penalties_simple(n_vocab, penalty_last_n, penalty_repeat),
            SamplerKind::TopK { k } => LlamaSampler::top_k(k),
            SamplerKind::TopP { p, min_keep } => LlamaSampler::top_p(p, min_keep),
            SamplerKind::Temp { t } => LlamaSampler::temp(t),
            SamplerKind::Dist { seed } => LlamaSampler::dist(seed),
        }
    }
}

/// The decoding chain for a set of resolved parameters. Empty means greedy
/// decoding (no sampler chain), which ignores every parameter but temperature.
pub(crate) fn sampler_chain(params: SamplingParams) -> Vec<SamplerKind> {
    if params.temperature <= 0.0 {
        Vec::new()
    } else {
        vec![
            SamplerKind::Penalties {
                penalty_last_n: params.repeat_last_n,
                penalty_repeat: params.repeat_penalty,
            },
            SamplerKind::TopK { k: params.top_k },
            SamplerKind::TopP {
                p: params.top_p,
                min_keep: 1,
            },
            SamplerKind::Temp {
                t: params.temperature,
            },
            SamplerKind::Dist { seed: params.seed },
        ]
    }
}

/// Build a grammar sampler, turning llama.cpp's failure modes into an error.
pub(crate) fn build_grammar_sampler(
    model: &LlamaModel,
    grammar: &str,
    root: &str,
) -> Result<LlamaSampler> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        LlamaSampler::grammar(model, grammar, root)
    }))
    .map_err(|_| {
        LlamaError::Protocol(format!(
            "grammar failed to compile (start rule {root:?}); check the GBNF syntax"
        ))
    })
}

/// Assemble the sampler for one request.
pub(crate) fn build_sampler(
    model: &LlamaModel,
    params: SamplingParams,
    grammar: Option<&Grammar>,
) -> Result<LlamaSampler> {
    let mut samplers: Vec<LlamaSampler> = Vec::new();

    if let Some(g) = grammar {
        samplers.push(build_grammar_sampler(model, &g.text, &g.root)?);
    }

    let kinds = sampler_chain(params);
    let greedy = kinds.is_empty();
    let n_vocab = model.n_vocab();
    samplers.extend(kinds.into_iter().map(|k| k.into_sampler(n_vocab)));
    if greedy {
        samplers.push(LlamaSampler::greedy());
    }

    Ok(LlamaSampler::chain_simple(samplers))
}
