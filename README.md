# Solidity LLM Benchmark

`solidity_llm_benchmark.py` creates reproducible Solidity inputs and measures
streaming LLM review responses. It is intentionally provider-neutral: select
one or more providers with repeatable `KIND:MODEL` values.

## Quick start

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY=...                         # when using OpenAI
export ANTHROPIC_API_KEY=...                      # when using Anthropic
export DEEPSEEK_API_KEY=...                       # when using DeepSeek
python solidity_llm_benchmark.py --count 200 --seed 20260717 \
  --write-dataset cases.jsonl --database results.sqlite \
  --provider openai:gpt-4o --provider anthropic:claude-3-5-sonnet-latest
```

Use a local model with `--provider ollama:llama3.1`; override its endpoint with
`--ollama-url`. DeepSeek uses its OpenAI-compatible streaming endpoint and may
be overridden with `--deepseek-url`.

## Reproducibility and outputs

Keep the generated JSONL file and command-line seed with the SQLite database.
Each JSONL record contains its source, generator seed, domain, and complexity.
SQLite stores raw model output, error information, attempt count, TTFT, RTT,
estimated or provider-reported output tokens, throughput, and a JSON metrics
object. The final console JSON reports mean, median, and sample standard
deviation of RTT per provider.

The runner bounds concurrency globally, applies exponential backoff with jitter
to failed requests (including rate limits), and applies a per-request timeout.
Run `solc` on `PATH` to additionally populate syntactic validity; its absence
is recorded as `null`, rather than treating compiler availability as a model
failure. The dataset uses six domains (ERC-20, ERC-721, AMM, lending, multisig,
and governance) and cycles three complexity levels while deriving unique names
from a seeded random nonce.
