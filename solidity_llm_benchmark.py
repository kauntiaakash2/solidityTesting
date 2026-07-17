#!/usr/bin/env python3
"""Reproducible Solidity generation and multi-provider LLM benchmark runner.

The runner deliberately stores raw outputs and every timing observation so that
aggregate results can be independently recomputed.  Secrets are only read from
environment variables and are never persisted in the SQLite database.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Protocol

SYSTEM_PROMPT = """You are a Solidity security reviewer. Return a corrected or
annotated version of the supplied contract. Return only Solidity source in one
```solidity fenced block; do not omit any code."""


@dataclass(frozen=True)
class ContractCase:
    """An immutable benchmark input with a stable, content-addressed identifier."""

    case_id: str
    domain: str
    complexity: int
    source: str
    seed: int


@dataclass
class InferenceResult:
    """The complete, per-attempt observation used for later analysis."""

    run_id: str
    case_id: str
    provider: str
    model: str
    started_at: str
    ttft_seconds: float | None
    rtt_seconds: float | None
    output_tokens: int | None
    tokens_per_second: float | None
    response: str | None
    error: str | None
    attempts: int


def _identifier(value: str) -> str:
    return "Benchmark" + re.sub(r"[^A-Za-z0-9]", "", value.title())


class SolidityDataset:
    """Synthesizes structurally varied, self-contained contracts without I/O."""

    DOMAINS = ("erc20", "erc721", "amm", "lending", "multisig", "governance")

    @staticmethod
    def generate(count: int, seed: int = 42) -> list[ContractCase]:
        if not 1 <= count <= 500:
            raise ValueError("count must be between 1 and 500")
        rng = random.Random(seed)
        cases: list[ContractCase] = []
        for index in range(count):
            domain = SolidityDataset.DOMAINS[index % len(SolidityDataset.DOMAINS)]
            complexity = 1 + (index // len(SolidityDataset.DOMAINS)) % 3
            nonce = rng.randrange(10**9)
            name = _identifier(f"{domain}_{index}_{nonce}")
            source = SolidityDataset._source(domain, name, complexity, nonce)
            digest = hashlib.sha256(source.encode()).hexdigest()[:16]
            cases.append(ContractCase(digest, domain, complexity, source, nonce))
        return cases

    @staticmethod
    def _source(domain: str, name: str, complexity: int, nonce: int) -> str:
        header = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n\n"
        utility = """abstract contract Owned { address public owner; constructor() { owner = msg.sender; } modifier onlyOwner() { require(msg.sender == owner, \"owner\"); _; } }\n"""
        if domain == "erc20":
            body = f"""contract {name} is Owned {{ mapping(address => uint256) public balanceOf; uint256 public totalSupply; event Transfer(address indexed from, address indexed to, uint256 value); constructor(uint256 supply) {{ totalSupply = supply; balanceOf[msg.sender] = supply; }} function transfer(address to, uint256 amount) external {{ require(balanceOf[msg.sender] >= amount, \"balance\"); unchecked {{ balanceOf[msg.sender] -= amount; balanceOf[to] += amount; }} emit Transfer(msg.sender, to, amount); }} }}"""
        elif domain == "erc721":
            body = f"""contract {name} is Owned {{ mapping(uint256 => address) public ownerOf; uint256 public nextId; event Transfer(address indexed from, address indexed to, uint256 indexed id); function mint(address to) external onlyOwner returns (uint256 id) {{ id = ++nextId; ownerOf[id] = to; emit Transfer(address(0), to, id); }} function transfer(address to, uint256 id) external {{ require(ownerOf[id] == msg.sender, \"owner\"); ownerOf[id] = to; emit Transfer(msg.sender, to, id); }} }}"""
        elif domain == "amm":
            body = f"""contract {name} {{ uint256 public reserveA; uint256 public reserveB; event Swap(address indexed user, uint256 input, uint256 output); constructor(uint256 a, uint256 b) payable {{ reserveA = a; reserveB = b; }} function swapAForB(uint256 amountIn) external returns (uint256 amountOut) {{ require(amountIn > 0 && reserveB > 0, \"liquidity\"); amountOut = (amountIn * reserveB) / (reserveA + amountIn); reserveA += amountIn; reserveB -= amountOut; emit Swap(msg.sender, amountIn, amountOut); }} }}"""
        elif domain == "lending":
            body = f"""contract {name} {{ mapping(address => uint256) public deposits; mapping(address => uint256) public debt; uint256 public constant COLLATERAL_RATIO = 150; function deposit() external payable {{ deposits[msg.sender] += msg.value; }} function borrow(uint256 amount) external {{ require(deposits[msg.sender] * 100 >= (debt[msg.sender] + amount) * COLLATERAL_RATIO, \"collateral\"); debt[msg.sender] += amount; }} function repay() external payable {{ require(msg.value <= debt[msg.sender], \"excess\"); debt[msg.sender] -= msg.value; }} }}"""
        elif domain == "multisig":
            body = f"""contract {name} is Owned {{ mapping(address => bool) public signer; mapping(bytes32 => uint256) public approvals; uint256 public immutable threshold; constructor(address[] memory signers, uint256 needed) {{ require(needed > 0 && needed <= signers.length, \"threshold\"); threshold = needed; for (uint256 i; i < signers.length; ++i) signer[signers[i]] = true; }} function approve(bytes32 operation) external {{ require(signer[msg.sender], \"signer\"); approvals[operation]++; }} function executable(bytes32 operation) external view returns (bool) {{ return approvals[operation] >= threshold; }} }}"""
        else:
            body = f"""contract {name} is Owned {{ mapping(address => uint256) public votingPower; mapping(bytes32 => uint256) public votes; uint256 public quorum; constructor(uint256 initialQuorum) {{ quorum = initialQuorum; }} function grantVotes(address voter, uint256 amount) external onlyOwner {{ votingPower[voter] += amount; }} function vote(bytes32 proposal) external {{ votes[proposal] += votingPower[msg.sender]; }} function passed(bytes32 proposal) external view returns (bool) {{ return votes[proposal] >= quorum; }} }}"""
        padding = "\n".join(
            f"    uint256 private constant VARIANT_{i} = {nonce % 997 + i};"
            for i in range(complexity - 1)
        )
        return (
            header + utility + body[:-1] + ("\n" + padding if padding else "") + "\n}\n"
        )

    @staticmethod
    def write_jsonl(cases: Iterable[ContractCase], path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(asdict(case)) + "\n")

    @staticmethod
    def read_jsonl(path: Path) -> list[ContractCase]:
        with path.open(encoding="utf-8") as handle:
            return [ContractCase(**json.loads(line)) for line in handle if line.strip()]


class Provider(Protocol):
    name: str
    model: str

    async def stream(self, prompt: str) -> AsyncIterator[tuple[str, int | None]]: ...


class OpenAIProvider:
    def __init__(
        self, model: str, name: str = "openai", base_url: str | None = None
    ) -> None:
        self.model, self.name, self.base_url = model, name, base_url

    async def stream(self, prompt: str) -> AsyncIterator[tuple[str, int | None]]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY") if self.name == "deepseek" else None,
            base_url=self.base_url,
        )
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        async for event in stream:
            text = event.choices[0].delta.content if event.choices else None
            if text:
                yield text, None


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str) -> None:
        self.model = model

    async def stream(self, prompt: str) -> AsyncIterator[tuple[str, int | None]]:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()
        async with client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text, None


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, base_url: str) -> None:
        self.model, self.base_url = model, base_url.rstrip("/")

    async def stream(self, prompt: str) -> AsyncIterator[tuple[str, int | None]]:
        import aiohttp

        payload = {
            "model": self.model,
            "prompt": SYSTEM_PROMPT + "\n\n" + prompt,
            "stream": True,
        }
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.base_url + "/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    event = json.loads(line)
                    if event.get("response"):
                        yield event["response"], event.get("eval_count")


async def infer(
    case: ContractCase, provider: Provider, retries: int, timeout: float
) -> InferenceResult:
    """Run one streaming request; retries preserve only the final observation."""
    prompt = "Review this Solidity contract:\n```solidity\n" + case.source + "\n```"
    started = datetime.now(timezone.utc).isoformat()
    for attempt in range(1, retries + 2):
        began = time.perf_counter()
        first: float | None = None
        pieces: list[str] = []
        tokens = None
        try:
            async with asyncio.timeout(timeout):
                async for text, reported_tokens in provider.stream(prompt):
                    if first is None:
                        first = time.perf_counter()
                    pieces.append(text)
                    tokens = reported_tokens or tokens
            elapsed = time.perf_counter() - began
            response = "".join(pieces)
            output_tokens = tokens or max(1, math.ceil(len(response) / 4))
            return InferenceResult(
                str(uuid.uuid4()),
                case.case_id,
                provider.name,
                provider.model,
                started,
                (first - began if first else None),
                elapsed,
                output_tokens,
                output_tokens / elapsed if elapsed else None,
                response,
                None,
                attempt,
            )
        except (
            Exception
        ) as exc:  # SDK-specific status exceptions are intentionally normalized.
            if attempt > retries:
                return InferenceResult(
                    str(uuid.uuid4()),
                    case.case_id,
                    provider.name,
                    provider.model,
                    started,
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"{type(exc).__name__}: {exc}",
                    attempt,
                )
            await asyncio.sleep(min(30.0, 2 ** (attempt - 1)) + random.random())
    raise AssertionError("unreachable")


def extract_solidity(response: str) -> str:
    match = re.search(
        r"```(?:solidity)?\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE
    )
    return (match.group(1) if match else response).strip() + "\n"


def analyze(original: str, response: str | None) -> dict[str, Any]:
    original = original.strip() + "\n"
    candidate = extract_solidity(response or "")
    matcher = SequenceMatcher(None, original.splitlines(), candidate.splitlines())
    inserts = deletes = 0
    for tag, _a, _b, _c, _d in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            inserts += _d - _c
        if tag in ("delete", "replace"):
            deletes += _b - _a
    solc = shutil.which("solc")
    valid: bool | None = None
    if solc and candidate:
        process = subprocess.run(
            [solc, "--standard-json"],
            input=json.dumps(
                {
                    "language": "Solidity",
                    "sources": {"Candidate.sol": {"content": candidate}},
                    "settings": {"outputSelection": {"*": {"": ["ast"]}}},
                }
            ),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        valid = process.returncode == 0 and not any(
            item.get("severity") == "error"
            for item in json.loads(process.stdout).get("errors", [])
        )
    return {
        "levenshtein": _levenshtein(original, candidate),
        "line_insertions": inserts,
        "line_deletions": deletes,
        "syntactically_valid": valid,
        "unified_diff": "".join(
            unified_diff(
                original.splitlines(True),
                candidate.splitlines(True),
                fromfile="original.sol",
                tofile="response.sol",
            )
        ),
    }


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    row = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(right, 1):
            previous, row[j] = row[j], min(
                row[j] + 1, row[j - 1] + 1, previous + (char != other)
            )
    return row[-1]


class ResultStore:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS results (run_id TEXT PRIMARY KEY, case_id TEXT, provider TEXT, model TEXT, started_at TEXT, ttft_seconds REAL, rtt_seconds REAL, output_tokens INTEGER, tokens_per_second REAL, response TEXT, error TEXT, attempts INTEGER, metrics_json TEXT)"
        )

    def save(self, result: InferenceResult, metrics: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO results VALUES (:run_id,:case_id,:provider,:model,:started_at,:ttft_seconds,:rtt_seconds,:output_tokens,:tokens_per_second,:response,:error,:attempts,:metrics)",
            {**asdict(result), "metrics": json.dumps(metrics)},
        )
        self.connection.commit()

    def summary(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT provider, rtt_seconds FROM results WHERE rtt_seconds IS NOT NULL"
        ).fetchall()
        groups: dict[str, list[float]] = {}
        for provider, value in rows:
            groups.setdefault(provider, []).append(value)
        return [
            {
                "provider": name,
                "count": len(values),
                "mean_rtt": statistics.mean(values),
                "median_rtt": statistics.median(values),
                "stdev_rtt": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
            for name, values in groups.items()
        ]


async def run(args: argparse.Namespace) -> None:
    cases = (
        SolidityDataset.read_jsonl(Path(args.dataset))
        if args.dataset
        else SolidityDataset.generate(args.count, args.seed)
    )
    if args.write_dataset:
        SolidityDataset.write_jsonl(cases, Path(args.write_dataset))
    providers: list[Provider] = []
    for spec in args.provider:
        kind, model = spec.split(":", 1)
        if kind == "ollama":
            providers.append(OllamaProvider(model, args.ollama_url))
        elif kind == "openai":
            providers.append(OpenAIProvider(model))
        elif kind == "deepseek":
            providers.append(OpenAIProvider(model, "deepseek", args.deepseek_url))
        elif kind == "anthropic":
            providers.append(AnthropicProvider(model))
        else:
            raise ValueError(
                "provider kind must be openai, anthropic, deepseek, or ollama"
            )
    store = ResultStore(Path(args.database))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def task(case: ContractCase, provider: Provider) -> None:
        async with semaphore:
            result = await infer(case, provider, args.retries, args.timeout)
            store.save(result, analyze(case.source, result.response))

    await asyncio.gather(
        *(task(case, provider) for case in cases for provider in providers)
    )
    print(json.dumps(store.summary(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", help="Existing JSONL dataset to replay")
    parser.add_argument("--write-dataset")
    parser.add_argument("--database", default="benchmark.sqlite")
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="KIND:MODEL",
        help="openai, anthropic, or ollama; may be repeated",
    )
    parser.add_argument(
        "--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434")
    )
    parser.add_argument(
        "--deepseek-url",
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.provider:
        parser.error("at least one --provider KIND:MODEL is required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
