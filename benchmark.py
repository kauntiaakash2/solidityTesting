import os
import json
import csv
import time
import asyncio
from typing import List, Optional

from pydantic import BaseModel
from openai import AsyncOpenAI


# ==========================================
# 1. DATA MODELS & CONFIGURATION
# ==========================================

class ContractTask(BaseModel):
    task_id: str
    source_code: str
    ground_truth_rating: str
    ground_truth_category: Optional[str] = None


class LLMPrediction(BaseModel):
    rating: str
    vulnerability_category: str
    vulnerable_lines: List[int]


class BenchmarkConfig:
    # Local Ollama OpenAI-compatible API
    API_BASE_URL = "http://localhost:11434/v1"
    API_KEY = "ollama"

    # Exact model tag being downloaded
    MODELS_TO_TEST = [
        "qwen2.5-coder:7b-instruct-q4_K_M"
    ]

    # Ollama is running on CPU with one parallel request
    MAX_CONCURRENT_REQUESTS = 1

    # Maximum time allowed for one model response
    REQUEST_TIMEOUT_SECONDS = 600

    SYSTEM_PROMPT = (
        "You are an expert AI Security Researcher. "
        "Analyze the provided Solidity smart contract for vulnerabilities. "
        "Return only one valid JSON object with exactly these keys: "
        "\"rating\": either \"SAFE\" or \"VULNERABLE\", "
        "\"vulnerability_category\": a lowercase vulnerability category such as "
        "\"reentrancy\", \"access_control\", \"integer_overflow\", "
        "\"unchecked_call\", or \"none\", "
        "\"vulnerable_lines\": an array of integer line numbers. "
        "Do not include Markdown, explanations, or code fences."
    )


# ==========================================
# 2. DATA INGESTION
# ==========================================

def load_dataset_from_directory(base_path: str) -> List[ContractTask]:
    """
    Reads Solidity files from category directories such as
    the SmartBugs Curated dataset.
    """

    dataset: List[ContractTask] = []

    print(f"[*] Scanning dataset directory: {base_path}")

    if not os.path.exists(base_path):
        print(f"[!] Dataset path does not exist: {base_path}")
        return dataset

    if not os.path.isdir(base_path):
        print(f"[!] Dataset path is not a directory: {base_path}")
        return dataset

    for category in sorted(os.listdir(base_path)):
        category_path = os.path.join(base_path, category)

        if not os.path.isdir(category_path):
            continue

        for file_name in sorted(os.listdir(category_path)):
            if not file_name.endswith(".sol"):
                continue

            file_path = os.path.join(category_path, file_name)

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    source_code = file.read()

                dataset.append(
                    ContractTask(
                        task_id=f"{category}_{file_name}",
                        source_code=source_code,
                        ground_truth_rating="VULNERABLE",
                        ground_truth_category=category,
                    )
                )

            except Exception as error:
                print(f"[!] Failed to read {file_path}: {error}")

    print(f"[*] Loaded {len(dataset)} contracts.")

    return dataset


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def normalize_prediction(prediction: LLMPrediction) -> LLMPrediction:
    rating = prediction.rating.strip().upper()

    if rating not in {"SAFE", "VULNERABLE"}:
        raise ValueError(
            f"Invalid rating '{prediction.rating}'. "
            "Expected SAFE or VULNERABLE."
        )

    category = prediction.vulnerability_category.strip().lower()

    if rating == "SAFE":
        category = "none"

    valid_lines = sorted(
        {
            line
            for line in prediction.vulnerable_lines
            if isinstance(line, int) and line > 0
        }
    )

    return LLMPrediction(
        rating=rating,
        vulnerability_category=category,
        vulnerable_lines=valid_lines,
    )


def calculate_rating_correctness(
    predicted_rating: str,
    ground_truth_rating: str,
) -> bool:
    return predicted_rating.upper() == ground_truth_rating.upper()


def calculate_category_correctness(
    predicted_category: str,
    ground_truth_category: Optional[str],
) -> bool:
    if ground_truth_category is None:
        return False

    predicted = predicted_category.lower().replace("-", "_").replace(" ", "_")
    ground_truth = (
        ground_truth_category.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    return predicted == ground_truth


# ==========================================
# 4. EXECUTION ENGINE
# ==========================================

async def evaluate_contract(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    contract: ContractTask,
) -> dict:

    messages = [
        {
            "role": "system",
            "content": BenchmarkConfig.SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Contract ID: {contract.task_id}\n\n"
                "Analyze the following Solidity smart contract.\n"
                "Line numbers start from 1.\n\n"
                f"{contract.source_code}"
            ),
        },
    ]

    async with semaphore:
        start_time = time.perf_counter()

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                extra_body={
                    # Keep the model loaded in memory
                    "keep_alive": -1,
                },
            )

            latency = time.perf_counter() - start_time

            raw_content = response.choices[0].message.content

            if not raw_content:
                raise ValueError("The model returned an empty response.")

            parsed_json = json.loads(raw_content)
            parsed_prediction = LLMPrediction(**parsed_json)
            prediction = normalize_prediction(parsed_prediction)

            rating_correct = calculate_rating_correctness(
                prediction.rating,
                contract.ground_truth_rating,
            )

            category_correct = calculate_category_correctness(
                prediction.vulnerability_category,
                contract.ground_truth_category,
            )

            print(
                f"[OK] {contract.task_id} | "
                f"{latency:.2f}s | "
                f"{prediction.rating} | "
                f"{prediction.vulnerability_category}"
            )

            return {
                "task_id": contract.task_id,
                "model": model_name,
                "ground_truth_rating": contract.ground_truth_rating,
                "ground_truth_category": (
                    contract.ground_truth_category or ""
                ),
                "predicted_rating": prediction.rating,
                "predicted_category": (
                    prediction.vulnerability_category
                ),
                "vulnerable_lines": json.dumps(
                    prediction.vulnerable_lines
                ),
                "rating_correct": rating_correct,
                "category_correct": category_correct,
                "latency_sec": round(latency, 4),
                "success": True,
                "error": "",
                "raw_response": raw_content,
            }

        except Exception as error:
            latency = time.perf_counter() - start_time

            print(
                f"[ERROR] {contract.task_id} | "
                f"{latency:.2f}s | {error}"
            )

            return {
                "task_id": contract.task_id,
                "model": model_name,
                "ground_truth_rating": contract.ground_truth_rating,
                "ground_truth_category": (
                    contract.ground_truth_category or ""
                ),
                "predicted_rating": "",
                "predicted_category": "",
                "vulnerable_lines": "[]",
                "rating_correct": False,
                "category_correct": False,
                "latency_sec": round(latency, 4),
                "success": False,
                "error": str(error),
                "raw_response": "",
            }


async def run_suite(dataset: List[ContractTask]) -> List[dict]:
    client = AsyncOpenAI(
        api_key=BenchmarkConfig.API_KEY,
        base_url=BenchmarkConfig.API_BASE_URL,
        timeout=BenchmarkConfig.REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    semaphore = asyncio.Semaphore(
        BenchmarkConfig.MAX_CONCURRENT_REQUESTS
    )

    results: List[dict] = []

    try:
        for model_name in BenchmarkConfig.MODELS_TO_TEST:
            print("=" * 70)
            print(f"[*] Starting evaluation for model: {model_name}")
            print(
                f"[*] Contracts to evaluate: {len(dataset)}"
            )
            print(
                "[*] Concurrent requests: "
                f"{BenchmarkConfig.MAX_CONCURRENT_REQUESTS}"
            )
            print("=" * 70)

            tasks = [
                evaluate_contract(
                    client=client,
                    semaphore=semaphore,
                    model_name=model_name,
                    contract=contract,
                )
                for contract in dataset
            ]

            model_results = await asyncio.gather(*tasks)
            results.extend(model_results)

    finally:
        await client.close()

    return results


# ==========================================
# 5. RESULT EXPORT
# ==========================================

def export_results(results: List[dict], output_directory: str) -> None:
    if not results:
        print("[!] No results available to export.")
        return

    os.makedirs(output_directory, exist_ok=True)

    json_path = os.path.join(
        output_directory,
        "telemetry.json",
    )

    csv_path = os.path.join(
        output_directory,
        "telemetry.csv",
    )

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(
            results,
            json_file,
            indent=4,
            ensure_ascii=False,
        )

    csv_fieldnames = [
        "task_id",
        "model",
        "ground_truth_rating",
        "ground_truth_category",
        "predicted_rating",
        "predicted_category",
        "vulnerable_lines",
        "rating_correct",
        "category_correct",
        "latency_sec",
        "success",
        "error",
        "raw_response",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"[*] JSON results saved to: {json_path}")
    print(f"[*] CSV results saved to:  {csv_path}")


# ==========================================
# 6. SUMMARY
# ==========================================

def print_summary(results: List[dict]) -> None:
    if not results:
        return

    total = len(results)
    successful = sum(
        1 for result in results if result["success"]
    )
    failed = total - successful

    rating_correct = sum(
        1
        for result in results
        if result["success"] and result["rating_correct"]
    )

    category_correct = sum(
        1
        for result in results
        if result["success"] and result["category_correct"]
    )

    successful_latencies = [
        result["latency_sec"]
        for result in results
        if result["success"]
    ]

    average_latency = (
        sum(successful_latencies) / len(successful_latencies)
        if successful_latencies
        else 0
    )

    rating_accuracy = (
        rating_correct / successful * 100
        if successful
        else 0
    )

    category_accuracy = (
        category_correct / successful * 100
        if successful
        else 0
    )

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total requests:       {total}")
    print(f"Successful requests:  {successful}")
    print(f"Failed requests:      {failed}")
    print(f"Average latency:      {average_latency:.2f} seconds")
    print(f"Rating accuracy:      {rating_accuracy:.2f}%")
    print(f"Category accuracy:    {category_accuracy:.2f}%")
    print("=" * 70)


# ==========================================
# 7. MAIN TRIGGER
# ==========================================

if __name__ == "__main__":
    DATASET_PATH = (
        "/home/kauntiakash2/AI-ML/solidityTest/"
        "datasets/smartbugs-curated/dataset"
    )

    OUTPUT_DIRECTORY = "results"

    contracts = load_dataset_from_directory(DATASET_PATH)

    if not contracts:
        print("[!] No Solidity contracts were loaded.")
        print("[!] Benchmark was not started.")
        raise SystemExit(1)

    print(
        "[*] Ensure Ollama is running with:\n"
        "    ollama serve"
    )

    print(
        "[*] Ensure the model exists with:\n"
        "    ollama list"
    )

    final_results = asyncio.run(run_suite(contracts))

    export_results(
        results=final_results,
        output_directory=OUTPUT_DIRECTORY,
    )

    print_summary(final_results)

    print("\n[*] Benchmark complete.")
    print("[*] Check the 'results' directory.")