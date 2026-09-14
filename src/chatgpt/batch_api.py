"""
Shared OpenAI Batch API mechanics for the Direct-style treatments.

Both Direct treatments (MAIN Direct and SECONDARY Direct+CoT) are a single
prompt -> single answer step without intermediate execution, which makes them
a natural fit for the asynchronous Batch API (50% cheaper, runs unattended).

This module contains only transport mechanics — building request lines,
chunking, submitting, polling, and parsing batch output. Which prompt is used
and where results are written is decided by the calling runner, so that a
main-experiment runner can never reach a CoT prompt through this layer.

CHUNK_TOKEN_BUDGET exists because OpenAI limits the total number of tokens
enqueued across all running batches of an organisation. A single 250-request
batch containing the largest instances exceeded that limit in practice and
failed with "token_limit_exceeded" before any request was processed, so
requests are split into chunks below the budget and processed sequentially.
"""

import json
import time

from openai import OpenAI

# Conservative per-chunk token budget (see module docstring).
CHUNK_TOKEN_BUDGET = 60_000


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token). Sufficient for chunk
    budgeting; not a replacement for exact tokenisation."""
    return max(1, len(text) // 4)


def estimate_request_tokens(request: dict) -> int:
    total = sum(estimate_tokens(m["content"]) for m in request["body"]["messages"])
    return total + 50  # small per-request overhead buffer


def chunk_requests(requests: list, token_budget: int = CHUNK_TOKEN_BUDGET) -> list:
    """Split requests into chunks that each stay below the token budget
    (at least one request per chunk, even if it alone exceeds the budget)."""
    chunks = []
    current_chunk, current_tokens = [], 0

    for req in requests:
        t = estimate_request_tokens(req)
        if current_chunk and current_tokens + t > token_budget:
            chunks.append(current_chunk)
            current_chunk, current_tokens = [], 0
        current_chunk.append(req)
        current_tokens += t

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def write_jsonl(requests: list, path: str) -> None:
    with open(path, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")


def build_request(custom_id: str, model: str, temperature: float, prompt: str) -> dict:
    """One Batch API request line (chat completions endpoint)."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def submit_batch(jsonl_path: str, client: OpenAI) -> str:
    """Upload the .jsonl file and start the batch job. Returns the batch id."""
    with open(jsonl_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def poll_batch(batch_id: str, client: OpenAI, poll_interval: int = 60):
    """Poll until the batch reaches a terminal state. Blocking by design —
    the Direct treatments are meant to run unattended."""
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(poll_interval)


def iter_batch_results(batch, client: OpenAI):
    """Yield (custom_id, raw_response_text_or_None, error) per output line.

    The output line order may differ from the input order, which is why the
    custom_id is the only correlation key used.
    """
    if batch.output_file_id is None:
        return

    content = client.files.content(batch.output_file_id).text
    for line in content.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        response_body = record.get("response", {}).get("body", {})
        raw_response = None
        if response_body.get("choices"):
            raw_response = response_body["choices"][0]["message"]["content"]
        yield record["custom_id"], raw_response, record.get("error")
