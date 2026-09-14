"""
SECONDARY experiment: Direct Solution + Zero-Shot Chain-of-Thought.

This is NOT one of the two main treatments. It repeats the MAIN Direct
protocol (zero-shot, single-turn, one call per run, no feedback) with exactly
one addition: the Zero-Shot-CoT trigger in the prompt (see
src.chatgpt.prompts.cot_prompts). Everything else — model, temperature,
instances, number of repetitions — is identical to the MAIN Direct treatment,
so the CoT trigger is the only difference between the two.

CoT results are stored in their own directories and are never counted among
the 1,000 main runs:
    SMTWTP -> data/results/direct_cot/smtwtp/
    SICLSP -> data/results/direct_cot_siclsp_final/siclsp/

Idempotent and resumable, like the main Direct runner.

    python -m src.chatgpt.run_direct_cot            # both problems
    python -m src.chatgpt.run_direct_cot siclsp     # one problem
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from src.config import CHATGPT_MODEL, N_RUNS_PER_INSTANCE, TEMPERATURE
from src.paths import APPROACH_LABELS, RESULT_DIRS, batch_state_dir
from src.chatgpt.batch_api import (
    CHUNK_TOKEN_BUDGET,
    build_request,
    chunk_requests,
    iter_batch_results,
    poll_batch,
    submit_batch,
    write_jsonl,
)
from src.chatgpt.prompts.cot_prompts import (
    build_direct_cot_prompt_siclsp,
    build_direct_cot_prompt_smtwtp,
)
from src.instances.loader import load_instances

TREATMENT = "direct_cot"

PROMPT_BUILDERS = {
    "smtwtp": build_direct_cot_prompt_smtwtp,
    "siclsp": build_direct_cot_prompt_siclsp,
}
PROBLEMS = tuple(PROMPT_BUILDERS)


def _result_path(problem: str, instance_id: str, run_number: int) -> Path:
    out_dir = RESULT_DIRS[(TREATMENT, problem)]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{instance_id}_run{run_number}.json"


def _batch_id_file(problem: str) -> Path:
    return batch_state_dir(TREATMENT, problem) / "_pending_batch_id.txt"


def build_pending_requests(problem: str) -> list:
    prompt_fn = PROMPT_BUILDERS[problem]
    requests = []

    for item in load_instances(problem):
        instance_id, data = item["instance_id"], item["data"]
        prompt = prompt_fn(data)

        for run_number in range(1, N_RUNS_PER_INSTANCE + 1):
            if _result_path(problem, instance_id, run_number).exists():
                continue
            requests.append(build_request(
                custom_id=f"{problem}__{instance_id}__run{run_number}",
                model=CHATGPT_MODEL, temperature=TEMPERATURE, prompt=prompt,
            ))

    return requests


def save_batch_results(problem: str, batch, client: OpenAI) -> int:
    saved = 0
    for custom_id, raw_response, error in iter_batch_results(batch, client):
        _prefix, instance_id, run_tag = custom_id.split("__")
        run_number = int(run_tag.replace("run", ""))

        result = {
            "instance_id": instance_id,
            "problem": problem,
            "approach": APPROACH_LABELS[TREATMENT],
            "run_number": run_number,
            "model": CHATGPT_MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_response": raw_response,
            "openai_error": error,
        }
        with open(_result_path(problem, instance_id, run_number), "w") as f:
            json.dump(result, f, indent=2)
        saved += 1
    return saved


def run_direct_cot(problem: str) -> None:
    """Run the SECONDARY Direct+CoT treatment for one problem."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    jsonl_path = batch_state_dir(TREATMENT, problem) / "_batch_requests.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    batch_id_file = _batch_id_file(problem)

    while True:
        if batch_id_file.exists():
            # A batch is already in flight (e.g. the process was interrupted):
            # resume polling instead of submitting — and paying for — a duplicate.
            batch_id = batch_id_file.read_text().strip()
            print(f"[direct_cot/{problem}] resuming existing batch {batch_id}")
        else:
            pending = build_pending_requests(problem)
            if not pending:
                print(f"[direct_cot/{problem}] complete — nothing left to do.")
                return

            chunk = chunk_requests(pending, CHUNK_TOKEN_BUDGET)[0]
            write_jsonl(chunk, str(jsonl_path))
            print(f"[direct_cot/{problem}] submitting {len(chunk)} of {len(pending)} pending requests...")
            batch_id = submit_batch(str(jsonl_path), client)
            batch_id_file.write_text(batch_id)
            print(f"[direct_cot/{problem}] batch started: {batch_id}")

        batch = poll_batch(batch_id, client)
        print(f"[direct_cot/{problem}] batch status: {batch.status}")

        if batch.status == "completed":
            n_saved = save_batch_results(problem, batch, client)
            print(f"[direct_cot/{problem}] {n_saved} results stored.")
            batch_id_file.unlink(missing_ok=True)
        else:
            print(f"[direct_cot/{problem}] batch not successful (status: {batch.status}).")
            if batch.error_file_id:
                print(client.files.content(batch.error_file_id).text[:3000])
            batch_id_file.unlink(missing_ok=True)
            return


def main(argv: list) -> None:
    problems = argv[1:] if len(argv) > 1 else list(PROBLEMS)
    for problem in problems:
        if problem not in PROBLEMS:
            raise SystemExit(f"Unknown problem {problem!r} (expected: {', '.join(PROBLEMS)})")
        run_direct_cot(problem)


if __name__ == "__main__":
    main(sys.argv)
