"""
Code-Based experiment runner with bounded technical debugging.

FINAL EXPERIMENTAL PROTOCOL
---------------------------
Each experimental run starts with one zero-shot GPT-4o code generation. If the
response cannot be executed or evaluated for purely technical reasons, the
model receives the technical failure information and may revise its code.
At most MAX_DEBUG_ITERATIONS technical debugging rounds are allowed after the
initial generation.

Technical feedback is limited to:
- missing ```python``` code block
- Python execution error / traceback
- execution timeout
- missing / malformed JSON output

NO solution-quality feedback is returned to the model. In particular, GPT-4o
is never told whether a parseable solution is feasible, whether its reported
objective is correct, or how far it is from the Gurobi optimum. As soon as code
executes successfully and produces parseable JSON, the LLM interaction stops.
The solution is then independently evaluated exactly once.

Thus one experimental run contains:
    initial generation + up to 5 technical debugging generations

A run that still has no parseable output after the allowed debugging rounds is
stored as a failed run.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError

from src.config import (
    CHATGPT_MODEL,
    CODE_EXECUTION_TIMEOUT,
    N_RUNS_PER_INSTANCE,
    TEMPERATURE,
)
from src.instances.loader import load_all_instances
from src.paths import APPROACH_LABELS, RESULT_DIRS
from src.chatgpt.prompts.main_prompts import (
    build_code_based_prompt_siclsp,
    build_code_based_prompt_smtwtp,
)

CODE_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", re.DOTALL)
FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

TREATMENT = "code_based"

# Five debugging rounds AFTER the initial generation.
MAX_DEBUG_ITERATIONS = 5

RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, InternalServerError)
MAX_API_RETRIES = 5


def call_chat_completion(client: OpenAI, model: str, messages: list,
                         temperature: float = None, reasoning_effort: str = None) -> str:
    """Call the model, retrying only transient API/infrastructure failures."""
    for attempt in range(MAX_API_RETRIES):
        try:
            if reasoning_effort is not None:
                response = client.responses.create(
                    model=model,
                    input=messages,
                    reasoning={"effort": reasoning_effort},
                )
                return response.output_text

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return completion.choices[0].message.content

        except RETRYABLE_ERRORS as e:
            if attempt == MAX_API_RETRIES - 1:
                raise
            wait_seconds = 2 ** attempt
            print(
                f"  [API retry {attempt + 1}/{MAX_API_RETRIES}] "
                f"{type(e).__name__}: {e} — retrying same request in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)


def _extract_code(response_text: str) -> str:
    match = CODE_BLOCK_PATTERN.search(response_text)
    if not match:
        raise ValueError("No ```python ...``` code block found in the response.")
    return match.group(1).strip()


def _find_balanced_json_objects(text: str) -> list:
    objects = []
    depth = 0
    start = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start:i + 1])

    return objects


def extract_json_from_text(text: str) -> dict:
    stripped = text.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for candidate in reversed(FENCED_JSON_PATTERN.findall(stripped)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    for candidate in reversed(_find_balanced_json_objects(stripped)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in program output.")


def run_code_in_sandbox(code: str, timeout: int = CODE_EXECUTION_TIMEOUT) -> dict:
    """Execute generated code in a separate Python process with a hard timeout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return {
            "success": False,
            "stdout": stdout,
            "stderr": stderr or f"TimeoutExpired: execution exceeded {timeout}s.",
            "returncode": None,
        }
    finally:
        os.unlink(tmp_path)


def _technical_feedback(failure_type: str, error: str, stdout: str = "") -> str:
    """
    Construct narrowly scoped execution feedback.

    Deliberately contains no feasibility, objective, optimality, or ground-truth
    information. The model is asked only to repair the technical issue and
    return a complete replacement program.
    """
    if failure_type == "missing_code_block":
        detail = error
    elif failure_type == "invalid_json_output":
        detail = (
            f"The program executed, but its stdout did not contain valid JSON.\n"
            f"Parser error: {error}\n"
            f"Program stdout:\n{stdout[-4000:]}"
        )
    else:
        detail = error[-6000:] if error else failure_type

    return (
        "The generated program could not be technically evaluated because of the "
        "following execution/output issue:\n\n"
        f"{detail}\n\n"
        "Fix only the technical issue and return the COMPLETE corrected Python "
        "program again in exactly one ```python ...``` code block. Preserve the "
        "original optimization task and required JSON output format. Do not "
        "explain the fix outside the code block."
    )


def _evaluate_solution(problem: str, parsed_output: dict, instance: dict) -> tuple:
    """Independently check feasibility and recompute objective after LLM interaction ends."""
    from src.evaluation.metrics import (
        check_feasibility_siclsp,
        check_feasibility_smtwtp,
        compute_siclsp_objective,
        compute_smtwtp_objective,
    )

    if problem == "smtwtp":
        sequence = parsed_output.get("sequence")
        feasible = sequence is not None and check_feasibility_smtwtp(sequence, instance)
        objective = compute_smtwtp_objective(sequence, instance) if feasible else None
        return feasible, objective

    if problem == "siclsp":
        production = parsed_output.get("production")
        feasible = production is not None and check_feasibility_siclsp(production, instance)
        objective = compute_siclsp_objective(production, instance) if feasible else None
        return feasible, objective

    raise ValueError(f"Unknown problem: {problem}")


def run_code_based_solution(client: OpenAI, initial_prompt: str, problem: str, instance: dict,
                            model: str = None, reasoning_effort: str = None) -> dict:
    """Run one Code-Based experiment with up to five technical debugging rounds."""
    effective_model = model or CHATGPT_MODEL
    effective_temperature = None if reasoning_effort is not None else TEMPERATURE

    messages = [{"role": "user", "content": initial_prompt}]
    attempt_history = []

    final_code = None
    final_output = None
    parsed_output = None
    feasible = None
    recomputed = None
    failure_type = None
    error = None
    n_debug_iterations = 0

    # attempt_index 0 = initial generation; 1..5 = debugging generations.
    for attempt_index in range(MAX_DEBUG_ITERATIONS + 1):
        response_text = call_chat_completion(
            client,
            effective_model,
            messages,
            temperature=effective_temperature,
            reasoning_effort=reasoning_effort,
        )
        messages.append({"role": "assistant", "content": response_text})

        current_code = None
        current_stdout = ""
        current_failure_type = None
        current_error = None

        try:
            current_code = _extract_code(response_text)
        except ValueError as e:
            current_failure_type = "missing_code_block"
            current_error = str(e)
        else:
            execution = run_code_in_sandbox(current_code)
            current_stdout = execution["stdout"]

            if not execution["success"]:
                current_failure_type = (
                    "timeout"
                    if execution["returncode"] is None
                    else "execution_error"
                )
                current_error = execution["stderr"]
            else:
                try:
                    parsed_output = extract_json_from_text(execution["stdout"])
                except ValueError as e:
                    current_failure_type = "invalid_json_output"
                    current_error = str(e)
                else:
                    # Technical success: stop LLM interaction immediately.
                    final_code = current_code
                    final_output = current_stdout
                    failure_type = None
                    error = None

                    attempt_history.append({
                        "attempt_index": attempt_index,
                        "is_debug_iteration": attempt_index > 0,
                        "technical_success": True,
                        "failure_type": None,
                        "error": None,
                        "code": current_code,
                        "stdout": current_stdout,
                    })
                    break

        attempt_history.append({
            "attempt_index": attempt_index,
            "is_debug_iteration": attempt_index > 0,
            "technical_success": False,
            "failure_type": current_failure_type,
            "error": current_error,
            "code": current_code,
            "stdout": current_stdout,
        })

        final_code = current_code
        final_output = current_stdout
        failure_type = current_failure_type
        error = current_error

        # No more model calls after the fifth debugging iteration.
        if attempt_index == MAX_DEBUG_ITERATIONS:
            break

        n_debug_iterations += 1
        feedback = _technical_feedback(
            current_failure_type,
            current_error or "",
            current_stdout,
        )
        messages.append({"role": "user", "content": feedback})

    # Independent evaluation happens only AFTER the LLM loop has ended.
    if parsed_output is not None:
        feasible, recomputed = _evaluate_solution(problem, parsed_output, instance)

    return {
        "conversation_history": messages,
        "attempt_history": attempt_history,
        "final_code": final_code,
        "final_output": final_output,
        "parsed_output": parsed_output,
        "feasible": feasible,
        "verified_objective": recomputed,
        "success": parsed_output is not None,
        "failure_type": failure_type if parsed_output is None else None,
        "error": error if parsed_output is None else None,
        "n_debug_iterations": n_debug_iterations,
        "n_model_generations": len([m for m in messages if m["role"] == "assistant"]),
        "model_used": effective_model,
        "reasoning_effort_used": reasoning_effort,
    }


def _build_prompt(problem: str, instance_data: dict) -> str:
    if problem == "smtwtp":
        return build_code_based_prompt_smtwtp(instance_data)
    if problem == "siclsp":
        return build_code_based_prompt_siclsp(instance_data)
    raise ValueError(f"Unknown problem: {problem}")


# The stored SMTWTP records carry two protocol fields and two placeholder
# fields on the exception path that the SICLSP records do not carry. The
# writers keep that asymmetry so re-runs stay schema-compatible with the
# collected data.
PROTOCOL_FIELDS = {
    "smtwtp": {"max_debug_iterations": MAX_DEBUG_ITERATIONS,
               "debug_feedback_scope": "technical_only"},
    "siclsp": {},
}
EXCEPTION_FIELDS = {
    "smtwtp": {"n_debug_iterations": None, "n_model_generations": None},
    "siclsp": {},
}


def _result_path(problem: str, instance_id: str, run_number: int) -> Path:
    out_dir = RESULT_DIRS[(TREATMENT, problem)]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{instance_id}_run{run_number}.json"


def run_all_code_based_solutions() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    instances = load_all_instances()

    for item in instances:
        problem = item["problem"]
        instance_id = item["instance_id"]
        data = item["data"]

        for run_number in range(1, N_RUNS_PER_INSTANCE + 1):
            out_path = _result_path(problem, instance_id, run_number)

            # Safe resume after interruption; never overwrite completed paid runs.
            if out_path.exists():
                continue

            prompt = _build_prompt(problem, data)

            try:
                run_result = run_code_based_solution(
                    client,
                    prompt,
                    problem,
                    data,
                )
                run_result.update({
                    "instance_id": instance_id,
                    "problem": problem,
                    "approach": APPROACH_LABELS[TREATMENT],
                    "run_number": run_number,
                    "model": CHATGPT_MODEL,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                run_result.update(PROTOCOL_FIELDS[problem])
            except Exception as e:
                run_result = {
                    "instance_id": instance_id,
                    "problem": problem,
                    "approach": APPROACH_LABELS[TREATMENT],
                    "run_number": run_number,
                    "model": CHATGPT_MODEL,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": False,
                    "failure_type": "runner_exception",
                    "error": str(e),
                }
                run_result.update(EXCEPTION_FIELDS[problem])
                run_result.update(PROTOCOL_FIELDS[problem])

            with open(out_path, "w") as f:
                json.dump(run_result, f, indent=2)

            status = "OK" if run_result.get("success") else "FAILED"
            dbg = run_result.get("n_debug_iterations")
            feasible = run_result.get("feasible")
            print(
                f"[{problem}] {instance_id} run {run_number}: {status} "
                f"| debug={dbg} | feasible={feasible}"
            )


if __name__ == "__main__":
    run_all_code_based_solutions()
