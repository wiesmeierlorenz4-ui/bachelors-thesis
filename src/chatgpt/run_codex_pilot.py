"""
Codex pilot: a small additional experiment, not part of the main study.

Runs the Code-Based task with a coding-specialised reasoning model on the first
instance (index 0) of every size, one run each, for both problems. Uses the
self-consistency solution loop (code_based_self_consistency.py), so its
debugging behaviour differs from the main Code-Based treatment.

Results are written to data/results/code_based_codex_pilot/, separate from the
main study.
"""

import json
import os
from pathlib import Path

from openai import OpenAI

from src.config import RANDOM_SEED, SMTWTP_SIZES
from src.paths import CODEX_DIRS
from src.chatgpt.prompts.main_prompts import build_code_based_prompt_siclsp, build_code_based_prompt_smtwtp
from src.chatgpt.code_based_self_consistency import run_code_based_solution
from src.evaluation.metrics import compute_optimality_gap
from src.instances.generate_siclsp import SICLSP_SIZES, generate_siclsp_instance
from src.instances.generate_smtwtp import generate_smtwtp_instance
from src.models.siclsp_model import solve_siclsp
from src.models.smtwtp_model import solve_smtwtp

CODEX_MODEL = "gpt-5.3-codex"
CODEX_REASONING_EFFORT = "medium"



def _pilot_instance(problem: str, size: int) -> dict:
    """Erzeugt exakt dieselbe Instanz #0 wie die Hauptstudie (identischer Seed)."""
    seed = RANDOM_SEED + size * 1000 + 0
    if problem == "smtwtp":
        return generate_smtwtp_instance(size, seed)
    return generate_siclsp_instance(size, seed)  # problem=="siclsp"


def _result_path(problem: str, size: int) -> Path:
    out_dir = CODEX_DIRS[problem]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{problem}_size{size}_codex.json"


def run_codex_pilot() -> list:
    """Führt den Codex-Pilot durch (idempotent) und gibt eine Ergebnisliste zurück."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    summary = []

    for problem, sizes in (("smtwtp", SMTWTP_SIZES), ("siclsp", SICLSP_SIZES)):
        for size in sizes:
            out_path = _result_path(problem, size)

            if out_path.exists():
                with open(out_path) as f:
                    record = json.load(f)
                print(f"[{problem} size={size}] bereits vorhanden (übersprungen, kein Gurobi-Solve nötig)")
            else:
                instance = _pilot_instance(problem, size)

                if problem == "smtwtp":
                    gt = solve_smtwtp(instance["jobs"], time_limit=300)
                else:
                    gt = solve_siclsp(instance["periods"], instance["capacity"], time_limit=300)

                prompt = (build_code_based_prompt_smtwtp(instance) if problem == "smtwtp"
                          else build_code_based_prompt_siclsp(instance))
                code_result = run_code_based_solution(
                    client, prompt, problem, instance,
                    model=CODEX_MODEL, reasoning_effort=CODEX_REASONING_EFFORT,
                )
                record = {
                    "problem": problem,
                    "size": size,
                    "model": CODEX_MODEL,
                    "reasoning_effort": CODEX_REASONING_EFFORT,
                    "success": code_result["success"],
                    "n_debug_iterations": code_result["n_debug_iterations"],
                    "verified_objective": code_result["verified_objective"],
                    "gurobi_optimal": gt["objective"],
                    "final_code": code_result["final_code"],
                }
                with open(out_path, "w") as f:
                    json.dump(record, f, indent=2)
                print(f"[{problem} size={size}] Erfolg={record['success']}, "
                      f"Debug-Iterationen={record['n_debug_iterations']}")

            gap = None
            if record.get("verified_objective") is not None and record.get("gurobi_optimal"):
                gap = compute_optimality_gap(record["verified_objective"], record["gurobi_optimal"])
            summary.append({**record, "gap_percent": gap})

    _print_summary(summary)
    return summary


def _print_summary(summary: list) -> None:
    print("\n=== Zusammenfassung Codex-Pilot ===")
    header = f"{'Problem':<8} {'Größe':<6} {'Erfolg':<8} {'Debug-It.':<10} {'Codex-Obj':<12} {'Gurobi-Opt':<12} {'Gap %':<8}"
    print(header)
    print("-" * len(header))
    for r in summary:
        gap_str = f"{r['gap_percent']:.1f}" if r["gap_percent"] is not None else "n/a"
        print(f"{r['problem']:<8} {r['size']:<6} {str(r['success']):<8} "
              f"{r['n_debug_iterations']:<10} {str(r.get('verified_objective')):<12} "
              f"{str(r.get('gurobi_optimal')):<12} {gap_str:<8}")


if __name__ == "__main__":
    run_codex_pilot()
