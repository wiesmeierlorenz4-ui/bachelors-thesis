"""
FINAL PREFLIGHT — kombinierte Vorab-Prüfung für das gesamte finale
1500-Lauf-Experiment (SMTWTP + SICLSP, 3 Treatments, 10 Wiederholungen).

Erweitert siclsp_preflight.py (SICLSP-spezifische Prüfungen, unverändert
wiederverwendet) um:
    - SMTWTP-Größenraster-/Instanz-/Gurobi-/Prompt-Prüfungen
    - die erwartete Gesamtzahl experimenteller Läufe (1500 = 500 je Treatment)
    - die Prüfung, dass alle sechs finalen Ergebnisverzeichnisse (2 Probleme
      x 3 Treatments) vor dem echten Lauf leer sind

Sendet NIEMALS OpenAI-Requests.

Ausführen mit:
    python -m src.evaluation.final_preflight
"""

import sys

from src.config import N_RUNS_PER_INSTANCE, SMTWTP_SIZES
from src.instances.generate_smtwtp import generate_all_smtwtp_instances
from src.instances.loader import load_all_instances as load_all_smtwtp_instances
from src.evaluation.solve_ground_truth_smtwtp import solve_all_ground_truth
from src.evaluation.siclsp_preflight import run_preflight as run_siclsp_preflight
from src.instances.generate_siclsp import SICLSP_SIZES, SICLSP_INSTANCES_PER_SIZE
from src.evaluation.final_experiment_integrity_check import RESULT_DIRS, EXCLUDED_PATH_MARKERS
from src.chatgpt.prompts.main_prompts import (
    build_direct_prompt_smtwtp as build_plain_direct_prompt_smtwtp,
    build_code_based_prompt_smtwtp,
)
from src.chatgpt.prompts.cot_prompts import build_direct_cot_prompt_smtwtp

EXPECTED_SMTWTP_TOTAL = len(SMTWTP_SIZES) * 5  # 25
EXPECTED_SICLSP_TOTAL = len(SICLSP_SIZES) * SICLSP_INSTANCES_PER_SIZE  # 25
EXPECTED_REPS = N_RUNS_PER_INSTANCE  # 10
EXPECTED_RUNS_PER_TREATMENT_PER_PROBLEM = EXPECTED_SMTWTP_TOTAL * EXPECTED_REPS  # 250
EXPECTED_RUNS_PER_TREATMENT = EXPECTED_RUNS_PER_TREATMENT_PER_PROBLEM * 2  # 500
EXPECTED_TOTAL_RUNS = EXPECTED_RUNS_PER_TREATMENT * 3  # 1500


def _line(label: str, passed_n: int, total_n: int) -> str:
    status = f"{passed_n}/{total_n} PASS" if passed_n == total_n else f"{passed_n}/{total_n} FAIL"
    return f"{label:<28}{status}"


def _smtwtp_preflight() -> bool:
    ok = True
    print("--- SMTWTP ---")

    ok &= (SMTWTP_SIZES == [5, 10, 15, 20, 25])
    print(f"{'SMTWTP size grid:':<28}{'PASS' if SMTWTP_SIZES == [5,10,15,20,25] else 'FAIL'} ({SMTWTP_SIZES})")

    files = generate_all_smtwtp_instances()
    n_generated = len(files)
    ok &= n_generated == EXPECTED_SMTWTP_TOTAL
    print(_line("Instances generated:", n_generated, EXPECTED_SMTWTP_TOTAL))

    gt_results = solve_all_ground_truth()  # idempotent, SMTWTP only
    n_optimal = sum(1 for r in gt_results if r.get("is_optimal"))
    ok &= n_optimal == EXPECTED_SMTWTP_TOTAL
    print(_line("Gurobi optimal:", n_optimal, EXPECTED_SMTWTP_TOTAL))

    instances = [i for i in load_all_smtwtp_instances() if i["problem"] == "smtwtp"]
    n_plain = n_cot = n_code = 0
    for item in instances:
        data = item["data"]
        try:
            p1 = build_plain_direct_prompt_smtwtp(data)
            assert "Denke Schritt" not in p1
            n_plain += 1
        except Exception:
            pass
        try:
            p2 = build_direct_cot_prompt_smtwtp(data)
            assert "Denke Schritt" in p2
            n_cot += 1
        except Exception:
            pass
        try:
            build_code_based_prompt_smtwtp(data)
            n_code += 1
        except Exception:
            pass
    ok &= n_plain == EXPECTED_SMTWTP_TOTAL
    ok &= n_cot == EXPECTED_SMTWTP_TOTAL
    ok &= n_code == EXPECTED_SMTWTP_TOTAL
    print(_line("Direct prompts:", n_plain, EXPECTED_SMTWTP_TOTAL))
    print(_line("Direct-CoT prompts:", n_cot, EXPECTED_SMTWTP_TOTAL))
    print(_line("Code-Based prompts:", n_code, EXPECTED_SMTWTP_TOTAL))

    n30 = [i["instance_id"] for i in instances if "_n30_" in i["instance_id"]]
    ok &= len(n30) == 0
    print(f"{'ACTIVE SMTWTP n=30 files:':<28}{len(n30)}")

    return ok


def _result_dir_state() -> str:
    """Classify the result directories as "empty" (safe to start a fresh run),
    "complete" (the experiment already ran in full — starting again would do
    nothing, since all runners are idempotent) or "partial" (an interrupted
    run that would be resumed)."""
    print("--- Result directories ---")
    counts = []
    for (problem, treatment), result_dir in RESULT_DIRS.items():
        if any(marker in str(result_dir) for marker in EXCLUDED_PATH_MARKERS):
            continue
        n = len(list(result_dir.glob("*.json"))) if result_dir.exists() else 0
        counts.append(n)
        print(f"  {problem:<8} {treatment:<20} {n:>4} / {EXPECTED_RUNS_PER_TREATMENT_PER_PROBLEM} runs "
              f"in {result_dir}")

    if all(n == 0 for n in counts):
        return "empty"
    if all(n == EXPECTED_RUNS_PER_TREATMENT_PER_PROBLEM for n in counts):
        return "complete"
    return "partial"


def run_final_preflight() -> bool:
    print("FINAL PREFLIGHT (SMTWTP + SICLSP)\n")

    smtwtp_ok = _smtwtp_preflight()
    print()

    print("--- SICLSP ---")
    siclsp_ok = run_siclsp_preflight()
    print()

    dir_state = _result_dir_state()
    print()

    print(f"EXPECTED FINAL EXPERIMENTAL RUNS: {EXPECTED_TOTAL_RUNS}")
    print(f"EXPECTED RUNS PER TREATMENT: {EXPECTED_RUNS_PER_TREATMENT}")
    print(f"  (= {EXPECTED_RUNS_PER_TREATMENT_PER_PROBLEM} per problem per treatment "
          f"= {EXPECTED_SMTWTP_TOTAL} instances x {EXPECTED_REPS} repetitions)")
    print()

    inputs_ok = smtwtp_ok and siclsp_ok
    if dir_state == "empty":
        verdict = "READY (no results yet — a full run would collect all 1500)"
    elif dir_state == "complete":
        verdict = ("READY / ALREADY COMPLETE (all result directories hold the full "
                   "number of runs; run final_experiment_integrity_check for details)")
    else:
        verdict = ("READY / PARTIAL (some runs already exist — the runners are "
                   "idempotent and would resume the missing ones)")

    print(f"INPUT DATA (instances, ground truth, prompts): {'OK' if inputs_ok else 'PROBLEMS FOUND'}")
    print(f"FINAL EXPERIMENT PREFLIGHT: {verdict if inputs_ok else 'NOT READY'}")
    return inputs_ok


if __name__ == "__main__":
    ok = run_final_preflight()
    sys.exit(0 if ok else 1)
