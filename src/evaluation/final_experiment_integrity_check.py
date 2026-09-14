"""
Completeness check for the collected experimental runs.

For every problem, size, instance and treatment there must be exactly
N_RUNS_PER_INSTANCE repetition ids. Missing, duplicate or out-of-range
repetitions are reported, as are instance ids that do not belong to the final
dataset.
"""

import re
import sys

from src.config import N_RUNS_PER_INSTANCE
from src.paths import APPROACH_LABELS, RESULT_DIRS as TREATMENT_RESULT_DIRS
from src.instances.loader import load_all_instances as load_all_smtwtp_instances
from src.evaluation.solve_ground_truth_siclsp import load_siclsp_variable_capacity_instances

EXPECTED_REPS = N_RUNS_PER_INSTANCE  # aktuell 10, siehe config.py

# (problem, approach label) -> Ergebnisverzeichnis der finalen Läufe.
RESULT_DIRS = {(problem, APPROACH_LABELS[t]): TREATMENT_RESULT_DIRS[(t, problem)]
               for t in ("direct", "direct_cot", "code_based")
               for problem in ("smtwtp", "siclsp")}

# Verzeichnisse, die NIEMALS als finale Ergebnisse gezählt werden dürfen.
EXCLUDED_PATH_MARKERS = ("data/archive", "data/test_runs", "/archive/", "/test_runs/")

FILENAME_PATTERN = re.compile(r"^(?P<instance_id>.+)_run(?P<run>\d+)\.json$")
SIZE_PATTERN = re.compile(r"(?:_n|_T)(\d+)_inst")


def _expected_instance_ids() -> dict:
    smtwtp_ids = {item["instance_id"] for item in load_all_smtwtp_instances()
                  if item["problem"] == "smtwtp"}
    siclsp_ids = {item["instance_id"] for item in load_siclsp_variable_capacity_instances()}
    return {"smtwtp": smtwtp_ids, "siclsp": siclsp_ids}


def _check_one(problem: str, treatment: str, result_dir, expected_ids: set) -> dict:
    report = {
        "problem": problem, "treatment": treatment, "dir": str(result_dir),
        "found": 0, "expected": len(expected_ids) * EXPECTED_REPS,
        "errors": [],
    }

    if any(marker in str(result_dir) for marker in EXCLUDED_PATH_MARKERS):
        report["errors"].append(f"result_dir points at an excluded path: {result_dir}")
        return report

    if not result_dir.exists():
        report["errors"].append("directory does not exist (0 runs so far)")
        return report

    per_instance_runs: dict = {iid: set() for iid in expected_ids}
    extra_instance_ids = set()

    for f in sorted(result_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue  # z.B. _pending_batch_id.txt-Marker, _batch_requests.jsonl
        match = FILENAME_PATTERN.match(f.name)
        if not match:
            report["errors"].append(f"unrecognized filename: {f.name}")
            continue

        instance_id = match.group("instance_id")
        run_number = int(match.group("run"))
        report["found"] += 1

        size_match = SIZE_PATTERN.search(instance_id)
        if size_match and int(size_match.group(1)) == 30:
            report["errors"].append(f"REJECTED: size=30 instance present ({f.name})")
            continue
        if instance_id.startswith("silsp_"):  # instance id from an archived dataset
            report["errors"].append(f"REJECTED: archived 'silsp_' instance present ({f.name})")
            continue

        if instance_id not in per_instance_runs:
            extra_instance_ids.add(instance_id)
            continue

        if run_number < 1 or run_number > EXPECTED_REPS:
            report["errors"].append(f"run number out of range [1,{EXPECTED_REPS}]: {f.name}")
            continue
        if run_number in per_instance_runs[instance_id]:
            report["errors"].append(f"DUPLICATE repetition: {f.name}")
            continue
        per_instance_runs[instance_id].add(run_number)

    if extra_instance_ids:
        report["errors"].append(f"unexpected instance_ids present: {sorted(extra_instance_ids)}")

    for iid, runs in per_instance_runs.items():
        missing = set(range(1, EXPECTED_REPS + 1)) - runs
        if missing:
            report["errors"].append(f"{iid}: missing repetitions {sorted(missing)}")

    return report


def run_integrity_check() -> bool:
    expected_ids = _expected_instance_ids()
    all_ok = True
    totals = {"plain_direct": 0, "direct_cot": 0, "code_based_debug5": 0}
    expected_totals = {"plain_direct": 0, "direct_cot": 0, "code_based_debug5": 0}

    print("=== FINAL EXPERIMENT INTEGRITY CHECK ===\n")
    for (problem, treatment), result_dir in RESULT_DIRS.items():
        report = _check_one(problem, treatment, result_dir, expected_ids[problem])
        ok = len(report["errors"]) == 0 and report["found"] == report["expected"]
        all_ok &= ok
        totals[treatment] += report["found"]
        expected_totals[treatment] += report["expected"]
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {problem:<8} {treatment:<20} {report['found']:>4}/{report['expected']:<4}  ({result_dir})")
        for e in report["errors"][:10]:
            print(f"        - {e}")
        if len(report["errors"]) > 10:
            print(f"        ... and {len(report['errors']) - 10} more errors")

    print()
    grand_total = sum(totals.values())
    grand_expected = sum(expected_totals.values())
    for treatment in ("plain_direct", "direct_cot", "code_based_debug5"):
        print(f"{treatment:<20} {totals[treatment]:>5}/{expected_totals[treatment]:<5}")
    print(f"{'TOTAL':<20} {grand_total:>5}/{grand_expected:<5}")

    print()
    print(f"INTEGRITY CHECK: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    ok = run_integrity_check()
    sys.exit(0 if ok else 1)
