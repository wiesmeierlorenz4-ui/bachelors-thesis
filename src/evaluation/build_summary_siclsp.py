"""
Links SICLSP ground truth and LLM results into one evaluation table.

Feasibility, objective recomputation, optimality gap and consistency all come
from evaluation/metrics.py, so the numbers match the rest of the analysis.
"""

import json

from src.paths import APPROACH_LABELS, RESULT_DIRS
from src.evaluation.metrics import (
    aggregate_results,
    check_feasibility_siclsp,
    compute_consistency,
    compute_optimality_gap,
    compute_siclsp_objective,
)
from src.evaluation.solve_ground_truth_siclsp import (
    SICLSP_GROUND_TRUTH_DIR,
    load_siclsp_variable_capacity_instances,
)

APPROACH_DIRS = {APPROACH_LABELS[t]: RESULT_DIRS[(t, "siclsp")]
                 for t in ("direct", "direct_cot", "code_based")}


def _load_ground_truth() -> dict:
    gt = {}
    if not SICLSP_GROUND_TRUTH_DIR.exists():
        return gt
    for f in SICLSP_GROUND_TRUTH_DIR.glob("*.json"):
        with open(f) as fh:
            record = json.load(fh)
        gt[record["instance_id"]] = record
    return gt


def _load_raw_results(approach: str) -> list:
    results = []
    approach_dir = APPROACH_DIRS[approach]
    if not approach_dir.exists():
        return results
    for f in approach_dir.glob("*.json"):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def _extract_solution(approach: str, raw: dict):
    """Analog zu build_summary_conditional._extract_solution, aber nur für
    die drei SICLSP-Treatment-Arme."""
    if approach == "code_based_debug5":
        parsed = raw.get("parsed_output")
        if not parsed:
            return None, None
        return parsed.get("production"), parsed.get("objective_value")

    from src.chatgpt.run_code_based import extract_json_from_text
    raw_response = raw.get("raw_response")
    if not raw_response:
        return None, None
    try:
        parsed = extract_json_from_text(raw_response)
    except ValueError:
        return None, None
    return parsed.get("production"), parsed.get("objective_value")


def build_summary_siclsp(approaches: tuple = ("plain_direct", "direct_cot", "code_based_debug5")) -> list:
    ground_truth = _load_ground_truth()
    instances = {item["instance_id"]: item["data"] for item in load_siclsp_variable_capacity_instances()}

    records = []
    for approach in approaches:
        for raw in _load_raw_results(approach):
            instance_id = raw["instance_id"]
            gt = ground_truth.get(instance_id)
            instance = instances.get(instance_id)

            if gt is None or instance is None:
                continue

            solution, reported_obj = _extract_solution(approach, raw)
            technical_success = solution is not None

            feasible = False
            recomputed_obj = None
            if technical_success:
                feasible = check_feasibility_siclsp(solution, instance)
                if feasible:
                    recomputed_obj = compute_siclsp_objective(solution, instance)

            gap = None
            if recomputed_obj is not None and gt["objective"] is not None:
                gap = compute_optimality_gap(recomputed_obj, gt["objective"])

            records.append({
                "problem": "siclsp",
                "variant": "variable_capacity",
                "approach": approach,
                "instance_id": instance_id,
                "size": instance["n_periods"],
                "run_number": raw.get("run_number"),
                "technical_success": technical_success,
                "feasible": feasible,
                "reported_objective": reported_obj,
                "recomputed_objective": recomputed_obj,
                "gurobi_optimal": gt["objective"],
                "optimality_gap": gap,
                "n_debug_iterations": raw.get("n_debug_iterations"),
            })

    return records


def build_consistency_table_siclsp(records: list) -> dict:
    grouped: dict = {}
    for r in records:
        key = (r["approach"], r["instance_id"])
        grouped.setdefault(key, {"objectives": [], "technical_success": []})
        grouped[key]["objectives"].append(r["recomputed_objective"])
        grouped[key]["technical_success"].append(r["technical_success"])

    return {
        key: compute_consistency(values["objectives"], technical_success_flags=values["technical_success"])
        for key, values in grouped.items()
    }


def build_and_save_summary_siclsp() -> dict:
    records = build_summary_siclsp()

    summary_path = RESULT_DIRS[("direct", "siclsp")].parents[1] / "summary_siclsp_final.json"
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"{len(records)} verknüpfte SICLSP-Records gespeichert in {summary_path}")

    if not records:
        print("Keine Records gefunden — noch keine LLM-Ergebnisse für die korrigierte SICLSP-Studie.")
        return {}

    # aggregate_results erwartet (problem, approach, size) als Gruppierungsschlüssel;
    # "problem" ist hier für alle Records konstant "siclsp".
    aggregated = aggregate_results(records)
    print()
    print("=== SICLSP (variable capacity): Aggregierte Ergebnisse ===")
    header = (f"{'Ansatz':<20} {'Größe':<6} {'n':<4} {'Tech. Success %':<16} "
              f"{'Feasibility %':<14} {'Overall Feas. %':<16} {'Ø Gap %':<10}")
    print(header)
    print("-" * len(header))
    for (_problem, approach, size), stats in sorted(aggregated.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        tech = f"{stats['technical_success_rate']*100:.0f}" if stats["technical_success_rate"] is not None else "n/a"
        feas = f"{stats['feasibility_rate']*100:.0f}" if stats["feasibility_rate"] is not None else "n/a"
        overall = f"{stats['overall_feasible_rate']*100:.0f}" if stats["overall_feasible_rate"] is not None else "n/a"
        gap = f"{stats['mean_optimality_gap']:.1f}" if stats["mean_optimality_gap"] is not None else "n/a"
        print(f"{approach:<20} {size:<6} {stats['n']:<4} {tech:<16} {feas:<14} {overall:<16} {gap:<10}")

    return aggregated


if __name__ == "__main__":
    build_and_save_summary_siclsp()
