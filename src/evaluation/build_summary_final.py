"""
Finale, kombinierte Auswertungstabelle über BEIDE Probleme und alle drei
Treatments des korrigierten 1500-Lauf-Experiments (5,10,15,20,25 Größen,
10 Wiederholungen).

Liest AUSSCHLIESSLICH aus den sechs finalen Ergebnisverzeichnissen:
    data/results/direct_smtwtp_final/smtwtp/       (SMTWTP Plain Direct)
    data/results/direct_cot/smtwtp/                (SMTWTP Direct-CoT)
    data/results/code_based_debug5/smtwtp/         (SMTWTP Code-Based Debug5)
    data/results/direct_siclsp_final/siclsp/       (SICLSP Plain Direct)
    data/results/direct_cot_siclsp_final/siclsp/   (SICLSP Direct-CoT)
    data/results/code_based_debug5_siclsp_final/siclsp/ (SICLSP Code-Based Debug5)

Liest NIEMALS aus data/archive/ oder data/test_runs/. Nutzt dieselbe
Metrik-Hierarchie wie build_summary_siclsp.py / build_summary_conditional.py
(evaluation/metrics.py, UNVERÄNDERT — siehe Auftrag "Metriken
bleiben unverändert").

Ausführen mit:
    python -m src.evaluation.build_summary_final
"""

import json

from src.paths import APPROACH_LABELS, GROUND_TRUTH_DIRS, RESULT_DIRS
from src.evaluation.metrics import (
    aggregate_results,
    check_feasibility_smtwtp,
    compute_optimality_gap,
    compute_smtwtp_objective,
)
from src.instances.loader import load_all_instances as load_all_smtwtp_instances
from src.evaluation.build_summary_siclsp import build_summary_siclsp

SMTWTP_APPROACH_DIRS = {APPROACH_LABELS[t]: RESULT_DIRS[(t, "smtwtp")]
                        for t in ("direct", "direct_cot", "code_based")}


def _load_smtwtp_ground_truth() -> dict:
    gt = {}
    gt_dir = GROUND_TRUTH_DIRS["smtwtp"]
    if not gt_dir.exists():
        return gt
    for f in gt_dir.glob("*.json"):
        record = json.loads(f.read_text())
        gt[record["instance_id"]] = record
    return gt


def _extract_smtwtp_solution(approach: str, raw: dict):
    if approach == "code_based_debug5":
        parsed = raw.get("parsed_output")
        if not parsed:
            return None, None
        return parsed.get("sequence"), parsed.get("objective_value")

    from src.chatgpt.run_code_based import extract_json_from_text
    raw_response = raw.get("raw_response")
    if not raw_response:
        return None, None
    try:
        parsed = extract_json_from_text(raw_response)
    except ValueError:
        return None, None
    return parsed.get("sequence"), parsed.get("objective_value")


def build_summary_smtwtp_final() -> list:
    ground_truth = _load_smtwtp_ground_truth()
    instances = {item["instance_id"]: item["data"] for item in load_all_smtwtp_instances()
                 if item["problem"] == "smtwtp"}

    records = []
    for approach, approach_dir in SMTWTP_APPROACH_DIRS.items():
        if not approach_dir.exists():
            continue
        for f in approach_dir.glob("*.json"):
            raw = json.loads(f.read_text())
            instance_id = raw["instance_id"]
            gt = ground_truth.get(instance_id)
            instance = instances.get(instance_id)
            if gt is None or instance is None:
                continue

            solution, reported_obj = _extract_smtwtp_solution(approach, raw)
            technical_success = solution is not None

            feasible = False
            recomputed_obj = None
            if technical_success:
                feasible = check_feasibility_smtwtp(solution, instance)
                if feasible:
                    recomputed_obj = compute_smtwtp_objective(solution, instance)

            gap = None
            if recomputed_obj is not None and gt["objective"] is not None:
                gap = compute_optimality_gap(recomputed_obj, gt["objective"])

            records.append({
                "problem": "smtwtp",
                "approach": approach,
                "instance_id": instance_id,
                "size": instance["n_jobs"],
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


def build_and_save_summary_final() -> dict:
    records = build_summary_smtwtp_final() + build_summary_siclsp()

    summary_path = RESULT_DIRS[("direct", "smtwtp")].parents[1] / "summary_final.json"
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"{len(records)} verknüpfte Records (SMTWTP+SICLSP, final) gespeichert in {summary_path}")

    if not records:
        print("Keine Records gefunden — noch keine LLM-Ergebnisse für das finale 1500-Lauf-Experiment.")
        return {}

    aggregated = aggregate_results(records)
    print()
    print("=== FINAL (SMTWTP + SICLSP): Aggregierte Ergebnisse ===")
    header = (f"{'Problem':<8} {'Ansatz':<20} {'Größe':<6} {'n':<4} {'Tech. Success %':<16} "
              f"{'Feasibility %':<14} {'Overall Feas. %':<16} {'Ø Gap %':<10}")
    print(header)
    print("-" * len(header))
    for (problem, approach, size), stats in sorted(aggregated.items()):
        tech = f"{stats['technical_success_rate']*100:.0f}" if stats["technical_success_rate"] is not None else "n/a"
        feas = f"{stats['feasibility_rate']*100:.0f}" if stats["feasibility_rate"] is not None else "n/a"
        overall = f"{stats['overall_feasible_rate']*100:.0f}" if stats["overall_feasible_rate"] is not None else "n/a"
        gap = f"{stats['mean_optimality_gap']:.1f}" if stats["mean_optimality_gap"] is not None else "n/a"
        print(f"{problem:<8} {approach:<20} {size:<6} {stats['n']:<4} {tech:<16} {feas:<14} {overall:<16} {gap:<10}")

    return aggregated


if __name__ == "__main__":
    build_and_save_summary_final()
