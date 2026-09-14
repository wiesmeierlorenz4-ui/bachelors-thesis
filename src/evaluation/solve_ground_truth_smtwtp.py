"""
TRUSTED GROUND TRUTH for SMTWTP.

Solves every SMTWTP instance with the manually implemented benchmark
formulation (src/models/smtwtp_model.py) and stores the proven optimum used
as the reference value in the evaluation.

This is role A of Gurobi in this project: a formulation written by hand and
trusted as the benchmark. It must not be confused with role B, where Gurobi
executes a formulation that GPT-4o generated in the Code-Based treatment —
there, a successful solve says nothing about whether the generated model
represents the intended problem.

Settings (see config.py): MIPGap = 0.0, TimeLimit = 300 s. Only runs with
Gurobi status OPTIMAL are accepted as benchmark values; anything else is
reported as a warning at the end of the run.

The SICLSP ground truth has its own module: solve_ground_truth_siclsp.py.

Idempotent: existing result files are skipped and never overwritten.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from src.config import GUROBI_MIP_GAP, GUROBI_TIME_LIMIT
from src.paths import GROUND_TRUTH_DIRS
from src.instances.loader import load_instances, parse_size_from_instance_id
from src.models.smtwtp_model import solve_smtwtp

logger = logging.getLogger(__name__)


def _result_path(problem: str, instance_id: str) -> Path:
    out_dir = GROUND_TRUTH_DIRS[problem]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{instance_id}.json"


def solve_and_save_instance(problem: str, instance_id: str, data: dict,
                             env: gp.Env = None) -> dict:
    """Löst eine einzelne Instanz und speichert das Ergebnis. Gibt es zurück."""
    t0 = time.time()

    if problem != "smtwtp":
        raise ValueError(f"This module only solves SMTWTP, got {problem!r} "
                         f"(SICLSP: see solve_ground_truth_siclsp.py)")
    result = solve_smtwtp(data["jobs"], time_limit=GUROBI_TIME_LIMIT,
                          mip_gap=GUROBI_MIP_GAP, env=env)

    wall_time = time.time() - t0

    record = {
        "instance_id": instance_id,
        "problem": problem,
        "size": parse_size_from_instance_id(instance_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gurobi_status": result["status"],
        "is_optimal": result["status"] == GRB.OPTIMAL,
        "objective": result["objective"],
        "gurobi_runtime_seconds": result["runtime_seconds"],
        "wall_time_seconds": wall_time,
    }
    record["sequence"] = result["sequence"]

    out_path = _result_path(problem, instance_id)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return record


def solve_all_ground_truth() -> list:
    """
    Löst alle Instanzen aus data/instances/ mit Gurobi (idempotent) und gibt
    die Liste aller Ergebnis-Records zurück (auch die bereits vorher
    vorhandenen, aus den gespeicherten Dateien geladen).
    """
    instances = load_instances("smtwtp")
    if not instances:
        logger.warning("Keine Instanzen gefunden — zuerst generate_smtwtp/generate_silsp ausführen.")
        return []

    env = gp.Env(empty=False)  # ein Environment für alle Runs wiederverwenden
    results = []
    non_optimal = []

    for item in instances:
        problem, instance_id, data = item["problem"], item["instance_id"], item["data"]
        out_path = _result_path(problem, instance_id)

        if out_path.exists():
            with open(out_path) as f:
                record = json.load(f)
            logger.info(f"[{problem}] {instance_id}: bereits vorhanden (übersprungen)")
        else:
            logger.info(f"[{problem}] {instance_id}: löse mit Gurobi (TimeLimit={GUROBI_TIME_LIMIT}s)...")
            record = solve_and_save_instance(problem, instance_id, data, env=env)
            status_str = "OPTIMAL" if record["is_optimal"] else f"NICHT OPTIMAL (Status {record['gurobi_status']})"
            logger.info(f"[{problem}] {instance_id}: {status_str}, "
                        f"Objective={record['objective']}, "
                        f"Gurobi-Zeit={record['gurobi_runtime_seconds']:.2f}s")

        results.append(record)
        if not record.get("is_optimal", False):
            non_optimal.append(record)

    if non_optimal:
        logger.warning(
            f"⚠️  {len(non_optimal)} von {len(results)} Instanzen wurden NICHT als "
            f"bewiesen optimal gelöst (TimeLimit erreicht oder anderer Status)! "
            f"Betroffene Instanzen: {[r['instance_id'] for r in non_optimal]}. "
            f"Das ist ein starkes Signal, dass die aktuelle Formulierung für diese "
            f"Größenordnung ggf. nicht ausreicht — siehe README 'Offene Punkte'."
        )
    else:
        logger.info(f"✅ Alle {len(results)} Instanzen bewiesen optimal gelöst.")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    solve_all_ground_truth()
