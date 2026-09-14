"""
Trusted ground truth for SICLSP.

Solves every SICLSP instance with the hand-written benchmark formulation
(models/siclsp_model.py) and stores the proven optimum used as the reference
value in the evaluation. MIPGap 0.0, TimeLimit 300 s; only Gurobi status
OPTIMAL is accepted.

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
from src.paths import GROUND_TRUTH_DIRS, INSTANCE_DIRS
from src.models.siclsp_model import solve_siclsp

logger = logging.getLogger(__name__)

SICLSP_INSTANCES_DIR = INSTANCE_DIRS["siclsp"]
SICLSP_GROUND_TRUTH_DIR = GROUND_TRUTH_DIRS["siclsp"]


def load_siclsp_variable_capacity_instances() -> list:
    """
    Load the SICLSP instances.

    Returns:
        list of dicts: {"problem": "siclsp", "instance_id": str, "data": dict}
    """
    instances = []
    if not SICLSP_INSTANCES_DIR.exists():
        return instances
    for f in sorted(SICLSP_INSTANCES_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        instances.append({"problem": "siclsp", "instance_id": f.stem, "data": data})
    return instances


def _result_path(instance_id: str) -> Path:
    out_dir = SICLSP_GROUND_TRUTH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{instance_id}.json"


def solve_and_save_siclsp_instance(instance_id: str, data: dict, env: gp.Env = None) -> dict:
    """Löst eine einzelne SICLSP-Instanz und speichert das Ergebnis."""
    t0 = time.time()
    result = solve_siclsp(data["periods"], data["capacity"],
                          time_limit=GUROBI_TIME_LIMIT, mip_gap=GUROBI_MIP_GAP, env=env)
    wall_time = time.time() - t0

    record = {
        "instance_id": instance_id,
        "problem": "siclsp",
        "variant": "variable_capacity",
        "size": data["n_periods"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gurobi_status": result["status"],
        "is_optimal": result["status"] == GRB.OPTIMAL,
        "objective": result["objective"],
        "gurobi_runtime_seconds": result["runtime_seconds"],
        "wall_time_seconds": wall_time,
        "production": result["production"],
    }

    out_path = _result_path(instance_id)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return record


def solve_all_siclsp_ground_truth() -> list:
    """Löst alle 25 korrigierten SICLSP-Instanzen (idempotent)."""
    instances = load_siclsp_variable_capacity_instances()
    if not instances:
        logger.warning(
            "Keine SICLSP-Instanzen gefunden — zuerst "
            "python -m src.instances.generate_siclsp ausführen."
        )
        return []

    env = gp.Env(empty=False)
    results = []
    non_optimal = []

    for item in instances:
        instance_id, data = item["instance_id"], item["data"]
        out_path = _result_path(instance_id)

        if out_path.exists():
            with open(out_path) as f:
                record = json.load(f)
            logger.info(f"[siclsp] {instance_id}: bereits vorhanden (übersprungen)")
        else:
            logger.info(f"[siclsp] {instance_id}: löse mit Gurobi (TimeLimit={GUROBI_TIME_LIMIT}s)...")
            record = solve_and_save_siclsp_instance(instance_id, data, env=env)
            status_str = "OPTIMAL" if record["is_optimal"] else f"NICHT OPTIMAL (Status {record['gurobi_status']})"
            logger.info(f"[siclsp] {instance_id}: {status_str}, Objective={record['objective']}, "
                        f"Gurobi-Zeit={record['gurobi_runtime_seconds']:.2f}s")

        results.append(record)
        if not record.get("is_optimal", False):
            non_optimal.append(record)

    if non_optimal:
        logger.warning(
            f"⚠️  {len(non_optimal)} von {len(results)} SICLSP-Instanzen NICHT bewiesen optimal "
            f"gelöst. Betroffene Instanzen: {[r['instance_id'] for r in non_optimal]}."
        )
    else:
        logger.info(f"✅ Alle {len(results)} SICLSP-Instanzen bewiesen optimal gelöst.")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    solve_all_siclsp_ground_truth()
