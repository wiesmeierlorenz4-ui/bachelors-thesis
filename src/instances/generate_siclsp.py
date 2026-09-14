"""
Instance generator for the SICLSP (Single-Item Capacitated Lot-Sizing Problem).

Parameter ranges per period:
    demand        ~ Uniform[10, 50]
    setup cost    ~ Uniform[100, 500]
    holding cost  ~ Uniform[1, 5]
    production cost = 0

Capacity is period-specific and scaled to that period's demand:

    multiplier_t   ~ Uniform[0.7, 1.6]
    raw_capacity_t = max(1, round(multiplier_t * demand_t))

followed by a deterministic prefix repair that adds the exact deficit wherever
cumulative capacity would fall below cumulative demand, so every instance is
feasible by construction while some periods stay individually binding. The
capacity rule is an experimental design decision of this thesis, not a rule
taken from the literature.

Seeds follow RANDOM_SEED + size * 1000 + instance_index, so the instances are
exactly reproducible.
"""

import json
import random

from src.config import (
    INSTANCES_DIR,
    RANDOM_SEED,
    SICLSP_SIZES,
    SICLSP_INSTANCES_PER_SIZE,
)
from src.evaluation.siclsp_validation import validate_siclsp_instance

# Size grid and instance count come from src/config.py — single source of
# truth shared with SMTWTP (both problems use 5/10/15/20/25 x 5 instances).
# Re-exported here so that existing imports of
# `from src.instances.generate_siclsp import SICLSP_SIZES` keep working.

DEMAND_LOW, DEMAND_HIGH = 10, 50
SETUP_COST_LOW, SETUP_COST_HIGH = 100, 500
HOLDING_COST_LOW, HOLDING_COST_HIGH = 1, 5

CAPACITY_MULTIPLIER_LOW, CAPACITY_MULTIPLIER_HIGH = 0.7, 1.6

GENERATOR_VERSION = "siclsp_variable_capacity_v1"

_MAX_DEGENERACY_ATTEMPTS = 50  # rein deterministisch, siehe Docstring oben


def _generate_capacity(rng: random.Random, demand: dict, n_periods: int) -> dict:
    """Zieht periodenabhängige Kapazität und repariert Präfix-Defizite deterministisch."""
    raw_capacity = {}
    for t in range(1, n_periods + 1):
        multiplier = rng.uniform(CAPACITY_MULTIPLIER_LOW, CAPACITY_MULTIPLIER_HIGH)
        raw_capacity[t] = max(1, round(multiplier * demand[t]))

    capacity = dict(raw_capacity)
    cumulative_capacity = 0
    cumulative_demand = 0
    for t in range(1, n_periods + 1):
        cumulative_capacity += capacity[t]
        cumulative_demand += demand[t]
        if cumulative_capacity < cumulative_demand:
            deficit = cumulative_demand - cumulative_capacity
            capacity[t] += deficit
            cumulative_capacity += deficit

    return capacity


def generate_siclsp_instance(n_periods: int, seed: int) -> dict:
    """
    Erzeugt eine einzelne SICLSP-Instanz (periodenabhängige Kapazität) mit
    n_periods Perioden. Validiert die Instanz vor der Rückgabe (siehe
    evaluation/siclsp_validation.py) — wirft RuntimeError statt eine
    ungültige Instanz zurückzugeben (siehe Docstring oben, Abschnitt
    KAPAZITÄTSERZEUGUNG zur Degenerationsbehandlung).

    Returns:
        dict with "n_periods", "seed", "periods", "capacity" and the
        metadata fields "problem"/"variant"/"generator_version".
    """
    rng = random.Random(seed)

    demand = {t: rng.randint(DEMAND_LOW, DEMAND_HIGH) for t in range(1, n_periods + 1)}
    setup_cost = {t: rng.randint(SETUP_COST_LOW, SETUP_COST_HIGH) for t in range(1, n_periods + 1)}
    holding_cost = {t: rng.randint(HOLDING_COST_LOW, HOLDING_COST_HIGH) for t in range(1, n_periods + 1)}
    prod_cost = {t: 0 for t in range(1, n_periods + 1)}

    capacity = _generate_capacity(rng, demand, n_periods)
    for attempt in range(_MAX_DEGENERACY_ATTEMPTS):
        if n_periods <= 1 or len(set(capacity.values())) > 1:
            break
        # Deterministisch: derselbe rng-Zustand wird weiter verbraucht, kein neuer Seed.
        capacity = _generate_capacity(rng, demand, n_periods)
    else:
        raise RuntimeError(
            f"Konnte nach {_MAX_DEGENERACY_ATTEMPTS} deterministischen Versuchen keine "
            f"periodenabhängige Kapazität für n_periods={n_periods}, seed={seed} erzeugen."
        )

    instance = {
        "n_periods": n_periods,
        "seed": seed,
        "problem": "siclsp",
        "variant": "variable_capacity",
        "generator_version": GENERATOR_VERSION,
        "periods": {
            t: {
                "demand": demand[t],
                "setup_cost": setup_cost[t],
                "prod_cost": prod_cost[t],
                "holding_cost": holding_cost[t],
            }
            for t in range(1, n_periods + 1)
        },
        "capacity": capacity,
    }

    errors = validate_siclsp_instance(instance, expected_n_periods=n_periods)
    if errors:
        raise RuntimeError(
            f"Generierte SICLSP-Instanz (n_periods={n_periods}, seed={seed}) ist ungültig, "
            f"wird NICHT gespeichert. Fehler: {errors}"
        )

    return instance


def generate_all_siclsp_instances() -> list:
    """
    Generiert alle Instanzen gemäß SICLSP_SIZES / SICLSP_INSTANCES_PER_SIZE,
    writes them to data/instances/siclsp_variable_capacity/ and returns the
    list of created file paths.
    """
    out_dir = INSTANCES_DIR / "siclsp_variable_capacity"
    out_dir.mkdir(parents=True, exist_ok=True)

    created_files = []
    for size in SICLSP_SIZES:
        for idx in range(SICLSP_INSTANCES_PER_SIZE):
            seed = RANDOM_SEED + size * 1000 + idx  # identische Konvention wie generate_silsp.py
            instance = generate_siclsp_instance(size, seed)

            out_path = out_dir / f"siclsp_T{size}_inst{idx}.json"
            with open(out_path, "w") as f:
                json.dump(instance, f, indent=2)
            created_files.append(out_path)

    return created_files


if __name__ == "__main__":
    files = generate_all_siclsp_instances()
    print(f"{len(files)} SICLSP-Instanzen (variable Kapazität) erzeugt in "
          f"{INSTANCES_DIR / 'siclsp_variable_capacity'}")
