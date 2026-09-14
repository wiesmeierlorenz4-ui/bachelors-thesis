"""
Instanzgenerator für das SMTWTP (Problem 1: Scheduling).

Wissenschaftliche Grundlage:
    Verwendet das in der Scheduling-Literatur etablierte Due-Date-Generierungs-
    schema über zwei Steuergrößen: Tardiness-Faktor τ (tau) und Range-Faktor R.
    Dieses Schema geht auf Baker (1974) zurück und wird u.a. bei
    Potts & Van Wassenhove (1985) für SMTWTP-Benchmarkinstanzen verwendet
    (siehe literaturliste.md, Abschnitt 1) — dadurch sind die generierten
    Instanzen mit denen der Standardreferenz methodisch vergleichbar, statt
    rein willkürlich generiert zu sein.

    p_j ~ Uniform[1, 100]                              (Bearbeitungszeiten)
    d_j ~ Uniform[P*(1-τ-R/2), P*(1-τ+R/2)]             (Fälligkeitsdaten)
    w_j ~ Uniform[1, 10]                                (Gewichte)
    mit P = Summe aller p_j.

    τ = 0.6, R = 0.6 sind in der Literatur gängige Werte für mittelschwere
    Instanzen (weder trivial früh noch trivial spät fällig) und werden hier
    fix verwendet, um Instanzgröße als einzige variierende Komplexitätsdimension
    zu isolieren (wichtig für die interne Validität des Größenvergleichs).
"""

import json
import random

from src.config import (
    INSTANCES_DIR,
    RANDOM_SEED,
    SMTWTP_SIZES,
    SMTWTP_INSTANCES_PER_SIZE,
)

TARDINESS_FACTOR = 0.6   # τ — Anteil der Jobs, die tendenziell verspätet sein sollen
RANGE_FACTOR = 0.6       # R — Streuung der Fälligkeitsdaten


def generate_smtwtp_instance(n_jobs: int, seed: int,
                              tau: float = TARDINESS_FACTOR,
                              r: float = RANGE_FACTOR) -> dict:
    """
    Erzeugt eine einzelne SMTWTP-Instanz mit n_jobs Jobs nach dem
    Baker/Potts-Van-Wassenhove-Schema.

    Returns:
        dict {job_id: {"processing_time", "due_date", "weight"}}
    """
    rng = random.Random(seed)

    processing_times = {j: rng.randint(1, 100) for j in range(n_jobs)}
    total_processing_time = sum(processing_times.values())

    lower = total_processing_time * (1 - tau - r / 2)
    upper = total_processing_time * (1 - tau + r / 2)
    lower = max(0, lower)  # Fälligkeitsdaten dürfen nicht negativ sein

    due_dates = {j: round(rng.uniform(lower, upper)) for j in range(n_jobs)}
    weights = {j: rng.randint(1, 10) for j in range(n_jobs)}

    instance = {
        "n_jobs": n_jobs,
        "seed": seed,
        "tardiness_factor": tau,
        "range_factor": r,
        "jobs": {
            j: {
                "processing_time": processing_times[j],
                "due_date": due_dates[j],
                "weight": weights[j],
            }
            for j in range(n_jobs)
        },
    }
    return instance


def generate_all_smtwtp_instances() -> list:
    """
    Generiert alle Instanzen gemäß SMTWTP_SIZES / SMTWTP_INSTANCES_PER_SIZE
    aus config.py, speichert sie als JSON-Dateien und gibt die Liste der
    erzeugten Dateipfade zurück.
    """
    out_dir = INSTANCES_DIR / "smtwtp"
    out_dir.mkdir(parents=True, exist_ok=True)

    created_files = []
    for size in SMTWTP_SIZES:
        for idx in range(SMTWTP_INSTANCES_PER_SIZE):
            seed = RANDOM_SEED + size * 1000 + idx  # deterministisch, pro Instanz eindeutig
            instance = generate_smtwtp_instance(size, seed)

            out_path = out_dir / f"smtwtp_n{size}_inst{idx}.json"
            with open(out_path, "w") as f:
                json.dump(instance, f, indent=2)
            created_files.append(out_path)

    return created_files


if __name__ == "__main__":
    files = generate_all_smtwtp_instances()
    print(f"{len(files)} SMTWTP-Instanzen erzeugt in {INSTANCES_DIR / 'smtwtp'}")
