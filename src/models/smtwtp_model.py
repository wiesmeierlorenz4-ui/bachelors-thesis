"""
Gurobi MIP-Modell für das Single Machine Total Weighted Tardiness Problem (SMTWTP).

Problem 1 der Bachelorarbeit: 1 || ΣwⱼTⱼ (Graham et al. 1979 Notation).

Wissenschaftliche Grundlage der Formulierung:
    - Notation und Problemdefinition: Pinedo, M. (2016). Scheduling: Theory,
      Algorithms, and Systems.
    - Standardreferenz für SMTWTP: Potts, C.N., & Van Wassenhove, L.N. (1985).
    - NP-Schwere: Lenstra, J.K., Rinnooy Kan, A.H.G., & Brucker, P. (1977).

Modellierungsansatz: Pairwise-Ordering-Formulierung mit Transitivitäts-
Constraints (Linear-Ordering-Polytop). Für jedes Job-Paar (i, j) mit i < j
wird eine Binärvariable y[i,j] eingeführt: y[i,j] = 1, falls Job i vor Job j
eingeplant wird. Da bei regulären Zielfunktionen (wie Total Weighted
Tardiness) niemals Leerlaufzeit optimal ist (Standardresultat der
Scheduling-Theorie, siehe Pinedo 2016, Kap. 3), lässt sich die
Fertigstellungszeit C_j direkt aus der Summe der Bearbeitungszeiten aller
vorangehenden Jobs herleiten.
"""

import gurobipy as gp
from gurobipy import GRB


def build_smtwtp_model(jobs: dict, env: gp.Env = None) -> gp.Model:
    """
    Baut das Gurobi-Modell für eine SMTWTP-Instanz.

    Args:
        jobs: Dict {job_id: {"processing_time": p_j, "due_date": d_j, "weight": w_j}}
              job_id muss 0..n-1 sein.
        env: Optionales Gurobi-Environment.

    Returns:
        Aufgebautes (noch nicht optimiertes) Gurobi-Modell. Variablen sind als
        Modell-Attribute zugänglich (m._y, m._C, m._T) für spätere Auswertung.
    """
    job_ids = sorted(jobs.keys())
    n = len(job_ids)
    p = {j: jobs[j]["processing_time"] for j in job_ids}
    d = {j: jobs[j]["due_date"] for j in job_ids}
    w = {j: jobs[j]["weight"] for j in job_ids}

    m = gp.Model("SMTWTP", env=env)

    # --- Variablen ---
    # y[i, j] = 1, falls Job i vor Job j eingeplant wird (nur für i < j definiert,
    # "j vor i" ergibt sich implizit als 1 - y[i, j])
    y = {}
    for idx_i, i in enumerate(job_ids):
        for jdx in job_ids[idx_i + 1:]:
            y[i, jdx] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{jdx}")

    C = m.addVars(job_ids, lb=0.0, name="C")  # Fertigstellungszeiten
    T = m.addVars(job_ids, lb=0.0, name="T")  # Verspätungen

    # --- Hilfsfunktion: "i vor j?" für beliebige Paare (auch i > j) ---
    def before(i, j):
        """Gibt den Ausdruck für 'i wird vor j eingeplant' zurück."""
        if i == j:
            raise ValueError("i == j nicht zulässig")
        return y[i, j] if i < j else (1 - y[j, i])

    # --- Transitivitäts-Constraints (verhindern Zyklen in der Ordnung) ---
    # Für alle drei verschiedenen Jobs i, j, k: schließt widersprüchliche
    # Reihenfolgen aus (z.B. i vor j, j vor k, aber k vor i).
    for i in job_ids:
        for j in job_ids:
            if i == j:
                continue
            for k in job_ids:
                if k == i or k == j:
                    continue
                m.addConstr(before(i, j) + before(j, k) - before(i, k) <= 1,
                            name=f"trans_{i}_{j}_{k}")

    # --- Fertigstellungszeit-Constraints ---
    # C_j >= p_j + Summe der Bearbeitungszeiten aller Jobs, die vor j liegen.
    # Da Leerlaufzeit bei regulären Zielfunktionen nie optimal ist, wird diese
    # Ungleichung durch die Minimierung von T_j (und damit C_j) im Optimum
    # automatisch zur Gleichheit (Standardtechnik, siehe Pinedo 2016).
    for j in job_ids:
        preceding_load = gp.quicksum(p[i] * before(i, j) for i in job_ids if i != j)
        m.addConstr(C[j] >= p[j] + preceding_load, name=f"completion_{j}")

    # --- Tardiness-Constraints ---
    for j in job_ids:
        m.addConstr(T[j] >= C[j] - d[j], name=f"tardiness_{j}")

    # --- Zielfunktion: gewichtete Verspätung minimieren ---
    m.setObjective(gp.quicksum(w[j] * T[j] for j in job_ids), GRB.MINIMIZE)

    # Referenzen für spätere Auswertung am Modell hinterlegen
    m._y = y
    m._C = C
    m._T = T
    m._job_ids = job_ids
    m._before = before

    return m


def solve_smtwtp(jobs: dict, time_limit: int = 300, mip_gap: float = 0.0,
                  env: gp.Env = None) -> dict:
    """
    Löst eine SMTWTP-Instanz und gibt die Ergebnisse strukturiert zurück.

    Returns:
        dict mit: status, objective, runtime_seconds, sequence (Liste von
        job_ids in optimaler Reihenfolge), completion_times, tardiness.
    """
    m = build_smtwtp_model(jobs, env=env)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPGap", mip_gap)
    m.optimize()

    result = {
        "status": m.status,
        "objective": None,
        "runtime_seconds": m.Runtime,
        "sequence": None,
        "completion_times": None,
        "tardiness": None,
    }

    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        job_ids = m._job_ids
        y = m._y

        def before_value(i: int, j: int) -> float:
            """Lösungswert von 'i vor j' — y[i,j].X falls i<j, sonst 1 - y[j,i].X."""
            return y[i, j].X if i < j else 1.0 - y[j, i].X

        # Sequenz aus paarweisen Vergleichen ableiten: Job j hat Rang =
        # Anzahl Jobs, die laut Lösung vor ihm liegen (topologische Sortierung
        # über die "before"-Relation).
        sequence = sorted(job_ids, key=lambda j: sum(
            1 for i in job_ids if i != j and before_value(i, j) > 0.5
        ))

        result["objective"] = m.ObjVal
        result["sequence"] = sequence
        result["completion_times"] = {j: m._C[j].X for j in job_ids}
        result["tardiness"] = {j: m._T[j].X for j in job_ids}

    return result


if __name__ == "__main__":
    # Kleiner manueller Test während der Entwicklung (5 Jobs, Beispielwerte)
    example_jobs = {
        0: {"processing_time": 4, "due_date": 6, "weight": 2},
        1: {"processing_time": 2, "due_date": 8, "weight": 1},
        2: {"processing_time": 6, "due_date": 5, "weight": 3},
        3: {"processing_time": 3, "due_date": 10, "weight": 1},
        4: {"processing_time": 5, "due_date": 12, "weight": 2},
    }
    res = solve_smtwtp(example_jobs)
    print(res)
