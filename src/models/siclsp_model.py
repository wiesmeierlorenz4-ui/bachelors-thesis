"""
Gurobi MIP-Modell für das capacitated Single Item Lot Sizing Problem (SILSP).

Problem 2 der Bachelorarbeit.

Wissenschaftliche Grundlage der Formulierung:
    - Wagner, H.M., & Whitin, T.M. (1958): Ursprung des Lot-Sizing-Problems
      (uncapacitated Variante, polynomiell lösbar via DP — deshalb hier NICHT
      verwendet, siehe Hinweis unten).
    - Pochet, Y., & Wolsey, L.A. (2006). Production Planning by Mixed Integer
      Programming: Standard-MIP-Formulierung für capacitated Lot Sizing
      (Kapitel 2), hier direkt übernommen.
    - Brahimi, N., Dauzère-Pérès, S., Najid, N.M., & Nordli, A. (2006):
      Literaturüberblick SILSP-Varianten.

WICHTIG — Warum capacitated statt uncapacitated:
    Die uncapacitated Variante ist mit dem Wagner-Whitin-Algorithmus in
    O(T^2) polynomiell lösbar und damit NICHT NP-schwer. Für einen fairen
    Komplexitätsvergleich mit dem NP-schweren SMTWTP (Problem 1) wird hier
    daher konsequent die CAPACITATED Variante (single-item CLSP) verwendet,
    deren NP-Schwere durch die Kapazitätsrestriktionen pro Periode entsteht.
"""

import gurobipy as gp
from gurobipy import GRB


def build_siclsp_model(periods: dict, capacity: dict, env: gp.Env = None) -> gp.Model:
    """
    Baut das Gurobi-Modell für eine capacitated SILSP-Instanz.

    Args:
        periods: Dict {t: {"demand": d_t, "setup_cost": s_t,
                            "prod_cost": p_t, "holding_cost": h_t}}
                 t muss 1..T sein (aufsteigend, lückenlos).
        capacity: Dict {t: capacity_t} — maximale Produktionsmenge pro Periode.
        env: Optionales Gurobi-Environment.

    Returns:
        Aufgebautes (noch nicht optimiertes) Gurobi-Modell.
    """
    # Keys zu int normalisieren: JSON serialisiert Dict-Keys immer als
    # Strings, was ohne Normalisierung zu lexikographischer statt
    # numerischer Sortierung führen würde (bricht die chronologische
    # Perioden-Kette der Lagerbilanz).
    periods = {int(t): v for t, v in periods.items()}
    capacity = {int(t): v for t, v in capacity.items()}
    T_list = sorted(periods.keys())

    m = gp.Model("SILSP", env=env)

    # --- Variablen ---
    x = m.addVars(T_list, lb=0.0, name="x")             # Produktionsmenge
    inv = m.addVars(T_list, lb=0.0, name="I")            # Lagerbestand am Ende der Periode
    y = m.addVars(T_list, vtype=GRB.BINARY, name="y")    # Setup-Indikator

    # --- Lagerbilanz-Constraints ---
    # I_{t-1} + x_t = d_t + I_t  (I_0 = 0: kein Anfangsbestand)
    prev_inv = 0.0
    for t in T_list:
        d_t = periods[t]["demand"]
        m.addConstr(prev_inv + x[t] == d_t + inv[t], name=f"inventory_balance_{t}")
        prev_inv = inv[t]

    # --- Kapazitäts-/Kopplungs-Constraints ---
    # x_t <= capacity_t * y_t: Produktion nur möglich, wenn Setup y_t = 1;
    # gleichzeitig durch capacity_t explizit (keine künstliche Big-M-Wahl nötig,
    # da die tatsächliche Kapazitätsgrenze der Periode als M dient — Standard
    # bei Pochet & Wolsey 2006).
    for t in T_list:
        m.addConstr(x[t] <= capacity[t] * y[t], name=f"capacity_{t}")

    # --- Zielfunktion: Setup- + Produktions- + Lagerhaltungskosten minimieren ---
    obj = gp.quicksum(
        periods[t]["setup_cost"] * y[t]
        + periods[t]["prod_cost"] * x[t]
        + periods[t]["holding_cost"] * inv[t]
        for t in T_list
    )
    m.setObjective(obj, GRB.MINIMIZE)

    m._x = x
    m._inv = inv
    m._y = y
    m._T_list = T_list

    return m


def solve_siclsp(periods: dict, capacity: dict, time_limit: int = 300,
                 mip_gap: float = 0.0, env: gp.Env = None) -> dict:
    """
    Löst eine capacitated SILSP-Instanz und gibt die Ergebnisse strukturiert zurück.

    Returns:
        dict mit: status, objective, runtime_seconds, production (dict t->x_t),
        inventory (dict t->I_t), setup (dict t->y_t).
    """
    m = build_siclsp_model(periods, capacity, env=env)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPGap", mip_gap)
    m.optimize()

    result = {
        "status": m.status,
        "objective": None,
        "runtime_seconds": m.Runtime,
        "production": None,
        "inventory": None,
        "setup": None,
    }

    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        T_list = m._T_list
        result["objective"] = m.ObjVal
        result["production"] = {t: m._x[t].X for t in T_list}
        result["inventory"] = {t: m._inv[t].X for t in T_list}
        result["setup"] = {t: round(m._y[t].X) for t in T_list}

    return result


if __name__ == "__main__":
    # Kleiner manueller Test während der Entwicklung (5 Perioden, Beispielwerte)
    example_periods = {
        1: {"demand": 20, "setup_cost": 100, "prod_cost": 2, "holding_cost": 1},
        2: {"demand": 30, "setup_cost": 100, "prod_cost": 2, "holding_cost": 1},
        3: {"demand": 15, "setup_cost": 100, "prod_cost": 2, "holding_cost": 1},
        4: {"demand": 25, "setup_cost": 100, "prod_cost": 2, "holding_cost": 1},
        5: {"demand": 10, "setup_cost": 100, "prod_cost": 2, "holding_cost": 1},
    }
    example_capacity = {t: 40 for t in example_periods}
    res = solve_siclsp(example_periods, example_capacity)
    print(res)
