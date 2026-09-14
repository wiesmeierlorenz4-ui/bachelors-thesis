"""
Berechnet Vergleichsmetriken zwischen ChatGPT-Lösungen und Gurobi-Optimallösungen.

Wissenschaftliche Grundlage:
    - Optimality Gap / Feasibility Rate: Standardmetriken der Scheduling-/
      Lot-Sizing-Literatur (vgl. Pinedo 2016; Pochet & Wolsey 2006).
    - Konsistenzmetrik (compute_consistency): angelehnt an Errica et al.
      (Sensitivity/Consistency-Metriken, siehe literaturliste.md Abschnitt 2)
      und theoretisch fundiert durch Wang et al. (2022, Self-Consistency,
      arXiv:2203.11171) — die Grundidee, dass Übereinstimmung über mehrere
      Samples hinweg ein aussagekräftiges Qualitätssignal ist, wird hier auf
      die Bewertung der ChatGPT-Zuverlässigkeit übertragen statt (wie im
      Original) auf die Auswahl der finalen Antwort per Mehrheitsentscheid.
"""

import math


def compute_optimality_gap(chatgpt_obj: float, gurobi_obj: float) -> float:
    """
    Berechnet die relative Abweichung der ChatGPT-Lösung vom Gurobi-Optimum,
    in Prozent. Beide Probleme sind Minimierungsprobleme, daher:
        gap = (chatgpt_obj - gurobi_obj) / gurobi_obj * 100
    Ein Wert von 0 bedeutet Optimalität, größere Werte schlechtere Lösungen.
    Gurobi-Optimum von 0 (z.B. keine Verspätung möglich) wird gesondert behandelt.
    """
    if gurobi_obj == 0:
        return 0.0 if chatgpt_obj == 0 else float("inf")
    return (chatgpt_obj - gurobi_obj) / gurobi_obj * 100


def check_feasibility_smtwtp(sequence: list, instance: dict) -> bool:
    """
    Prüft, ob eine SMTWTP-Lösung (Job-Reihenfolge) gültig ist:
    - jede Job-ID aus der Instanz kommt genau einmal vor
    - keine zusätzlichen/unbekannten Job-IDs

    HINWEIS: Prüft nur die STRUKTURELLE Gültigkeit der Sequenz (valide
    Permutation), NICHT ob ein mitgelieferter objective_value korrekt ist.
    Für Code-Based-Lösungen, bei denen ChatGPT sowohl Sequenz als auch
    Zielfunktionswert selbst berechnet, muss zusätzlich compute_smtwtp_objective()
    zur Verifikation genutzt werden — ein Code kann eine valide Permutation
    ausgeben und trotzdem einen falschen (z.B. durch einen Modellierungsfehler
    zu niedrigen) Zielfunktionswert behaupten.
    """
    job_ids = set(int(j) for j in instance["jobs"].keys())
    try:
        sequence_ids = set(int(j) for j in sequence)
    except (TypeError, ValueError):
        return False

    return sequence_ids == job_ids and len(sequence) == len(job_ids)


def compute_smtwtp_objective(sequence: list, instance: dict) -> float:
    """
    Berechnet den TATSÄCHLICHEN Zielfunktionswert (ΣwⱼTⱼ) für eine gegebene
    Sequenz direkt aus den Instanzdaten — unabhängig davon, was ChatGPT als
    objective_value behauptet hat. Notwendiger Cross-Check für Code-Based-
    Lösungen: der generierte Code kann eine valide Sequenz UND einen falschen
    (durch einen Modellierungsfehler zustande gekommenen) Wert liefern.
    """
    jobs = instance["jobs"]
    t = 0.0
    total = 0.0
    for j in sequence:
        key = j if j in jobs else str(j)
        p, d, w = jobs[key]["processing_time"], jobs[key]["due_date"], jobs[key]["weight"]
        t += p
        total += w * max(0.0, t - d)
    return total


def verify_reported_objective_smtwtp(sequence: list, reported_objective: float,
                                      instance: dict, tolerance: float = 1e-3) -> dict:
    """
    Vergleicht den von ChatGPT behaupteten objective_value mit dem tatsächlich
    aus der Sequenz nachgerechneten Wert. Bei Code-Based-Lösungen essenziell,
    da ein "erfolgreich ausgeführter" Code trotzdem ein falsches Modell
    implementiert haben kann (Ausführungserfolg != Korrektheit).

    Returns:
        dict mit "recomputed_objective", "matches_reported" (bool),
        "reported_objective".
    """
    recomputed = compute_smtwtp_objective(sequence, instance)
    return {
        "reported_objective": reported_objective,
        "recomputed_objective": recomputed,
        "matches_reported": abs(recomputed - reported_objective) <= tolerance,
    }


def classify_smtwtp_result(code_success: bool, sequence: list = None,
                            reported_objective: float = None, instance: dict = None) -> str:
    """
    Ordnet ein Ergebnis einer der vier Fehlerkategorien für die qualitative
    Analyse (Kap. 5.3, Failure Modes) zu:

    - "execution_error":   Code lief nicht durch bzw. kein valides JSON
                            (code_success=False)
    - "infeasible":        valides JSON, aber Sequenz ist keine gültige
                            Permutation der Jobs
    - "objective_mismatch": strukturell gültige Sequenz, aber der behauptete
                            objective_value stimmt nicht mit dem tatsächlich
                            nachgerechneten Wert überein (wie beim Job-Index-
                            vs-Positions-Fehler in der Completion-Time-
                            Constraint — ein eigener, für Code-Based
                            charakteristischer Fehlermodus)
    - "success":            strukturell gültig UND objective_value korrekt
    """
    if not code_success or sequence is None:
        return "execution_error"

    if not check_feasibility_smtwtp(sequence, instance):
        return "infeasible"

    if reported_objective is not None:
        verification = verify_reported_objective_smtwtp(sequence, reported_objective, instance)
        if not verification["matches_reported"]:
            return "objective_mismatch"

    return "success"


def check_feasibility_siclsp(production: dict, instance: dict) -> bool:
    """
    Prüft, ob eine SILSP-/SICLSP-Lösung (Produktionsplan) gültig ist:
    - genau die erwarteten Periodenschlüssel sind vorhanden (keine fehlenden,
      keine unerwarteten — ein fehlender Schlüssel wird NICHT mehr still als
      0 interpretiert)
    - jeder Produktionswert ist eine endliche, nicht-negative Zahl (kein
      NaN/Infinity/String — Python's json.loads akzeptiert NaN/Infinity als
      valide Fließkommawerte, die bei den reinen "<"/">"-Vergleichen der
      vorherigen Implementierung unbemerkt durchgerutscht wären, da jeder
      Vergleich mit NaN False ergibt)
    - Kapazitätsgrenze in jeder Periode eingehalten (periodenabhängig, siehe
      instance["capacity"])
    - kumulierte Produktion deckt kumulierte Nachfrage in jeder Periode
      (kein negativer Lagerbestand)

    HINWEIS: Prüft nur strukturelle Gültigkeit, NICHT ob ein mitgelieferter
    objective_value korrekt ist — siehe compute_siclsp_objective().
    """
    periods = instance["periods"]
    capacity = instance["capacity"]

    if not isinstance(production, dict):
        return False

    expected_keys = {str(int(t)) for t in periods.keys()}
    actual_keys = set()
    for k in production.keys():
        try:
            actual_keys.add(str(int(k)))
        except (TypeError, ValueError):
            return False  # nicht-normalisierbarer Periodenschlüssel

    if actual_keys != expected_keys:
        return False  # fehlende und/oder unerwartete Periodenschlüssel

    cumulative_production = 0.0
    cumulative_demand = 0.0

    for t_str in sorted(periods.keys(), key=int):
        t = int(t_str)
        raw_prod = production.get(str(t), production.get(t))

        if isinstance(raw_prod, bool) or not isinstance(raw_prod, (int, float)):
            return False  # z.B. String/None/Bool statt einer Zahl
        if not math.isfinite(raw_prod):
            return False  # NaN/Infinity ablehnen statt stillschweigend durchzulassen
        prod = float(raw_prod)

        cap = float(capacity.get(str(t), capacity.get(t, 0.0)))

        if prod < -1e-6 or prod > cap + 1e-6:
            return False  # Kapazitätsverletzung

        cumulative_production += prod
        cumulative_demand += periods[t_str]["demand"]

        if cumulative_production < cumulative_demand - 1e-6:
            return False  # Nachfrage nicht gedeckt -> negativer Lagerbestand

    return True


def compute_siclsp_objective(production: dict, instance: dict) -> float:
    """
    Berechnet den TATSÄCHLICHEN Zielfunktionswert (Setup + Produktion +
    Lagerhaltung) für einen gegebenen Produktionsplan direkt aus den
    Instanzdaten — unabhängig davon, was ChatGPT als objective_value
    behauptet hat.
    """
    periods = instance["periods"]
    total = 0.0
    inventory = 0.0

    for t_str in sorted(periods.keys(), key=int):
        t = int(t_str)
        x = float(production.get(str(t), production.get(t, 0.0)))
        p = periods[t_str]
        inventory += x - p["demand"]
        setup_indicator = 1.0 if x > 1e-6 else 0.0
        total += p["setup_cost"] * setup_indicator + p["prod_cost"] * x + p["holding_cost"] * inventory

    return total


def verify_reported_objective_siclsp(production: dict, reported_objective: float,
                                     instance: dict, tolerance: float = 1e-3) -> dict:
    """Counterpart of verify_reported_objective_smtwtp for SICLSP."""
    recomputed = compute_siclsp_objective(production, instance)
    return {
        "reported_objective": reported_objective,
        "recomputed_objective": recomputed,
        "matches_reported": abs(recomputed - reported_objective) <= tolerance,
    }


def compute_consistency(objective_values: list, technical_success_flags: list = None,
                        tolerance: float = 1e-3) -> dict:
    """
    Berechnet die Konsistenz der mathematisch auswertbaren Ergebnisse über
    wiederholte Runs derselben Instanz.

    Technisch vollständig fehlgeschlagene Runs werden aus dem Nenner
    ausgeschlossen, da sie die technische Hürde nicht überschritten und daher
    keinen Lösungsoutput erzeugt haben. Technisch erfolgreiche, aber infeasible
    Runs bleiben dagegen im Nenner: Sie haben einen auswertbaren Lösungsoutput
    erzeugt, jedoch keine gültige Lösung.

    Args:
        objective_values:
            Liste der verifizierten Zielfunktionswerte. None steht für einen
            technisch erfolgreichen, aber infeasible Run ODER einen technischen
            Failure; die Unterscheidung erfolgt über technical_success_flags.
        technical_success_flags:
            Parallele Bool-Liste. True bedeutet, dass ein parsebarer,
            strukturell auswertbarer Lösungsoutput erzeugt wurde. Falls None,
            wird aus Abwärtskompatibilitätsgründen jeder Run als technisch
            erfolgreich behandelt.
        tolerance:
            Absolute Toleranz, innerhalb derer zwei Objective Values als gleich
            gelten.

    Returns:
        dict mit:
            - n_runs_total: alle experimentellen Runs
            - n_technical_success: technisch erfolgreiche Runs
            - n_feasible: Runs mit verifiziertem Objective Value
            - n_distinct_values: verschiedene Objective Values
            - modal_value: häufigster Objective Value
            - consistency_rate: modal_count / n_technical_success
              Technische Failures sind damit ausgeschlossen; technisch
              erfolgreiche infeasible Runs senken die Konsistenz.
    """
    n_runs_total = len(objective_values)

    if technical_success_flags is None:
        technical_success_flags = [True] * n_runs_total

    if len(technical_success_flags) != n_runs_total:
        raise ValueError("technical_success_flags muss gleich lang wie objective_values sein.")

    evaluable_values = [
        value
        for value, technical_success in zip(objective_values, technical_success_flags)
        if technical_success
    ]
    feasible_values = [v for v in evaluable_values if v is not None]

    n_technical_success = len(evaluable_values)
    n_feasible = len(feasible_values)

    if n_technical_success == 0:
        return {
            "n_runs_total": n_runs_total,
            "n_technical_success": 0,
            "n_feasible": 0,
            "n_distinct_values": 0,
            "modal_value": None,
            "consistency_rate": None,
        }

    if n_feasible == 0:
        return {
            "n_runs_total": n_runs_total,
            "n_technical_success": n_technical_success,
            "n_feasible": 0,
            "n_distinct_values": 0,
            "modal_value": None,
            "consistency_rate": 0.0,
        }

    groups = []
    for v in feasible_values:
        matched = False
        for i, (rep, count) in enumerate(groups):
            if abs(v - rep) <= tolerance:
                groups[i] = (rep, count + 1)
                matched = True
                break
        if not matched:
            groups.append((v, 1))

    modal_value, modal_count = max(groups, key=lambda g: g[1])

    return {
        "n_runs_total": n_runs_total,
        "n_technical_success": n_technical_success,
        "n_feasible": n_feasible,
        "n_distinct_values": len(groups),
        "modal_value": modal_value,
        "consistency_rate": modal_count / n_technical_success,
    }

def aggregate_results(results: list) -> dict:
    """
    Aggregiert Einzelergebnisse nach (problem, approach, size).

    Die Kennzahlen werden bewusst in drei Stufen getrennt:

    1. technical_success_rate
       Anteil aller Runs, die einen parsebaren und strukturell auswertbaren
       Lösungsoutput erzeugt haben.

    2. feasibility_rate
       Anteil mathematisch feasible Lösungen NUR unter technisch erfolgreichen
       Runs. Vollständig technische Failures werden aus diesem Nenner
       ausgeschlossen.

    3. mean_optimality_gap
       Durchschnittlicher Optimality Gap NUR über feasible Lösungen.

    Zusätzlich wird overall_feasible_rate ausgewiesen:
       feasible Runs / alle experimentellen Runs.
    Diese End-to-End-Kennzahl bleibt transparent sichtbar, obwohl sie nicht
    als primäre Feasibility-Metrik verwendet wird.
    """
    from src.instances.loader import parse_size_from_instance_id

    grouped: dict = {}
    for r in results:
        size = r.get("size")
        if size is None:
            size = parse_size_from_instance_id(r["instance_id"])
        key = (r["problem"], r["approach"], size)
        grouped.setdefault(key, []).append(r)

    aggregated = {}
    for key, group in grouped.items():
        n_total = len(group)
        technical_successful = [r for r in group if r.get("technical_success")]
        n_technical_success = len(technical_successful)

        feasible_successful = [r for r in technical_successful if r.get("feasible")]
        n_feasible = len(feasible_successful)

        gaps = [
            r["optimality_gap"]
            for r in feasible_successful
            if r.get("optimality_gap") is not None
        ]

        aggregated[key] = {
            "n": n_total,
            "n_technical_success": n_technical_success,
            "n_feasible": n_feasible,
            "technical_success_rate": (
                n_technical_success / n_total if n_total else None
            ),
            "feasibility_rate": (
                n_feasible / n_technical_success
                if n_technical_success
                else None
            ),
            "overall_feasible_rate": (
                n_feasible / n_total if n_total else None
            ),
            "mean_optimality_gap": (
                sum(gaps) / len(gaps) if gaps else None
            ),
            "n_gap_observations": len(gaps),
        }

    return aggregated

