"""
Validator for generated SICLSP instances.

Checks structural and numerical validity before an instance is written: period
keys, required cost/demand fields, positive integer capacities, capacity that
varies across periods, and prefix as well as total feasibility.
"""

import math


def validate_siclsp_instance(instance: dict, expected_n_periods: int = None) -> list:
    """
    Check one SICLSP instance. Returns a list of error messages (empty = valid).

    Geprüft wird:
        1.  n_periods stimmt mit der angeforderten Größe überein (falls angegeben)
        2.  demand existiert für jede Periode
        3.  setup_cost existiert für jede Periode
        4.  holding_cost existiert für jede Periode
        5.  prod_cost existiert für jede Periode
        6.  capacity existiert für jede Periode
        7.  alle demand-Werte sind positive Ganzzahlen
        8.  alle capacity-Werte sind positive Ganzzahlen
        9.  capacity ist periodenabhängig (nicht konstant) für T > 1
        10. kumulierte Kapazität >= kumulierte Nachfrage für jedes Präfix
        11. Gesamtkapazität >= Gesamtnachfrage
        12. genau die erwarteten Periodenschlüssel 1..T sind vorhanden,
            keine fehlenden, keine unerwarteten
    """
    errors = []

    n_periods = instance.get("n_periods")
    if expected_n_periods is not None and n_periods != expected_n_periods:
        errors.append(
            f"n_periods={n_periods} weicht von erwarteter Größe "
            f"{expected_n_periods} ab."
        )

    periods = instance.get("periods")
    capacity = instance.get("capacity")

    if not isinstance(periods, dict):
        errors.append("'periods' fehlt oder ist kein dict.")
        return errors  # weitere Prüfungen ohne 'periods' nicht sinnvoll
    if not isinstance(capacity, dict):
        errors.append("'capacity' fehlt oder ist kein dict.")
        return errors

    if n_periods is None:
        errors.append("'n_periods' fehlt.")
        return errors

    expected_keys = {str(t) for t in range(1, n_periods + 1)}
    period_keys = {str(k) for k in periods.keys()}
    capacity_keys = {str(k) for k in capacity.keys()}

    missing_period_keys = expected_keys - period_keys
    extra_period_keys = period_keys - expected_keys
    if missing_period_keys:
        errors.append(f"'periods' fehlen Schlüssel: {sorted(missing_period_keys, key=int)}")
    if extra_period_keys:
        errors.append(f"'periods' hat unerwartete Schlüssel: {sorted(extra_period_keys, key=int)}")

    missing_capacity_keys = expected_keys - capacity_keys
    extra_capacity_keys = capacity_keys - expected_keys
    if missing_capacity_keys:
        errors.append(f"'capacity' fehlen Schlüssel: {sorted(missing_capacity_keys, key=int)}")
    if extra_capacity_keys:
        errors.append(f"'capacity' hat unerwartete Schlüssel: {sorted(extra_capacity_keys, key=int)}")

    # Pro-Feld-Existenz und Wertebereichsprüfung nur für tatsächlich vorhandene
    # Perioden (fehlende Perioden sind bereits oben gemeldet).
    demand_values = {}
    capacity_values = {}
    for t_str in sorted(period_keys & expected_keys, key=int):
        entry = periods.get(t_str, periods.get(int(t_str)))
        if not isinstance(entry, dict):
            errors.append(f"Periode {t_str}: kein gültiger Eintrag.")
            continue
        for field in ("demand", "setup_cost", "holding_cost", "prod_cost"):
            if field not in entry:
                errors.append(f"Periode {t_str}: Feld '{field}' fehlt.")

        d = entry.get("demand")
        if not isinstance(d, int) or isinstance(d, bool) or d < 0:
            errors.append(f"Periode {t_str}: demand={d!r} ist keine gültige nicht-negative Ganzzahl.")
        else:
            demand_values[t_str] = d

    for t_str in sorted(capacity_keys & expected_keys, key=int):
        c = capacity.get(t_str, capacity.get(int(t_str)))
        if (not isinstance(c, int)) or isinstance(c, bool) or c <= 0 or not math.isfinite(c):
            errors.append(f"Periode {t_str}: capacity={c!r} ist keine gültige positive Ganzzahl.")
        else:
            capacity_values[t_str] = c

    if errors:
        # Bei bereits fehlenden/ungültigen Basisdaten sind Präfix-/Summenchecks
        # nicht aussagekräftig -> hier abbrechen, statt Folgefehler zu melden.
        return errors

    if n_periods > 1 and len(set(capacity_values.values())) <= 1:
        errors.append(
            "Kapazität ist über alle Perioden konstant (len(set(capacity.values())) <= 1) "
            "- das ist genau der zu korrigierende constant-capacity Spezialfall."
        )

    cumulative_capacity = 0
    cumulative_demand = 0
    for t_str in sorted(expected_keys, key=int):
        cumulative_capacity += capacity_values[t_str]
        cumulative_demand += demand_values[t_str]
        if cumulative_capacity < cumulative_demand:
            errors.append(
                f"Präfix-Infeasibilität bei Periode {t_str}: kumulierte Kapazität "
                f"{cumulative_capacity} < kumulierte Nachfrage {cumulative_demand}."
            )

    if cumulative_capacity < cumulative_demand:
        errors.append(
            f"Gesamtkapazität {cumulative_capacity} < Gesamtnachfrage {cumulative_demand}."
        )

    return errors


def is_valid_siclsp_instance(instance: dict, expected_n_periods: int = None) -> bool:
    """Bequeme bool-Variante von validate_siclsp_instance()."""
    return len(validate_siclsp_instance(instance, expected_n_periods)) == 0
