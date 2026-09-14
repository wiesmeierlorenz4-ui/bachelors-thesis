"""
Prompt templates of the two MAIN treatments of the final experiment.

    Direct Solution      -> build_direct_prompt_smtwtp / build_direct_prompt_siclsp
    Code-Based Solution  -> build_code_based_prompt_smtwtp / build_code_based_prompt_siclsp

The Direct prompts in this module contain NO Chain-of-Thought trigger. The
Zero-Shot-CoT variant is a SECONDARY experiment and lives in a separate
module (cot_prompts.py) so that a main-experiment runner can never
accidentally route to a CoT prompt.

Standardisation: each prompt type x problem has exactly ONE fixed wording that
is identical across all instances of that category (only the instance data
vary). Rationale: prompt-sensitivity literature reports large performance
variance from mere rephrasing (see docs/literaturliste.md); non-standardised
prompts would confound differences between instance sizes.

Structured output: every prompt requests a machine-readable JSON result so
that the independent evaluation pipeline can parse solutions without
error-prone free-text extraction.

The exact wording is experimental material: changing it would change the
experiment, so these strings are fixed.
"""

import json


def build_direct_prompt_smtwtp(instance: dict) -> str:
    """MAIN Direct prompt for SMTWTP — zero-shot, single-turn, no CoT trigger."""
    jobs_table = "\n".join(
        f"  Job {j}: Bearbeitungszeit={data['processing_time']}, "
        f"Fälligkeitsdatum={data['due_date']}, Gewicht={data['weight']}"
        for j, data in sorted(instance["jobs"].items())
    )

    return f"""Du löst ein Single-Machine-Scheduling-Problem (1||ΣwⱼTⱼ): {instance['n_jobs']} Jobs \
müssen auf einer einzigen Maschine nacheinander (ohne Unterbrechung, ohne \
Leerlaufzeit) eingeplant werden. Ziel: Minimiere die Summe der gewichteten \
Verspätungen aller Jobs.

Jobdaten:
{jobs_table}

Definition: Verspätung eines Jobs = max(0, Fertigstellungszeit - Fälligkeitsdatum).


Gib am Ende deiner Antwort AUSSCHLIESSLICH einen JSON-Block in folgendem Format an:
```json
{{"sequence": [<Job-IDs in eingeplanter Reihenfolge>], "objective_value": <Zahl>}}
```"""


def build_direct_prompt_siclsp(instance: dict) -> str:
    """MAIN Direct prompt for SICLSP — zero-shot, single-turn, no CoT trigger.

    The period table always carries the PERIOD-SPECIFIC capacity of the
    capacitated problem (instance["capacity"] is a per-period mapping)."""
    capacity = instance["capacity"]

    def _cap(t):
        """Capacity lookup that tolerates int or str keys (JSON deserialisation
        turns dict keys into strings)."""
        return capacity.get(t, capacity.get(str(t)))

    periods_table = "\n".join(
        f"  Periode {t}: Nachfrage={data['demand']}, Setup-Kosten={data['setup_cost']}, "
        f"Produktionskosten/Einheit={data['prod_cost']}, Lagerkosten/Einheit={data['holding_cost']}, "
        f"Kapazität={_cap(t)}"
        for t, data in sorted(instance["periods"].items(), key=lambda kv: int(kv[0]))
    )

    return f"""Du löst ein Single-Item Lot-Sizing-Problem mit Kapazitätsrestriktion über \
{instance['n_periods']} Perioden. In jeder Periode kann produziert werden \
(mit fixen Setup-Kosten, falls produziert wird, plus variablen \
Produktionskosten pro Einheit), bis zur jeweiligen Kapazitätsgrenze. \
Nicht verkaufte Mengen werden mit Lagerkosten pro Einheit auf die \
Folgeperiode übertragen. Kein Anfangsbestand. Ziel: Minimiere die \
Gesamtkosten (Setup + Produktion + Lagerhaltung) über alle Perioden, \
sodass die Nachfrage jeder Periode exakt gedeckt wird.

Periodendaten:
{periods_table}


Gib am Ende deiner Antwort AUSSCHLIESSLICH einen JSON-Block in folgendem Format an:
```json
{{"production": {{"<periode>": <menge>, ...}}, "objective_value": <Zahl>}}
```"""


def build_code_based_prompt_smtwtp(instance: dict) -> str:
    """MAIN Code-Based prompt for SMTWTP — asks for an executable program."""
    jobs_json = json.dumps(instance["jobs"], indent=2)

    return f"""Du löst ein Single-Machine-Scheduling-Problem (1||ΣwⱼTⱼ): {instance['n_jobs']} Jobs \
müssen auf einer einzigen Maschine nacheinander (ohne Unterbrechung, ohne \
Leerlaufzeit) eingeplant werden. Ziel: Minimiere die Summe der gewichteten \
Verspätungen aller Jobs.

Jobdaten (JSON, job_id -> {{processing_time, due_date, weight}}):
{jobs_json}

Definition: Verspätung eines Jobs = max(0, Fertigstellungszeit - Fälligkeitsdatum).

Schreibe ein vollständiges, direkt ausführbares Python-Skript, das dieses \
Problem exakt löst (z.B. via gurobipy) und am Ende via print() GENAU ein \
JSON-Objekt im Format \
{{"sequence": [...], "objective_value": <Zahl>}} ausgibt. WICHTIG: \
Unterdrücke jegliche Solver-Ausgabe (bei gurobipy z.B. \
model.Params.OutputFlag = 0 direkt nach Modell-Erstellung setzen) und gib \
KEINE weiteren print()-Ausgaben aus außer dem finalen JSON-Objekt. Gib \
ausschließlich den Code in einem einzigen ```python ...``` Block zurück."""


def build_code_based_prompt_siclsp(instance: dict) -> str:
    """MAIN Code-Based prompt for SICLSP — asks for an executable program.

    The full per-period capacity vector is handed to the model as JSON."""
    periods_json = json.dumps(instance["periods"], indent=2)
    capacity_json = json.dumps(instance["capacity"], indent=2)

    return f"""Du löst ein Single-Item Lot-Sizing-Problem mit Kapazitätsrestriktion über \
{instance['n_periods']} Perioden. In jeder Periode kann produziert werden \
(mit fixen Setup-Kosten, falls produziert wird, plus variablen \
Produktionskosten pro Einheit), bis zur jeweiligen Kapazitätsgrenze. \
Nicht verkaufte Mengen werden mit Lagerkosten pro Einheit auf die \
Folgeperiode übertragen. Kein Anfangsbestand. Ziel: Minimiere die \
Gesamtkosten (Setup + Produktion + Lagerhaltung) über alle Perioden, \
sodass die Nachfrage jeder Periode exakt gedeckt wird.

Periodendaten (JSON, periode -> {{demand, setup_cost, prod_cost, holding_cost}}):
{periods_json}

Kapazität pro Periode (JSON):
{capacity_json}

Schreibe ein vollständiges, direkt ausführbares Python-Skript, das dieses \
Problem exakt löst (z.B. via gurobipy) und am Ende via print() GENAU ein \
JSON-Objekt im Format \
{{"production": {{"<periode>": <menge>, ...}}, "objective_value": <Zahl>}} \
ausgibt. WICHTIG: Unterdrücke jegliche Solver-Ausgabe (bei gurobipy z.B. \
model.Params.OutputFlag = 0 direkt nach Modell-Erstellung setzen) und gib \
KEINE weiteren print()-Ausgaben aus außer dem finalen JSON-Objekt. Gib \
ausschließlich den Code in einem einzigen ```python ...``` Block zurück."""
