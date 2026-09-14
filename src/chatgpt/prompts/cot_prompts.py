"""
Prompt templates of the SECONDARY Chain-of-Thought experiment.

Chain-of-Thought is NOT one of the two main treatments. It is an additional
Direct variant that adds exactly ONE Zero-Shot-CoT trigger to the otherwise
identical MAIN Direct prompt (see main_prompts.py). Its results are stored
separately and are never counted among the 1,000 main runs.

Keeping these functions in their own module makes the separation structural:
a main-experiment runner imports main_prompts and cannot reach a CoT prompt.

The exact wording is experimental material and is fixed.
"""

# Zero-Shot-CoT trigger (Kojima et al., 2022) — the ONLY difference between
# these prompts and the corresponding MAIN Direct prompts.
COT_TRIGGER = "Denke Schritt für Schritt, bevor du dein finales Ergebnis angibst."


def build_direct_cot_prompt_smtwtp(instance: dict) -> str:
    """SECONDARY Direct+CoT prompt for SMTWTP."""
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

{COT_TRIGGER}

Gib am Ende deiner Antwort AUSSCHLIESSLICH einen JSON-Block in folgendem Format an:
```json
{{"sequence": [<Job-IDs in eingeplanter Reihenfolge>], "objective_value": <Zahl>}}
```"""


def build_direct_cot_prompt_siclsp(instance: dict) -> str:
    """SECONDARY Direct+CoT prompt for SICLSP (capacitated, per-period capacity)."""
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

{COT_TRIGGER}

Gib am Ende deiner Antwort AUSSCHLIESSLICH einen JSON-Block in folgendem Format an:
```json
{{"production": {{"<periode>": <menge>, ...}}, "objective_value": <Zahl>}}
```"""
