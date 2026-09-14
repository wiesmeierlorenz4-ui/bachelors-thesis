# Evaluating GPT-4o on Two NP-hard Combinatorial Optimization Problems

Repository accompanying the bachelor thesis. `src/` is the code of the
experiment that produced the thesis data: cleaned up (naming, comments,
module organisation) but behaviourally unchanged. Equivalence with the
as-executed sources kept in `archive/as_executed/` is asserted by
`tests/test_as_executed_equivalence.py`; the data-to-code lineage is
documented in `EXPERIMENT_PROVENANCE.md`.

## Research objective

How well does GPT-4o solve NP-hard combinatorial optimization problems, and
does it matter whether the model produces a solution **directly** or writes a
**program** that computes it? Results are measured against provably optimal
Gurobi benchmarks across increasing instance sizes.

## What was run

| Treatment | Runs | Model | Protocol |
|---|---|---|---|
| **Direct** (main) | 500 | `gpt-4o-2024-08-06`, T=0 | zero-shot, single turn, one call per run, **no CoT trigger** (verified against submitted payloads) |
| **Code-Based** (main) | 500 | `gpt-4o-2024-08-06`, T=0 | model writes a Python/Gurobi program, executed locally (60 s timeout), up to 5 **technical** debugging rounds |
| **Direct + CoT** (secondary) | 500 | `gpt-4o-2024-08-06`, T=0 | identical to Direct plus one Zero-Shot-CoT trigger line |
| **Codex** (exploratory) | 10 | `gpt-5.3-codex`, effort `medium` | Code-Based task, but with the **self-consistency** loop, 1 instance per size, 1 run each |

Problems and sizes (both problems, verified from the instance files):

* **SMTWTP** — Single Machine Total Weighted Tardiness, `1‖Σ wⱼTⱼ`, n = 5/10/15/20/25
* **SICLSP** — Single-Item Capacitated Lot-Sizing, **period-specific capacities**, T = 5/10/15/20/25

5 deterministic instances per size, 10 repetitions per instance and
treatment. Main experiment = 2 problems × 5 sizes × 5 instances × 10
repetitions × 2 main treatments = **1,000 runs**; plus 500 CoT and 10 Codex
runs (1,510 stored run records in total).

Identical nominal size levels do **not** imply equal computational difficulty
between SMTWTP n=25 and SICLSP T=25.

## Repository layout

```
src/                              experiment code
├── config.py                       model, temperature, repetitions, Gurobi settings
├── paths.py                        instance / ground-truth / result directories
├── instances/
│   ├── generate_smtwtp.py          SMTWTP generator
│   ├── generate_siclsp.py          SICLSP generator (per-period capacity)
│   └── loader.py
├── models/                         TRUSTED ground-truth formulations
│   ├── smtwtp_model.py
│   └── siclsp_model.py
├── chatgpt/
│   ├── prompts/main_prompts.py     Direct (no CoT) and Code-Based prompts
│   ├── prompts/cot_prompts.py      Direct + CoT prompts
│   ├── batch_api.py                Batch API helpers (chunking, polling, resume)
│   ├── run_direct.py               -> Direct, both problems       (500 runs)
│   ├── run_direct_cot.py           -> Direct + CoT, both problems (500 runs)
│   ├── run_code_based.py           -> Code-Based, both problems   (500 runs)
│   ├── code_based_self_consistency.py  solution loop used only by the Codex pilot
│   └── run_codex_pilot.py          -> Codex exploratory           (10 runs)
└── evaluation/
    ├── metrics.py                  feasibility, objective recomputation, gap, consistency
    ├── solve_ground_truth_smtwtp.py, solve_ground_truth_siclsp.py
    ├── build_summary_final.py, build_summary_siclsp.py
    ├── final_preflight.py, siclsp_preflight.py, siclsp_validation.py
    └── final_experiment_integrity_check.py

analysis/final_results_summary/   the final thesis datasets and the script that builds them
tests/                            equivalence, design and integrity tests (no API calls)
archive/                          as-executed sources and superseded artefacts
data/                             instances, results, archived datasets (READ-ONLY)
```

## Where the data are

| Dataset | Directory |
|---|---|
| SMTWTP Direct / Code-Based | `data/results/direct_smtwtp_final/smtwtp/` · `data/results/code_based_debug5/smtwtp/` |
| SICLSP Direct / Code-Based | `data/results/direct_siclsp_final/siclsp/` · `data/results/code_based_debug5_siclsp_final/siclsp/` |
| CoT (SMTWTP / SICLSP) | `data/results/direct_cot/smtwtp/` · `data/results/direct_cot_siclsp_final/siclsp/` |
| Codex | `data/results/code_based_codex_pilot/` |
| Ground truth | `data/results/ground_truth/smtwtp/` · `data/results/ground_truth_siclsp_final/` |
| Instances | `data/instances/smtwtp/` · `data/instances/siclsp_variable_capacity/` |
| **Final analysis datasets** | `analysis/final_results_summary/` |

Directory names carry development-stage suffixes (`debug5`, `_final`). They
are kept exactly as the data were written — renaming them would break the link
between datasets and the runs that produced them. All of them are defined in
one place, `src/paths.py`.

## Reproducing things

Instances and ground truth (deterministic, no API calls):

```bash
python -m src.instances.generate_smtwtp
python -m src.instances.generate_siclsp
python -m src.evaluation.solve_ground_truth_smtwtp
python -m src.evaluation.solve_ground_truth_siclsp
```

Final analysis datasets from the raw results (deterministic, read-only w.r.t.
the result files):

```bash
python analysis/final_results_summary/generate_final_results_summary.py
```

Completeness check of the collected runs:

```bash
python -m src.evaluation.final_experiment_integrity_check
```

Tests (no API calls, no writes into result directories):

```bash
python -m unittest discover -s tests
```

Re-running the LLM experiment itself (**sends real, paid requests**;
all runners are idempotent and skip existing result files):

```bash
caffeinate -i ./run_final_experiment.sh
```

## Evaluation

Every returned solution is re-checked independently; the objective reported by
the model is never trusted. Reported metrics: technical success rate,
conditional feasibility (feasible / technically successful), overall feasible
rate (feasible / all), optimality gap for feasible solutions only, and
consistency across the 10 repetitions of the same instance. Runtime was not
recorded per run and is not a metric of this thesis.

## Setup

```bash
pip install -r requirements.txt
export GRB_LICENSE_FILE=~/path/to/gurobi.lic   # otherwise gurobipy falls back to the size-limited licence
# OPENAI_API_KEY in .env at the project root
python check_env.py
```

## Provenance and known documentation gaps

* `docs/architecture.md` — structural map of the pipeline.
* `EXPERIMENT_PROVENANCE.md` — full data-to-code lineage with the evidence
  behind every claim (submitted prompt payloads, stored conversations, schema
  fingerprints, reproduction checks).
* `THESIS_METHODOLOGY_CORRECTIONS.md` — points where the written methodology
  differs from what was actually executed. The executed experiment is
  authoritative; the thesis text is to be corrected accordingly.
