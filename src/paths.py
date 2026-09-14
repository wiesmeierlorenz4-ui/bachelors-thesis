"""
Result directory layout.

The directory names below are the ones the experiment actually wrote to and
must not be changed: renaming them would break the link between the stored
result files and the runs that produced them. They are collected here so that
no other module has to hard-code them.
"""

from src.config import INSTANCES_DIR, RESULTS_DIR

PROBLEMS = ("smtwtp", "siclsp")
TREATMENTS = ("direct", "direct_cot", "code_based")

INSTANCE_DIRS = {
    "smtwtp": INSTANCES_DIR / "smtwtp",
    "siclsp": INSTANCES_DIR / "siclsp_variable_capacity",
}

GROUND_TRUTH_DIRS = {
    "smtwtp": RESULTS_DIR / "ground_truth" / "smtwtp",
    "siclsp": RESULTS_DIR / "ground_truth_siclsp_final",
}

# (treatment, problem) -> directory holding one JSON file per experimental run.
RESULT_DIRS = {
    ("direct", "smtwtp"): RESULTS_DIR / "direct_smtwtp_final" / "smtwtp",
    ("direct", "siclsp"): RESULTS_DIR / "direct_siclsp_final" / "siclsp",
    ("direct_cot", "smtwtp"): RESULTS_DIR / "direct_cot" / "smtwtp",
    ("direct_cot", "siclsp"): RESULTS_DIR / "direct_cot_siclsp_final" / "siclsp",
    ("code_based", "smtwtp"): RESULTS_DIR / "code_based_debug5" / "smtwtp",
    ("code_based", "siclsp"): RESULTS_DIR / "code_based_debug5_siclsp_final" / "siclsp",
}

CODEX_DIRS = {
    "smtwtp": RESULTS_DIR / "code_based_codex_pilot" / "smtwtp",
    "siclsp": RESULTS_DIR / "code_based_codex_pilot" / "siclsp",
}

# The value written into the "approach" field of each result record.
APPROACH_LABELS = {
    "direct": "plain_direct",
    "direct_cot": "direct_cot",
    "code_based": "code_based_debug5",
}

# Directories that must never be read by the final pipeline.
EXCLUDED_PATH_MARKERS = ("archive", "test_runs")


def batch_state_dir(treatment: str, problem: str):
    """Directory holding the batch marker and request payload of one Direct-style run."""
    return RESULT_DIRS[(treatment, problem)].parent
