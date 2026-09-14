"""
Shared loading of the final benchmark instances and size parsing.

Both problems of the final experiment live in their own directory:

    data/instances/smtwtp/                   smtwtp_n{5,10,15,20,25}_inst{0..4}.json
    data/instances/siclsp_variable_capacity/ siclsp_T{5,10,15,20,25}_inst{0..4}.json

Earlier lot-sizing datasets used the instance-id prefix "silsp_T"; the
pattern below still accepts it so archived files stay readable, but only the
two directories above are ever loaded.
"""

import json
import re

from src.paths import INSTANCE_DIRS

_SIZE_PATTERN = re.compile(r"(?:smtwtp_n|siclsp_T|silsp_T)(\d+)_inst\d+")


def load_instances(problem: str) -> list:
    """
    Load the final instances of one problem.

    Args:
        problem: "smtwtp" or "siclsp".

    Returns:
        List of dicts: {"problem": str, "instance_id": str, "data": dict}
    """
    if problem not in INSTANCE_DIRS:
        raise ValueError(f"Unknown problem: {problem!r} (expected 'smtwtp' or 'siclsp')")

    directory = INSTANCE_DIRS[problem]
    instances = []
    if not directory.exists():
        return instances

    for f in sorted(directory.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        instances.append({"problem": problem, "instance_id": f.stem, "data": data})
    return instances


def load_all_instances() -> list:
    """Load the final instances of both problems (SMTWTP first, then SICLSP)."""
    return load_instances("smtwtp") + load_instances("siclsp")


def parse_size_from_instance_id(instance_id: str) -> int:
    """
    Extract the size level (n resp. T) from an instance id.

    Examples:
        "smtwtp_n25_inst3" -> 25
        "siclsp_T20_inst1" -> 20
    """
    match = _SIZE_PATTERN.search(instance_id)
    if not match:
        raise ValueError(f"Could not parse instance size from '{instance_id}'.")
    return int(match.group(1))
