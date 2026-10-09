"""Read training-only MSI inputs for the current on-policy loop."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rcs_opsd.artifact_paths import M_RESULTS, S_RESULTS, I_RESULTS

REPAIR_TOOLS = {"edit", "write", "patch", "apply_patch"}


def by_finding(path: Path) -> dict:
    return {record["finding_id"]: record for line in path.read_text().splitlines()
            if line.strip() for record in [json.loads(line)]}
