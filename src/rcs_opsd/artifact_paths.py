"""Artifact paths shared by the current graph, MSI, and on-policy pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/opsd-v2"
GRAPH_BASE_ROOT = ARTIFACT_ROOT / "graphs/base"
GRAPH_ROOT = ARTIFACT_ROOT / "graphs/current"
GRAPH_INDEX = ARTIFACT_ROOT / "graph_index"
M_ROOT = ARTIFACT_ROOT / "msi/m"
S_ROOT = ARTIFACT_ROOT / "msi/s"
I_ROOT = ARTIFACT_ROOT / "msi/i"
M_RESULTS = M_ROOT / "complete_findings.jsonl"
S_RESULTS = S_ROOT / "complete_findings.jsonl"
I_RESULTS = I_ROOT / "complete_findings.jsonl"
