"""Phase 1 validation-result classification and aggregation.

This module consumes serialized ``ValidatorResult`` matrices.  It does not
run validators, choose a backend, or interpret a patch, keeping reporting
separate from both the runner and the later repair/evaluation phases.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


VALIDATOR_PHASES = ("compile", "functionality", "security")
VALIDATION_WORLDS = ("vulnerable", "repaired")


def classify_world_result(world: str, result: object) -> str:
    """Return one deterministic Phase 1 failure class for a world result."""

    if world not in VALIDATION_WORLDS:
        raise ValueError(f"Unknown validation world: {world}")
    if not isinstance(result, Mapping):
        return "VALIDATION_REPORT_INCOMPLETE"
    if result.get("execution_error"):
        return "RUNTIME_EXECUTION_ERROR"
    patch = result.get("patch")
    if world == "repaired" and isinstance(patch, Mapping) and not patch.get("success"):
        return "REFERENCE_PATCH_APPLY_FAILED"
    matrix = result.get("matrix")
    if not isinstance(matrix, Mapping):
        return "VALIDATION_REPORT_INCOMPLETE"
    expected = {
        "compile": "PASS",
        "functionality": "PASS",
        "security": "VULNERABILITY_DETECTED" if world == "vulnerable" else "PASS",
    }
    for phase, expected_status in expected.items():
        phase_result = matrix.get(phase)
        if not isinstance(phase_result, Mapping):
            return f"{phase.upper()}_REPORT_MISSING"
        status = phase_result.get("status")
        if status == expected_status:
            continue
        if status == "NOT_RUN":
            return f"{phase.upper()}_NOT_RUN_PREREQUISITE"
        if status == "TIMEOUT":
            return f"{phase.upper()}_TIMEOUT"
        diagnostics = "\n".join(
            str(phase_result.get(field, "")) for field in ("stdout", "stderr")
        ).lower()
        if _is_network_toolchain_failure(diagnostics):
            return "TOOLCHAIN_NETWORK_UNAVAILABLE"
        if phase == "compile":
            return "COMPILE_VALIDATOR_FAILED"
        if phase == "functionality":
            return "FUNCTIONALITY_VALIDATOR_MISMATCH"
        return "SECURITY_TRANSITION_MISMATCH"
    return "MATCHED"


def aggregate_validation_results(entries: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Count every recorded phase status and one failure class per world.

    ``entries`` are cohort records containing a ``results`` object.  Missing
    matrices are represented explicitly rather than being silently omitted.
    This lets incomplete runs remain visible in the persisted coverage data.
    """

    phase_statuses: Counter[str] = Counter()
    failure_classes: Counter[str] = Counter()
    world_outcomes: Counter[str] = Counter()
    for entry in entries:
        results = entry.get("results")
        if not isinstance(results, Mapping):
            for world in VALIDATION_WORLDS:
                phase_statuses[f"{world}/runtime/REPORT_MISSING"] += 1
                failure_classes["VALIDATION_REPORT_INCOMPLETE"] += 1
                world_outcomes[f"{world}/unmatched"] += 1
            continue
        for world in VALIDATION_WORLDS:
            result = results.get(world)
            failure_class = classify_world_result(world, result)
            failure_classes[failure_class] += 1
            matched = isinstance(result, Mapping) and result.get("matches_expected") is True
            world_outcomes[f"{world}/{'matched' if matched else 'unmatched'}"] += 1
            matrix = result.get("matrix") if isinstance(result, Mapping) else None
            if not isinstance(matrix, Mapping):
                phase_statuses[f"{world}/runtime/REPORT_MISSING"] += 1
                continue
            for phase in VALIDATOR_PHASES:
                phase_result = matrix.get(phase)
                status = (
                    str(phase_result.get("status", "REPORT_MISSING"))
                    if isinstance(phase_result, Mapping)
                    else "REPORT_MISSING"
                )
                phase_statuses[f"{world}/{phase}/{status}"] += 1
    return {
        "phase_statuses": dict(sorted(phase_statuses.items())),
        "failure_classes": dict(sorted(failure_classes.items())),
        "world_outcomes": dict(sorted(world_outcomes.items())),
    }


def _is_network_toolchain_failure(diagnostics: str) -> bool:
    """Identify common unavailable-registry/DNS diagnostics from toolchains."""

    indicators = (
        "binaries.soliditylang.org",
        "temporary failure in name resolution",
        "dns error",
        "eai_again",
        "getaddrinfo",
        "could not resolve",
        "network is unreachable",
    )
    return any(indicator in diagnostics for indicator in indicators)
