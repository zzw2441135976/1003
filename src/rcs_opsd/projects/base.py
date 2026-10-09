"""ProjectAdapter boundary.

The adapters classify a repository and expose configuration/artifact discovery.
They do not know a finding's security oracle.  Compile/functionality/security
entry points are instead represented by ValidatorSpec and run by validators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rcs_opsd.dataset.finding import ProjectFramework, ProjectInfo


_SKIPPED_DIRECTORIES = {
    ".git",
    "node_modules",
    "cache",
    "out",
    "artifacts",
    "build",
    "dist",
    "vendor",
    "lib",
    "third_party",
}
_FOUNDRY_CONFIG = {"foundry.toml", "remappings.txt"}
_HARDHAT_CONFIG = {"hardhat.config.js", "hardhat.config.ts", "hardhat.config.cjs", "hardhat.config.mjs"}
_TRUFFLE_CONFIG = {"truffle-config.js", "truffle.js"}
_BROWNIE_CONFIG = {"brownie-config.yaml", "brownie-config.yml", "brownie-config.json"}


@dataclass(frozen=True)
class PreparationResult:
    prepared: bool
    notes: tuple[str, ...] = ()


class ProjectAdapter(ABC):
    """Framework-specific repository operations.

    Implementations may later add dependency preparation and native artifact
    collection.  Phase 1 deliberately delegates actual compilation to the
    dataset's compile validator so it never substitutes project configuration.
    """

    framework: ProjectFramework

    @abstractmethod
    def supports(self, configuration_files: Iterable[Path]) -> bool:
        """Whether this adapter owns the observed configuration evidence."""

    @abstractmethod
    def detect(self, source_root: Path, configuration_files: tuple[Path, ...]) -> ProjectInfo:
        """Produce an evidence-backed ProjectInfo without mutating source."""

    def prepare(self, workspace: Path) -> PreparationResult:
        """Reserved for adapter-owned dependency preparation; Phase 1 is a no-op."""

        return PreparationResult(prepared=True, notes=("No implicit dependency installation in Phase 1.",))

    def build(self, workspace: Path) -> None:
        """Builds must be invoked through the compile ValidatorAdapter.

        Keeping this explicit prevents a framework-specific command from
        bypassing the finding-specific compile validator.
        """

        raise NotImplementedError("Use ValidatorRunner.run_validator(..., ValidatorType.COMPILE).")

    def collect_compiler_artifacts(self, workspace: Path) -> tuple[Path, ...]:
        """Return existing compiler artifacts without constructing new ones."""

        locations: list[Path] = []
        for name in ("out", "artifacts"):
            candidate = workspace / name
            if candidate.is_dir():
                locations.append(candidate)
        return tuple(locations)


def _configuration_files(source_root: Path, max_depth: int = 4) -> tuple[Path, ...]:
    """Find framework configuration without descending into dependency/build trees."""

    discovered: list[Path] = []
    stack: list[tuple[Path, int]] = [(source_root, 0)]
    config_names = _FOUNDRY_CONFIG | _HARDHAT_CONFIG | _TRUFFLE_CONFIG | _BROWNIE_CONFIG
    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_file() and entry.name in config_names:
                discovered.append(entry)
            elif entry.is_dir() and depth < max_depth and entry.name not in _SKIPPED_DIRECTORIES:
                stack.append((entry, depth + 1))
    return tuple(sorted(discovered))


class _ConfigAdapter(ProjectAdapter):
    config_names: frozenset[str] = frozenset()

    def supports(self, configuration_files: Iterable[Path]) -> bool:
        return any(path.name in self.config_names for path in configuration_files)

    def detect(self, source_root: Path, configuration_files: tuple[Path, ...]) -> ProjectInfo:
        owned = tuple(path for path in configuration_files if path.name in self.config_names)
        compiler = owned
        return ProjectInfo(
            framework=self.framework,
            source_root=source_root,
            configuration_files=configuration_files,
            compiler_configuration_files=compiler,
            evidence=tuple(str(path.relative_to(source_root)) for path in owned),
        )


class FoundryAdapter(_ConfigAdapter):
    framework = ProjectFramework.FOUNDRY
    config_names = frozenset(_FOUNDRY_CONFIG)


class HardhatAdapter(_ConfigAdapter):
    framework = ProjectFramework.HARDHAT
    config_names = frozenset(_HARDHAT_CONFIG)


class GenericAdapter(_ConfigAdapter):
    framework = ProjectFramework.OTHER
    config_names = frozenset(_TRUFFLE_CONFIG | _BROWNIE_CONFIG)

    def detect(self, source_root: Path, configuration_files: tuple[Path, ...]) -> ProjectInfo:
        if not self.supports(configuration_files):
            return ProjectInfo(
                framework=ProjectFramework.UNKNOWN,
                source_root=source_root,
                configuration_files=configuration_files,
                compiler_configuration_files=(),
                evidence=("No supported framework configuration found within depth 4.",),
            )
        return super().detect(source_root, configuration_files)


class ProjectAdapterRegistry:
    """Single location for framework classification; Runner Core never branches on it."""

    def __init__(self, adapters: tuple[ProjectAdapter, ...] | None = None) -> None:
        self._foundry = FoundryAdapter()
        self._hardhat = HardhatAdapter()
        self._generic = GenericAdapter()
        self._adapters = adapters or (self._foundry, self._hardhat, self._generic)

    def detect(self, source_root: Path) -> ProjectInfo:
        configuration_files = _configuration_files(source_root)
        has_foundry = self._foundry.supports(configuration_files)
        has_hardhat = self._hardhat.supports(configuration_files)
        if has_foundry and has_hardhat:
            compiler = tuple(
                path
                for path in configuration_files
                if path.name in (_FOUNDRY_CONFIG | _HARDHAT_CONFIG)
            )
            return ProjectInfo(
                framework=ProjectFramework.FOUNDRY_HARDHAT,
                source_root=source_root,
                configuration_files=configuration_files,
                compiler_configuration_files=compiler,
                evidence=tuple(str(path.relative_to(source_root)) for path in compiler),
            )
        for adapter in self._adapters:
            if adapter.supports(configuration_files):
                return adapter.detect(source_root, configuration_files)
        return self._generic.detect(source_root, configuration_files)
