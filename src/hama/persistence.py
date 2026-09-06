"""Directory persistence for HAMA harness parameters."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .harness import Harness, MemoryBank, SkillLibrary
from .types import Factor, FactorPool, MemoryEntry, Skill


def save_harness(harness: Harness, path: str | Path) -> Path:
    """Save a Harness using its directory representation.

    Exposed descriptions live in ``skills/manifest.json`` and each loaded
    strategy lives in its own
    ``skills/<id>/SKILL.md``. Existing auxiliary files under a skill directory
    are preserved.
    """

    target = Path(path).expanduser().resolve()
    if target.suffix.lower() == ".json":
        raise ValueError("a Harness path must be a directory, not a JSON file")

    skills_dir = target / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for skill in harness.skills.skills:
        directory_name = _skill_directory_name(skill.id)
        relative_strategy = Path(directory_name) / "SKILL.md"
        write_text(skills_dir / relative_strategy, skill.strategy.rstrip() + "\n")
        manifest.append(
            {
                "id": skill.id,
                "description": skill.description,
                "strategy_path": relative_strategy.as_posix(),
            }
        )

    write_json(
        skills_dir / "manifest.json",
        {"format": "hama-skills-v1", "skills": manifest},
    )
    write_json(
        target / "memory.json",
        {
            "format": "hama-memory-v1",
            "memory": [_memory_to_dict(entry) for entry in harness.memory.entries],
        },
    )
    return target


def load_harness(path: str | Path) -> Harness:
    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Harness path is not a directory: {source}")
    return _load_harness_directory(source)


def factor_pool_to_list(factor_pool: FactorPool) -> list[dict[str, str]]:
    return [
        {"name": factor.name, "expression": factor.expression} for factor in factor_pool
    ]


def factor_pool_from_list(value: Any) -> FactorPool:
    if not isinstance(value, list):
        raise TypeError("factor_pool must be a list")
    return tuple(
        Factor(name=str(item["name"]), expression=str(item["expression"]))
        for item in value
    )


def save_factor_pool(factor_pool: FactorPool, path: str | Path) -> Path:
    """Persist one factor pool as a portable JSON artifact."""

    return write_json(
        path,
        {"format": "hama-factor-pool-v1", "factors": factor_pool_to_list(factor_pool)},
    )


def load_factor_pool(path: str | Path) -> FactorPool:
    """Load a factor pool written by :func:`save_factor_pool`."""

    payload = read_json(path)
    if payload.get("format") != "hama-factor-pool-v1":
        raise ValueError("unsupported factor-pool format")
    return factor_pool_from_list(payload.get("factors"))


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON root must be an object")
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically write one UTF-8 JSON object."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def write_text(path: str | Path, value: str) -> Path:
    """Atomically write one UTF-8 text file."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def _load_harness_directory(source: Path) -> Harness:
    skills_dir = source / "skills"
    manifest = read_json(skills_dir / "manifest.json")
    raw_skills = manifest.get("skills")
    if not isinstance(raw_skills, list):
        raise TypeError("skills/manifest.json must contain a skills list")

    skills: list[Skill] = []
    for item in raw_skills:
        if not isinstance(item, dict):
            raise TypeError("each skill manifest entry must be an object")
        strategy_path = (skills_dir / str(item["strategy_path"])).resolve()
        if skills_dir != strategy_path and skills_dir not in strategy_path.parents:
            raise ValueError("skill strategy_path escapes the skills directory")
        skills.append(
            Skill(
                id=str(item["id"]),
                description=str(item["description"]),
                strategy=strategy_path.read_text(encoding="utf-8").strip(),
            )
        )

    memory_path = source / "memory.json"
    raw_memory = (
        read_json(memory_path).get("memory", []) if memory_path.exists() else []
    )
    if not isinstance(raw_memory, list):
        raise TypeError("memory.json must contain a memory list")
    memory = tuple(_memory_from_dict(item) for item in raw_memory)
    return Harness(SkillLibrary(tuple(skills)), MemoryBank(memory))


def _memory_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "key": entry.key,
        "factor_edit": entry.factor_edit,
        "evaluation": dict(entry.evaluation),
        "value": entry.value,
    }


def _memory_from_dict(value: Any) -> MemoryEntry:
    if not isinstance(value, dict):
        raise TypeError("each memory entry must be an object")
    return MemoryEntry(
        id=str(value["id"]),
        key=str(value["key"]),
        factor_edit=str(value["factor_edit"]),
        evaluation=_mapping(value.get("evaluation", {})),
        value=str(value["value"]),
    )


def _skill_directory_name(skill_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", skill_id):
        raise ValueError(
            "skill id must use only letters, digits, dot, underscore, or hyphen"
        )
    if skill_id in {".", ".."}:
        raise ValueError("invalid skill id")
    return skill_id


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("evaluation must be an object")
    return value
