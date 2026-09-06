"""Git-backed version history for the evolving harness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .harness import Harness
from .persistence import save_harness, write_json


class HarnessRepository:
    """Persist one training run as a linear sequence of harness commits."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def initialize(self, harness: Harness) -> None:
        if (self.path / ".git").exists():
            raise FileExistsError(f"harness repository already exists: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q")
        self._git("config", "user.name", "HAMA Optimizer")
        self._git("config", "user.email", "hama@local")
        save_harness(harness, self.path)
        self._git("add", "skills", "memory.json")
        self._git("commit", "-q", "-m", "harness: initialize")

    def history(self, limit: int = 8) -> str:
        return self._git(
            "log",
            f"-{limit}",
            "--format=commit %h%n%s%n%b",
            "-p",
            "--",
            "skills",
            "memory.json",
        ).stdout.strip()

    def commit(
        self,
        round_index: int,
        harness: Harness,
        edits: Sequence[Any],
        gradients: Sequence[Any] = (),
    ) -> None:
        if not edits:
            return
        save_harness(harness, self.path)
        evidence = [
            {
                "component": edit.parameter.component.value,
                "item_id": edit.parameter.item_id,
                "before": edit.before,
                "after": edit.after,
            }
            for edit in edits
        ]
        metadata = self.path / ".hama" / f"round-{round_index:04d}.json"
        write_json(
            metadata,
            {
                "round": round_index,
                "edits": evidence,
                "semantic_gradients": [
                    {
                        "component": gradient.parameter.component.value,
                        "item_id": gradient.parameter.item_id,
                        "feedback": gradient.feedback,
                        "rationale": gradient.rationale,
                    }
                    for gradient in gradients
                ],
            },
        )
        self._git("add", "skills", "memory.json", str(metadata.relative_to(self.path)))
        fields = ", ".join(
            f"{edit.parameter.component.value}:{edit.parameter.item_id}"
            for edit in edits
        )
        self._git("commit", "-q", "-m", f"harness: round {round_index + 1}", "-m", fields)

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.path,
            check=True,
            text=True,
            capture_output=True,
        )
