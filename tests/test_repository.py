import subprocess

from hama import Harness, MemoryBank, Skill, SkillLibrary
from hama.optimization import HarnessComponent, ParameterEdit, ParameterRef
from hama.repository import HarnessRepository


def test_harness_repository_records_initial_and_optimizer_commits(tmp_path):
    initial = Harness(
        skills=SkillLibrary((Skill("refine", "old description", "strategy"),)),
        memory=MemoryBank(()),
    )
    updated = Harness(
        skills=SkillLibrary((Skill("refine", "new description", "strategy"),)),
        memory=MemoryBank(()),
    )
    repository = HarnessRepository(tmp_path / "harness-repo")
    repository.initialize(initial)
    repository.commit(
        0,
        updated,
        (
            ParameterEdit(
                ParameterRef(HarnessComponent.SKILL_DESCRIPTION, "refine"),
                "old description",
                "new description",
            ),
        ),
    )

    log = repository.history()
    assert "harness: initialize" in log
    assert "harness: round 1" in log
    assert "skill_description:refine" in log
    assert (repository.path / ".hama" / "round-0000.json").exists()
    assert subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository.path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout == ""
