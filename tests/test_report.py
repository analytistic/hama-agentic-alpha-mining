import json

from hama.report import build_training_report


def test_build_training_report_embeds_rounds_and_metrics(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "round-0000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trajectory.json").write_text(
        json.dumps({"steps": [{"reward": 0.1, "accepted": True}]}),
        encoding="utf-8",
    )
    (checkpoint / "attribution.json").write_text("{}", encoding="utf-8")
    (checkpoint / "rollout-factor-pools.json").write_text(
        json.dumps({"final_factors": []}), encoding="utf-8"
    )
    (run / "backtest-test.json").write_text(
        json.dumps({"metrics": {"ic": 0.02}}), encoding="utf-8"
    )

    output = build_training_report(tmp_path / "report.html", [run])
    document = output.read_text(encoding="utf-8")

    assert "HAMA training record" in document
    assert '"ic": 0.02' in document
    assert "${x[0]}" in document
