"""Configuration, schedules and the CLI entry points."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cropguard.config import TrainConfig
from cropguard.train import build_parser, config_from_args, cosine_lr, set_seed, softmax

ROOT = Path(__file__).resolve().parent.parent


def test_shipped_configs_load_and_are_consistent():
    for name in ("default.yaml", "edge_nano.yaml"):
        cfg = TrainConfig.load(ROOT / "configs" / name)
        assert cfg.optim.epochs > 0
        assert cfg.data.image_size > 0
        assert 0.0 <= cfg.policy.min_threshold < 1.0
        # A non-pretrained config must not also ask for a frozen warm start.
        if not cfg.model.pretrained:
            assert cfg.optim.freeze_epochs == 0
            assert cfg.optim.backbone_lr_scale == 1.0


def test_dotted_overrides_apply_to_nested_sections():
    cfg = TrainConfig().apply_overrides(
        ["optim.lr=1e-3", "data.batch_size=8", "model.pretrained=false", "run_name=x"]
    )
    assert cfg.optim.lr == pytest.approx(1e-3)
    assert cfg.data.batch_size == 8
    assert cfg.model.pretrained is False
    assert cfg.run_name == "x"


def test_unknown_override_is_rejected_rather_than_ignored():
    """A typo'd override that silently does nothing wastes a training run."""
    with pytest.raises(KeyError):
        TrainConfig().apply_overrides(["optim.learning_rate=1e-3"])
    with pytest.raises(KeyError):
        TrainConfig().apply_overrides(["nonsense.key=1"])
    with pytest.raises(ValueError):
        TrainConfig().apply_overrides(["optim.lr"])


def test_config_round_trips_through_json(tmp_path):
    cfg = TrainConfig().apply_overrides(["optim.epochs=7", "data.image_size=160"])
    loaded = TrainConfig.load(cfg.save(tmp_path / "cfg.json"))
    assert loaded.optim.epochs == 7
    assert loaded.data.image_size == 160
    assert isinstance(loaded.data.batch_size, int)


def test_cli_flags_override_the_config_file(tmp_path):
    cfg_path = TrainConfig().apply_overrides(["optim.epochs=99"]).save(tmp_path / "c.json")
    args = build_parser().parse_args(
        ["--config", str(cfg_path), "--epochs", "3", "--lr", "0.01",
         "--no-pretrained", "--set", "data.aug_strength=0.5"]
    )
    cfg = config_from_args(args)
    assert cfg.optim.epochs == 3
    assert cfg.optim.lr == pytest.approx(0.01)
    assert cfg.model.pretrained is False
    assert cfg.data.aug_strength == pytest.approx(0.5)


def test_cosine_schedule_warms_up_then_decays():
    total, warmup = 100, 10
    values = [cosine_lr(s, total, warmup, 0.02) for s in range(total)]
    assert values[0] < values[warmup - 1] <= 1.0      # ramping up
    assert values[warmup - 1] == pytest.approx(1.0)
    assert values[-1] < values[warmup]                # decaying after
    assert min(values) >= 0.0
    assert all(b <= a + 1e-9 for a, b in zip(values[warmup:], values[warmup + 1:]))


def test_cosine_schedule_handles_degenerate_inputs():
    assert cosine_lr(0, 0, 0, 0.02) == 1.0
    assert 0.0 <= cosine_lr(500, 100, 10, 0.02) <= 1.0


def test_softmax_with_temperature_is_a_distribution():
    logits = np.random.default_rng(0).normal(0, 3, (5, 7))
    for t in (0.25, 1.0, 4.0):
        p = softmax(logits, t)
        assert np.allclose(p.sum(1), 1.0)
        assert (p >= 0).all()
    assert softmax(logits, 0.25).max(1).mean() > softmax(logits, 4.0).max(1).mean()


def test_softmax_is_numerically_stable_on_large_logits():
    p = softmax(np.array([[1000.0, 999.0, -1000.0]]), 1.0)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(1), 1.0)


def test_set_seed_makes_a_run_reproducible():
    import torch

    set_seed(123)
    a = (torch.randn(4), np.random.rand(4))
    set_seed(123)
    b = (torch.randn(4), np.random.rand(4))
    assert torch.equal(a[0], b[0])
    assert np.allclose(a[1], b[1])


@pytest.mark.slow
def test_prepare_dataset_cli_builds_a_manifest(tmp_path):
    out = tmp_path / "manifest.csv"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "prepare_dataset.py"),
            "--synthetic", "--per-class", "6", "--image-size", "48",
            "--classes", "background,tomato__healthy,pest__aphid",
            "--synthetic-dir", str(tmp_path / "images"),
            "--out", str(out),
            "--save-taxonomy", str(tmp_path / "taxonomy.json"),
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert (tmp_path / "taxonomy.json").exists()

    summary = json.loads((tmp_path / "manifest.summary.json").read_text())
    assert summary["classes"] == 3
    assert summary["leaked_groups"] == []
    assert set(summary["splits"]) == {"train", "val", "test"}


def test_prepare_dataset_cli_requires_a_source():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_dataset.py"), "--out", "/tmp/x.csv"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "--synthetic" in result.stderr
