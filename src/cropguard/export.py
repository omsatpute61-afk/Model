"""Export a trained checkpoint to the formats a field device can actually run.

    python -m cropguard.export --run artifacts/runs/cropguard --formats onnx,int8,torchscript

Targets, and why each exists:

``onnx``        ONNX Runtime on a Raspberry Pi / Jetson / mini-PC gateway. The
                practical default: one runtime, no torch on the device.
``int8``        Quantised ONNX. ~4x smaller and ~2-3x faster on ARM CPU, which
                is the difference between a 1.5 s and a 0.5 s answer on a Pi.
``torchscript`` Mobile interpreter bundle for an in-app Android/iOS model, so
                the farmer's phone works with no gateway at all.

Two rules are enforced rather than documented:

1. Every artefact is written with its ``model_card.json`` beside it. A model
   file without its label order and preprocessing is a misdiagnosis waiting to
   happen.
2. Export is **verified**, not assumed. Torch and ONNX are run on the same
   inputs and the outputs compared; quantisation is additionally checked for
   agreement on predicted class. A silent export regression would show up in
   the field as wrong advice, so it fails here instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model_card import CARD_FILENAME, ModelCard
from .ood import OOD_FILENAME
from .models.detector import CropGuardNet, ExportWrapper

LOGGER = logging.getLogger("cropguard.export")

DEFAULT_OPSET = 13


@dataclass
class ExportResult:
    format: str
    path: Path
    size_mb: float
    verified: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "path": str(self.path),
            "size_mb": round(self.size_mb, 3),
            "verified": self.verified,
            "detail": self.detail,
        }


def _size_mb(path: Path) -> float:
    """Total on-device footprint, including any external-weights sidecar."""
    total = path.stat().st_size
    sidecar = path.with_suffix(path.suffix + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total / (1024 * 1024)


def _export_params() -> set[str]:
    import inspect

    return set(inspect.signature(torch.onnx.export).parameters)


def _example_input(card: ModelCard, batch: int = 1) -> torch.Tensor:
    s = card.preprocess.image_size
    return torch.randn(batch, 3, s, s)


def export_onnx(
    model: CropGuardNet,
    card: ModelCard,
    out_path: Path,
    opset: int = DEFAULT_OPSET,
    dynamic_batch: bool = True,
) -> ExportResult:
    """Export probabilities (not logits) so the device cannot mis-apply softmax."""
    wrapper = ExportWrapper(model, temperature=card.policy.temperature).eval()
    example = _example_input(card)

    dynamic_axes = (
        {
            "image": {0: "batch"},
            "label_probs": {0: "batch"},
            "category_probs": {0: "batch"},
            "severity_probs": {0: "batch"},
            "embedding": {0: "batch"},
        }
        if dynamic_batch
        else None
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if "external_data" in _export_params():
        # Newer exporters default to writing weights into a sidecar
        # "<model>.onnx.data". That is fine on a workstation and a trap in the
        # field: copying only the .onnx to a device ships a model with no
        # weights. These models are a few MB, so keep one self-contained file.
        kwargs["external_data"] = False

    torch.onnx.export(
        wrapper,
        example,
        str(out_path),
        input_names=["image"],
        output_names=["label_probs", "category_probs", "severity_probs", "embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        **kwargs,
    )

    sidecar = out_path.with_suffix(out_path.suffix + ".data")
    if sidecar.exists():
        # The exporter ignored the request; refuse to hand back a stub file.
        raise RuntimeError(
            f"ONNX weights were written to {sidecar.name} instead of being embedded. "
            f"Deploy both files together, or re-export with a torch version that "
            f"honours external_data=False."
        )

    verified, detail = _verify_onnx(wrapper, out_path, card)
    return ExportResult("onnx", out_path, _size_mb(out_path), verified, detail)


def _verify_onnx(wrapper: torch.nn.Module, path: Path, card: ModelCard, n: int = 4) -> tuple[bool, str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return False, "onnxruntime not installed - export written but NOT verified"

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = _example_input(card, batch=n)
    with torch.no_grad():
        expected = wrapper(x)[0].numpy()
    got = sess.run(None, {"image": x.numpy()})[0]

    max_diff = float(np.abs(expected - got).max())
    agree = float((expected.argmax(1) == got.argmax(1)).mean())
    ok = max_diff < 1e-3 and agree == 1.0
    return ok, f"max|Δprob|={max_diff:.2e}, top-1 agreement={agree:.0%}"


def export_int8(
    onnx_path: Path,
    out_path: Path,
    calibration_images: list[Path] | None = None,
    card: ModelCard | None = None,
) -> ExportResult:
    """Quantise to INT8.

    Static (calibrated) quantisation is used when real images are supplied -
    it is meaningfully more accurate than dynamic quantisation for a CNN,
    because activation ranges come from actual leaf photos rather than
    guesses. Falls back to dynamic quantisation when no calibration data is
    available.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        return ExportResult("int8", out_path, 0.0, False, "onnxruntime not installed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "dynamic"
    try:
        if calibration_images and card is not None:
            from onnxruntime.quantization import CalibrationDataReader, quantize_static

            class _Reader(CalibrationDataReader):
                def __init__(self, paths: list[Path], card: ModelCard):
                    from .edge.preprocess import load_and_preprocess

                    self._data = iter(
                        [{"image": load_and_preprocess(p, card.preprocess)} for p in paths]
                    )

                def get_next(self):
                    return next(self._data, None)

            quantize_static(
                str(onnx_path), str(out_path), _Reader(calibration_images, card),
                weight_type=QuantType.QInt8,
            )
            mode = f"static ({len(calibration_images)} calibration images)"
        else:
            quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QUInt8)
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail the export
        LOGGER.warning("static quantisation failed (%s); using dynamic", exc)
        try:
            quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QUInt8)
            mode = "dynamic (static failed)"
        except Exception as exc2:  # noqa: BLE001
            return ExportResult("int8", out_path, 0.0, False, f"quantisation failed: {exc2}")

    verified, detail = _verify_quantised(onnx_path, out_path, card, calibration_images)
    return ExportResult("int8", out_path, _size_mb(out_path), verified, f"{mode}; {detail}")


def _verify_quantised(
    fp32: Path, int8: Path, card: ModelCard | None, images: list[Path] | None = None
) -> tuple[bool, str]:
    """INT8 must agree with FP32 where FP32 is actually decided.

    Two traps this avoids:

    * **Random-noise inputs.** On noise the FP32 model itself is near-uniform,
      so any rounding flips the argmax and a perfectly good quantisation looks
      broken. Real images are used whenever they exist.
    * **Unconfident predictions.** Even on real images, agreement on a sample
      where FP32 says 0.26 vs 0.25 is meaningless. Agreement is therefore
      measured only where FP32 has actually committed, and probability drift is
      checked separately across everything.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return False, "not verified (onnxruntime missing)"

    size = card.preprocess.image_size if card else 224
    used_real = False
    if images and card is not None:
        from .edge.preprocess import load_and_preprocess

        try:
            x = np.concatenate(
                [load_and_preprocess(p, card.preprocess) for p in images[:32]], axis=0
            )
            used_real = True
        except Exception:  # noqa: BLE001
            x = np.random.randn(8, 3, size, size).astype(np.float32)
    else:
        x = np.random.randn(8, 3, size, size).astype(np.float32)

    run = lambda path: ort.InferenceSession(  # noqa: E731
        str(path), providers=["CPUExecutionProvider"]
    ).run(None, {"image": x})[0]
    a, b = run(fp32), run(int8)

    mean_drift = float(np.abs(a - b).mean())
    max_drift = float(np.abs(a - b).max())
    confident = a.max(axis=1) >= 0.5
    if confident.any():
        agree = float((a[confident].argmax(1) == b[confident].argmax(1)).mean())
        basis = f"{int(confident.sum())} confident sample(s)"
    else:
        agree = 1.0
        basis = "no confident FP32 predictions to compare (agreement not meaningful)"

    source = f"{x.shape[0]} {'real images' if used_real else 'random inputs'}"
    ok = agree >= 0.95 and mean_drift < 0.05
    return (
        ok,
        f"top-1 agreement {agree:.0%} on {basis} from {source}; "
        f"mean|Dprob|={mean_drift:.4f}, max|Dprob|={max_drift:.3f}",
    )


def export_torchscript(
    model: CropGuardNet, card: ModelCard, out_path: Path, mobile: bool = True
) -> ExportResult:
    """TorchScript bundle for the PyTorch Mobile interpreter (Android/iOS).

    Each optimisation stage is tried, then **verified by reloading the saved
    file and comparing outputs**, falling back to the next-simplest form if it
    does not survive the round trip. This is not defensive padding: on this
    torch build, ``optimize_for_inference`` produces a graph that saves
    happily and then fails to load with "required keyword attribute 'value' is
    undefined". Shipping that means the mobile app gets a model file it cannot
    open, and the export step would have called it a success.
    """
    wrapper = ExportWrapper(model, temperature=card.policy.temperature).eval()
    example = _example_input(card)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
        expected = wrapper(example)[0]

    def _save_and_check(save_fn, label: str) -> tuple[bool, str]:
        try:
            save_fn()
        except Exception as exc:  # noqa: BLE001
            return False, f"{label}: save failed ({_short(exc)})"
        try:
            reloaded = torch.jit.load(str(out_path))
            with torch.no_grad():
                got = reloaded(example)[0]
            diff = float((expected - got).abs().max())
        except Exception as exc:  # noqa: BLE001
            return False, f"{label}: saved but could not be reloaded ({_short(exc)})"
        if diff >= 1e-4:
            return False, f"{label}: reloaded output differs by {diff:.2e}"
        return True, f"{label}; verified by reload, max|Dprob|={diff:.2e}"

    attempts = []
    if mobile:
        def _mobile_save():
            from torch.utils.mobile_optimizer import optimize_for_mobile

            optimize_for_mobile(traced)._save_for_lite_interpreter(str(out_path))

        attempts.append((_mobile_save, "lite interpreter (mobile)"))
        attempts.append(
            (lambda: torch.jit.freeze(traced).save(str(out_path)), "frozen torchscript")
        )
    attempts.append((lambda: traced.save(str(out_path)), "plain torchscript"))

    notes = []
    for save_fn, label in attempts:
        ok, detail = _save_and_check(save_fn, label)
        if ok:
            return ExportResult(
                "torchscript", out_path, _size_mb(out_path), True,
                "; ".join([*notes, detail]) if notes else detail,
            )
        notes.append(detail)
        LOGGER.warning("torchscript export: %s - trying a simpler form", detail)

    return ExportResult(
        "torchscript", out_path,
        _size_mb(out_path) if out_path.exists() else 0.0,
        False, "; ".join(notes),
    )


def _short(exc: Exception, limit: int = 120) -> str:
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[:limit] + "..."


def export_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    formats: tuple[str, ...] = ("onnx", "int8"),
    checkpoint: str = "best.pt",
    opset: int = DEFAULT_OPSET,
    calibration_manifest: str | Path | None = None,
    calibration_size: int = 64,
) -> dict:
    from .evaluate import load_run

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "export"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, taxonomy, card, _ = load_run(run_dir, torch.device("cpu"), checkpoint)
    if card is None:
        raise FileNotFoundError(
            f"{run_dir/CARD_FILENAME} is missing - refusing to export a model "
            "whose label order and preprocessing are unknown"
        )
    card.validate(num_outputs=len(taxonomy))
    model.eval()

    results: list[ExportResult] = []
    onnx_path = out_dir / "cropguard.onnx"

    if "onnx" in formats or "int8" in formats:
        res = export_onnx(model, card, onnx_path, opset=opset)
        LOGGER.info("onnx: %.2f MB verified=%s (%s)", res.size_mb, res.verified, res.detail)
        if "onnx" in formats:
            results.append(res)

    if "int8" in formats:
        calib = _calibration_images(calibration_manifest, run_dir, calibration_size)
        res = export_int8(onnx_path, out_dir / "cropguard.int8.onnx", calib, card)
        LOGGER.info("int8: %.2f MB verified=%s (%s)", res.size_mb, res.verified, res.detail)
        results.append(res)

    if "torchscript" in formats:
        res = export_torchscript(model, card, out_dir / "cropguard.ptl")
        LOGGER.info("torchscript: %.2f MB verified=%s (%s)", res.size_mb, res.verified, res.detail)
        results.append(res)

    if "onnx" not in formats and onnx_path.exists():
        onnx_path.unlink()  # only produced as an intermediate for int8

    card.save(out_dir / CARD_FILENAME)
    ood_src = run_dir / OOD_FILENAME
    if ood_src.exists():
        shutil.copy(ood_src, out_dir / OOD_FILENAME)
    else:
        LOGGER.warning(
            "no %s in the run - the exported bundle will accept out-of-distribution "
            "images (a photo of the sky becomes a disease diagnosis). Retrain with "
            "policy.fit_ood=true.", OOD_FILENAME,
        )
    tax_src = run_dir / "taxonomy.json"
    if tax_src.exists():
        shutil.copy(tax_src, out_dir / "taxonomy.json")
    for name in ("advisory.json",):
        src = Path(__file__).parent / "resources" / name
        if src.exists():
            shutil.copy(src, out_dir / name)

    summary = {
        "run": str(run_dir),
        "out_dir": str(out_dir),
        "classes": len(taxonomy),
        "artifacts": [r.to_dict() for r in results],
        "all_verified": all(r.verified for r in results) if results else False,
    }
    with open(out_dir / "export_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _calibration_images(
    manifest: str | Path | None, run_dir: Path, limit: int
) -> list[Path] | None:
    """Real training images make INT8 calibration meaningful."""
    from .data.manifest import read_manifest

    if manifest is None:
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            try:
                manifest = json.loads(cfg_path.read_text())["data"]["manifest"]
            except Exception:  # noqa: BLE001
                return None
    if manifest is None or not Path(manifest).exists():
        return None
    records = [r for r in read_manifest(manifest) if r.split == "train"]
    if not records:
        return None
    step = max(1, len(records) // limit)  # spread across classes, not the first N
    return [Path(r.path) for r in records[::step][:limit]]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cropguard.export")
    p.add_argument("--run", required=True)
    p.add_argument("--out-dir")
    p.add_argument("--formats", default="onnx,int8",
                   help="comma separated: onnx, int8, torchscript")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--calibration-manifest")
    p.add_argument("--calibration-size", type=int, default=64)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-7s %(message)s")
    summary = export_run(
        args.run,
        out_dir=args.out_dir,
        formats=tuple(f.strip() for f in args.formats.split(",") if f.strip()),
        checkpoint=args.checkpoint,
        opset=args.opset,
        calibration_manifest=args.calibration_manifest,
        calibration_size=args.calibration_size,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
