"""Measure what the model actually costs on the device it will run on.

    python -m cropguard.benchmark --bundle artifacts/runs/demo/export

"Runs on the edge" is a claim, not a design. This prints the numbers that
decide whether it is true: cold-start time, per-image latency (p50/p95/p99),
throughput, model size on disk, and peak RSS. Run it on the target board, not
on a workstation - a laptop number tells you nothing about a Pi.

The latency budget that matters in the field: a farmer pointing a phone at a
leaf will tolerate roughly a second. Anything past ~2 s and the device gets
left in the shed.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class BenchmarkResult:
    backend: str
    model: str
    model_size_mb: float
    image_size: int
    runs: int
    batch_size: int
    load_seconds: float
    latency_ms: dict[str, float] = field(default_factory=dict)
    throughput_ips: float = 0.0
    peak_rss_mb: float | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model,
            "model_size_mb": round(self.model_size_mb, 3),
            "image_size": self.image_size,
            "runs": self.runs,
            "batch_size": self.batch_size,
            "load_seconds": round(self.load_seconds, 3),
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
            "throughput_ips": round(self.throughput_ips, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 1) if self.peak_rss_mb else None,
            "environment": self.environment,
        }

    def format(self) -> str:
        lines = [
            f"backend        {self.backend}",
            f"model          {self.model} ({self.model_size_mb:.2f} MB)",
            f"input          {self.image_size}x{self.image_size}, batch {self.batch_size}",
            f"cold start     {self.load_seconds:.2f} s",
            f"latency p50    {self.latency_ms.get('p50', 0):.1f} ms/image",
            f"latency p95    {self.latency_ms.get('p95', 0):.1f} ms/image",
            f"latency p99    {self.latency_ms.get('p99', 0):.1f} ms/image",
            f"throughput     {self.throughput_ips:.1f} images/s",
        ]
        if self.peak_rss_mb:
            lines.append(f"peak RSS       {self.peak_rss_mb:.0f} MB")
        lines.append(f"host           {self.environment.get('platform', '?')}")
        budget = self.latency_ms.get("p95", 0)
        verdict = (
            "well within a field budget" if budget < 500
            else "usable" if budget < 1500
            else "TOO SLOW for interactive field use - quantise, shrink the input, or use a smaller backbone"
        )
        lines.append(f"verdict        p95 {budget:.0f} ms - {verdict}")
        return "\n".join(lines)


def _peak_rss_mb() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return usage / 1024.0 if platform.system() != "Darwin" else usage / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        return {}
    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
        return ordered[idx]
    return {
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99),
        "max": ordered[-1],
        "stdev": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def benchmark_bundle(
    bundle_dir: str | Path,
    runs: int = 60,
    warmup: int = 8,
    batch_size: int = 1,
    model_file: str | None = None,
    backend: str = "auto",
    num_threads: int | None = None,
    image: str | Path | None = None,
    include_preprocessing: bool = True,
) -> BenchmarkResult:
    """Time the full path a field image takes, preprocessing included.

    Preprocessing is timed by default because that is what the device really
    does. Benchmarks that time only the forward pass routinely understate
    end-to-end latency by 30-50% on a slow CPU.
    """
    from .edge.runtime import EdgeClassifier

    load_start = time.perf_counter()
    clf = EdgeClassifier(
        bundle_dir, model_file=model_file, backend=backend, num_threads=num_threads
    )
    load_seconds = time.perf_counter() - load_start

    size = clf.card.preprocess.image_size
    if image is not None:
        with Image.open(image) as im:
            sample = im.convert("RGB").copy()
    else:
        rng = np.random.default_rng(0)
        sample = Image.fromarray(
            rng.integers(0, 255, (size * 2, size * 2, 3), dtype=np.uint8)
        )
    images = [sample] * batch_size

    if include_preprocessing:
        def step() -> None:
            clf.diagnose_batch(images, with_advisory=False)
    else:
        from .edge.preprocess import preprocess_image

        batch = np.concatenate([preprocess_image(im, clf.card.preprocess) for im in images], axis=0)

        def step() -> None:
            clf._run(batch)

    for _ in range(warmup):
        step()

    samples: list[float] = []
    total_start = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        step()
        samples.append((time.perf_counter() - t0) * 1000.0 / batch_size)
    total = time.perf_counter() - total_start

    return BenchmarkResult(
        backend=clf.backend,
        model=clf._model_path.name,
        model_size_mb=clf._model_path.stat().st_size / (1024 * 1024),
        image_size=size,
        runs=runs,
        batch_size=batch_size,
        load_seconds=load_seconds,
        latency_ms=_percentiles(samples),
        throughput_ips=(runs * batch_size) / total if total > 0 else 0.0,
        peak_rss_mb=_peak_rss_mb(),
        environment={
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
    )


def compare_backends(
    bundle_dir: str | Path, runs: int = 40, batch_size: int = 1
) -> list[BenchmarkResult]:
    """Benchmark every artefact in the bundle - the size/speed trade in one table."""
    bundle = Path(bundle_dir)
    results = []
    for name in ("cropguard.onnx", "cropguard.int8.onnx", "cropguard.ptl"):
        if not (bundle / name).exists():
            continue
        backend = "onnx" if name.endswith(".onnx") else "torch"
        try:
            results.append(
                benchmark_bundle(bundle, runs=runs, batch_size=batch_size,
                                 model_file=name, backend=backend)
            )
        except Exception as exc:  # noqa: BLE001 - one broken artefact should not stop the table
            print(f"  {name}: benchmark failed ({exc})")
    return results


def format_comparison(results: list[BenchmarkResult]) -> str:
    if not results:
        return "no artefacts benchmarked"
    header = f"{'artefact':<24} {'backend':<8} {'MB':>7} {'p50 ms':>8} {'p95 ms':>8} {'img/s':>8}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.model:<24} {r.backend:<8} {r.model_size_mb:>7.2f} "
            f"{r.latency_ms.get('p50', 0):>8.1f} {r.latency_ms.get('p95', 0):>8.1f} "
            f"{r.throughput_ips:>8.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cropguard.benchmark")
    p.add_argument("--bundle", required=True, help="exported bundle directory")
    p.add_argument("--runs", type=int, default=60)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--model-file")
    p.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"])
    p.add_argument("--threads", type=int, help="intra-op threads (set to 1 to model a single-core device)")
    p.add_argument("--image", help="benchmark on a real photo instead of noise")
    p.add_argument("--forward-only", action="store_true", help="exclude preprocessing from the timing")
    p.add_argument("--compare", action="store_true", help="benchmark every artefact in the bundle")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if args.compare:
        results = compare_backends(args.bundle, runs=args.runs, batch_size=args.batch_size)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print(format_comparison(results))
        return 0

    result = benchmark_bundle(
        args.bundle, runs=args.runs, warmup=args.warmup, batch_size=args.batch_size,
        model_file=args.model_file, backend=args.backend, num_threads=args.threads,
        image=args.image, include_preprocessing=not args.forward_only,
    )
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
