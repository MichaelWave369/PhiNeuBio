"""Phase 9: reproducible, seed-paired uncertainty audit for Phase 8 datasets.

Computes predeclared paired comparisons. A percentile bootstrap resamples
*whole random seeds*, not correlated timesteps. Small sweeps are inconclusive.
Audit-only CLI can reprocess saved JSON without rerunning machine profiling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Sequence

from .fairness import MODELS


COMPARISONS = (
    ("spiking_lif", "trainable_gru"),
    ("spiking_lif", "tanh_reservoir"),
    ("spiking_lif", "history_window"),
    ("trainable_gru", "history_window"),
)
MIN_INTERVAL_SEEDS = 5


def _check_phase8(source: dict) -> dict:
    if not isinstance(source, dict) or source.get("schema_version") != "8.0" or source.get("task") != "validation_tuned_multisignal_comparison":
        raise ValueError("expected a Phase 8 fairness JSON report, schema 8.0")
    if source.get("substrate") != "software_simulation":
        raise ValueError("only software simulation reports are supported")
    cfg = source.get("configuration")
    runs = source.get("runs")
    if not isinstance(cfg, dict) or not isinstance(runs, list):
        raise ValueError("missing Phase 8 configuration or runs")
    groups = {}
    dimensions = ("signal_families", "noise_levels", "delays", "seeds")
    if any(not isinstance(cfg.get(k), list) or not cfg[k] for k in dimensions):
        raise ValueError("configuration must declare nonempty signal/noise/delay/seed lists")
    for k in dimensions:
        if len(set(cfg[k])) != len(cfg[k]):
            raise ValueError("configuration contains duplicate " + k)
    if len(runs) != math.prod(len(cfg[k]) for k in dimensions):
        raise ValueError("number of runs does not cover the full predeclared experiment grid")
    for row in runs:
        if not isinstance(row, dict):
            raise ValueError("invalid run")
        try:
            family, noise, delay, seed = (row[key] for key in ("signal_family", "noise", "delay", "seed"))
            metrics = row["metrics"]
            samples = row["samples"]
        except (KeyError, TypeError) as error:
            raise ValueError("run is missing required fields") from error
        if family not in cfg["signal_families"] or noise not in cfg["noise_levels"] or delay not in cfg["delays"] or seed not in cfg["seeds"]:
            raise ValueError("run is outside the predeclared experiment grid")
        if not isinstance(metrics, dict) or any(name not in metrics or type(metrics[name]) not in (int, float) or not math.isfinite(metrics[name]) or not 0 <= metrics[name] <= 1 for name in MODELS):
            raise ValueError("missing, nonfinite or invalid model accuracies")
        if not isinstance(samples, dict) or any(type(samples.get(k)) is not int or samples[k] < 1 for k in ("train", "validation", "test", "profiling_probe")):
            raise ValueError("missing or invalid sample counts")
        key = (family, noise, delay)
        group = groups.setdefault(key, {})
        if seed in group:
            raise ValueError("duplicate run for a seed and condition")
        group[seed] = row
    for family in cfg["signal_families"]:
        for noise in cfg["noise_levels"]:
            for delay in cfg["delays"]:
                group = groups.get((family, noise, delay), {})
                if set(group) != set(cfg["seeds"]):
                    raise ValueError("a condition has missing or extra random seeds")
                for split in ("train", "validation", "test", "profiling_probe"):
                    if len({row["samples"][split] for row in group.values()}) != 1:
                        raise ValueError("paired seed study requires common sample counts")
    return groups


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def paired_interval(
    deltas: Sequence[float],
    bootstrap_seed: int = 369,
    resamples: int = 2000,
) -> dict:
    """Descriptive seed-paired effect and optional 95% percentile interval.

    For fewer than five seeds, return no interval. Bootstrap intervals are
    exploratory even for larger n. No per-timestep pseudoreplication.
    """
    if not 200 <= resamples <= 20000:
        raise ValueError("resamples must be in [200,20000]")
    values = list(deltas)
    if not values or any(not math.isfinite(x) or not -1 <= x <= 1 for x in values):
        raise ValueError("requires finite paired differences in [-1,1]")
    result = {
        "paired_seed_count": len(values),
        "mean_difference": statistics.mean(values),
        "population_stdev": statistics.pstdev(values),
        "median_difference": statistics.median(values),
        "positive_seed_count": sum(x > 0 for x in values),
        "negative_seed_count": sum(x < 0 for x in values),
        "tie_seed_count": sum(x == 0 for x in values),
        "bootstrap_95_interval": None,
        "interpretation": "insufficient_seeds" if len(values) < MIN_INTERVAL_SEEDS else "exploratory_bootstrap",
    }
    if len(values) >= MIN_INTERVAL_SEEDS:
        rng = random.Random(bootstrap_seed)
        size = len(values)
        distribution = sorted(
            sum(values[rng.randrange(size)] for _ in range(size)) / size
            for _ in range(resamples)
        )
        result["bootstrap_95_interval"] = [
            _percentile(distribution, 0.025),
            _percentile(distribution, 0.975),
        ]
    return result


def run_audit(
    phase8_report: dict, bootstrap_seed: int = 369, resamples: int = 2000
) -> dict:
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be an integer")
    if type(resamples) is not int or not 200 <= resamples <= 20000:
        raise ValueError("resamples must be an integer in [200,20000]")
    groups = _check_phase8(phase8_report)
    cfg = phase8_report["configuration"]
    audit_rows = []
    for family in cfg["signal_families"]:
        for noise in cfg["noise_levels"]:
            for delay in cfg["delays"]:
                condition = groups[(family, noise, delay)]
                ordered = [condition[seed] for seed in sorted(condition)]
                comparisons = []
                for index, (left, right) in enumerate(COMPARISONS):
                    per_seed = [
                        {"seed": row["seed"], "difference": row["metrics"][left] - row["metrics"][right]}
                        for row in ordered
                    ]
                    effect = paired_interval(
                        [item["difference"] for item in per_seed],
                        bootstrap_seed=bootstrap_seed + len(audit_rows) * len(COMPARISONS) + index,
                        resamples=resamples,
                    )
                    comparisons.append({
                        "first": left, "second": right,
                        "effect": effect,
                        "per_seed": per_seed,
                    })
                model_accuracy = {
                    model: {
                        "mean": statistics.mean(row["metrics"][model] for row in ordered),
                        "population_stdev": statistics.pstdev(row["metrics"][model] for row in ordered),
                    }
                    for model in MODELS
                }
                audit_rows.append({
                    "signal_family": family,
                    "noise": noise,
                    "delay": delay,
                    "paired_seed_count": len(ordered),
                    "test_samples_per_seed": ordered[0]["samples"]["test"],
                    "accuracy": model_accuracy,
                    "comparisons": comparisons,
                })
    source_text = json.dumps(phase8_report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    return {
        "schema_version": "9.0",
        "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "seed_paired_evidence_audit",
        "source": {
            "schema_version": "8.0",
            "canonical_json_sha256": digest,
            "environment_recorded_during_source_run": phase8_report.get("runtime_environment"),
            "source_run_count": len(phase8_report["runs"]),
        },
        "settings": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": resamples,
            "minimum_seeds_for_interval": MIN_INTERVAL_SEEDS,
            "predeclared_pairs": [list(pair) for pair in COMPARISONS],
            "confidence_label": "95% seed-level percentile bootstrap, exploratory; not a simultaneous or multiplicity-adjusted interval",
        },
        "conditions": audit_rows,
        "limitations": [
            "Bootstrap resamples independent random seeds, not time steps or frames.",
            "When fewer than five seeds are present, no interval is reported. Even five seeds are far too few for robust conclusions.",
            "Percentile intervals from small, synthetic experiments can have poor coverage; they are not proof of statistical significance.",
            "Several conditions and four comparisons are explored without multiplicity adjustment. Avoid cherry-picking apparent winners.",
            "Hardware/profiling metadata, when present, describes the original Phase 8 run; this offline audit does not rebenchmark the models.",
            "The task families and architecture budgets remain limited and not equal-compute; no real neural tissue, power measurements, or awareness.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "signal_family", "noise", "delay", "first", "second", "paired_seed_count",
            "mean_difference", "bootstrap_95_low", "bootstrap_95_high",
            "positive_seeds", "negative_seeds", "ties", "interpretation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in report["conditions"]:
            for pair in condition["comparisons"]:
                effect = pair["effect"]
                bounds = effect["bootstrap_95_interval"]
                writer.writerow({
                    "signal_family": condition["signal_family"],
                    "noise": condition["noise"], "delay": condition["delay"],
                    "first": pair["first"], "second": pair["second"],
                    "paired_seed_count": effect["paired_seed_count"],
                    "mean_difference": effect["mean_difference"],
                    "bootstrap_95_low": "" if bounds is None else bounds[0],
                    "bootstrap_95_high": "" if bounds is None else bounds[1],
                    "positive_seeds": effect["positive_seed_count"],
                    "negative_seeds": effect["negative_seed_count"],
                    "ties": effect["tie_seed_count"],
                    "interpretation": effect["interpretation"],
                })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PhiNeuBio Phase 9: audit a saved Phase 8 fairness report")
    parser.add_argument("--input", type=Path, required=True, help="Saved Phase 8 fairness JSON file")
    parser.add_argument("--output", type=Path, help="Write audit JSON; otherwise print to stdout")
    parser.add_argument("--csv-output", type=Path, help="Optional paired-comparison CSV")
    parser.add_argument("--bootstrap-seed", type=int, default=369)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        report = run_audit(source, bootstrap_seed=args.bootstrap_seed, resamples=args.resamples)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print("Saved Phase 9 audit: " + str(args.output))
    if args.csv_output:
        write_csv(report, args.csv_output)
        print("Saved paired CSV: " + str(args.csv_output), file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
