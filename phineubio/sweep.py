"""Phase 12: multi-budget CPU experiments and a narrowly defined work ledger.

Counts algorithmic events, not FLOPs, machine instructions, power or joules.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys

from .compute import ComputeConfig, RECURRENT, run_compute
from .fairness import MODELS, generate_sequences
from .reservoir import Reservoir, ReservoirConfig


@dataclass(frozen=True)
class SweepConfig:
    budgets: tuple[int, ...] = (27, 75, 108)
    seeds: tuple[int, ...] = (369, 370)
    signal_families: tuple[str, ...] = ("iid", "markov", "periodic")
    noise_levels: tuple[float, ...] = (0.15,)
    delays: tuple[int, ...] = (2,)
    train_sequences: int = 6
    validation_sequences: int = 3
    test_sequences: int = 3
    probe_sequences: int = 3
    sequence_length: int = 24
    burn_in: int = 6
    epochs: int = 5
    learning_rates: tuple[float, ...] = (0.03, 0.1)
    inference_repeats: int = 3
    stay_probability: float = 0.85
    period: int = 6

    def __post_init__(self):
        if (not self.budgets or len(self.budgets) > 8 or
            any(type(b) is not int or not 3 <= b <= 3072 for b in self.budgets) or
            tuple(sorted(set(self.budgets))) != self.budgets):
            raise ValueError("budgets must be sorted, distinct integers in [3,3072]")
        if len(self.budgets)*len(self.seeds)*len(self.signal_families)*len(self.noise_levels)*len(self.delays) > 120:
            raise ValueError("sweep exceeds 120 experiment cells")
        self.compute_config(self.budgets[0])

    def compute_config(self, budget):
        return ComputeConfig(
            seeds=self.seeds, signal_families=self.signal_families,
            noise_levels=self.noise_levels, delays=self.delays,
            pair_visit_budget=budget, train_sequences=self.train_sequences,
            validation_sequences=self.validation_sequences,
            test_sequences=self.test_sequences, probe_sequences=self.probe_sequences,
            sequence_length=self.sequence_length, burn_in=self.burn_in,
            epochs=self.epochs, learning_rates=self.learning_rates,
            inference_repeats=self.inference_repeats,
            stay_probability=self.stay_probability, period=self.period,
        )


def lif_activity(seed, width, sequences):
    """Shadow-run exact seeded LIF steps; count active previous-spike terms.

    The implementation sums a row's weight whenever source spike is true,
    even if that weight is zero. No targets, learning or timing involved.
    """
    if not sequences:
        raise ValueError("probe must contain sequences")
    network = Reservoir(ReservoirConfig(seed=seed, neurons=width))
    visited = active = nonzero = emitted = 0
    for _clean, observed in sequences:
        network.reset()
        for bit in observed:
            old_spikes = network.spikes
            active += sum(old_spikes) * width
            nonzero += sum(1 for row in network.weights for j, w in enumerate(row)
                           if old_spikes[j] and w != 0)
            network.step(float(bit))
            emitted += sum(network.spikes)
            visited += 1
    return {"steps": visited, "active_weight_addends": active,
            "nonzero_active_edges": nonzero, "emitted_spikes": emitted}


def operation_ledger(costs, all_steps, retained_steps, activity):
    """Separate dense product terms, conditional additions and activation counts."""
    if not 1 <= retained_steps <= all_steps or activity["steps"] != all_steps:
        raise ValueError("probe length mismatch")
    ledger = {}
    for name in MODELS:
        cost = costs[name]
        width = cost["hidden_width"]
        if name == "trainable_gru":
            activations = (3*width+1)*all_steps
        elif name == "trainable_elman":
            activations = (width+1)*all_steps
        elif name == "tanh_reservoir":
            activations = width*all_steps + retained_steps
        elif name == "majority":
            activations = 0
        else:
            activations = retained_steps
        ledger[name] = {
            "recurrent_dense_scalar_products": (
                cost["recurrent_pair_visits_per_step"]*all_steps
                if name in ("trainable_gru","trainable_elman","tanh_reservoir") else 0
            ),
            "readout_scalar_products": cost["readout_feature_width"] * (
                all_steps if name in ("trainable_gru","trainable_elman") else retained_steps
            ),
            "nonlinear_evaluations": activations,
            "lif_threshold_checks": width*all_steps if name == "spiking_lif" else 0,
            "lif_active_weight_addends": activity["active_weight_addends"] if name == "spiking_lif" else 0,
            "lif_nonzero_active_edges": activity["nonzero_active_edges"] if name == "spiking_lif" else 0,
            "lif_emitted_spikes": activity["emitted_spikes"] if name == "spiking_lif" else 0,
            "fixed_parameter_slots": cost["fixed_parameter_slots"],
            "trainable_parameter_slots": cost["trainable_parameter_slots"],
        }
    return ledger


def pareto(points):
    """Descriptive accuracy-maximizing/recurrent-visits-minimizing screen."""
    if any(x["model"] not in RECURRENT or x["visits"] < 1 or
           not 0 <= x["accuracy"] <= 1 for x in points):
        raise ValueError("Pareto screen is for valid recurrent points only")
    return [x for x in points if not any(
        y is not x and y["visits"] <= x["visits"] and y["accuracy"] >= x["accuracy"] and
        (y["visits"] < x["visits"] or y["accuracy"] > x["accuracy"])
        for y in points
    )]


def run_sweep(config=None):
    cfg = config or SweepConfig()
    entries = []
    for budget in cfg.budgets:
        report = run_compute(cfg.compute_config(budget))
        costs = report["structural_cost_profiles"]
        rows = []
        for run in report["runs"]:
            probe = generate_sequences(
                run["signal_family"], cfg.probe_sequences, cfg.sequence_length,
                run["seed"]*1009+404, cfg.stay_probability, cfg.period, run["noise"]
            )
            width = report["recurrent_widths"]["spiking_lif"]["hidden_width"]
            activity = lif_activity(run["seed"], width, probe)
            begin = max(cfg.burn_in, max(cfg.delays))
            rows.append({
                "seed": run["seed"], "signal_family": run["signal_family"],
                "noise": run["noise"], "delay": run["delay"],
                "events": operation_ledger(
                    costs, cfg.probe_sequences*cfg.sequence_length,
                    cfg.probe_sequences*(cfg.sequence_length-begin), activity
                ),
            })
        entries.append({
            "budget": budget, "costs": costs,
            "runs": report["runs"], "summary": report["summary"],
            "probe_operations": rows, "environment": report["runtime_environment"],
        })
    fronts = []
    for family in cfg.signal_families:
        for noise in cfg.noise_levels:
            for delay in cfg.delays:
                points = []
                for entry in entries:
                    summary = next(x for x in entry["summary"] if
                                   (x["signal_family"],x["noise"],x["delay"])==(family,noise,delay))
                    for name in RECURRENT:
                        points.append({
                            "budget": entry["budget"], "model": name,
                            "visits": entry["costs"][name]["recurrent_pair_visits_per_step"],
                            "accuracy": summary["accuracy"][name]["mean"],
                        })
                fronts.append({"signal_family": family, "noise": noise, "delay": delay,
                               "points": points, "nondominated": pareto(points)})
    return {
        "schema_version": "12.0", "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "multi_budget_recurrent_work_audit",
        "configuration": asdict(cfg),
        "protocol": {
            "splits": "Same seeded independent training, validation, test and probe streams across all budgets",
            "model_selection": "Validation-only learning-rate tuning within each budget",
            "work": "Defined scalar product, conditional-addend, nonlinear and threshold events, not FLOPs or electrical energy",
            "lif_probe": "Untimed target-free replay counts previous-spike conditional additions, including zero-weight entries",
            "timing": "Inherited randomized-order independent-probe CPU measurements from Phase 11",
            "pareto": "Descriptive recurrent model accuracy vs structural matrix-visit cost; no statistical significance",
        },
        "budgets": entries, "frontiers": fronts,
        "limitations": [
            "Algorithmic event categories cannot be added into an authoritative operation count or treated as hardware counters.",
            "Recurrent width is capped by matrix visits, not total machine instructions or training compute.",
            "Fixed reservoirs have untrained recurrent weights while GRU and Elman train them; training budgets differ.",
            "CPU wall time depends on runtime, machine and system load; not physical energy or universal efficiency.",
            "Small synthetic binary tasks and seed sets do not establish general superiority or consciousness.",
        ],
    }


def write_csv(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("budget", "signal_family", "seed", "noise", "delay", "model", "test_accuracy",
              "recurrent_visits", "recurrent_dense_products", "readout_products",
              "lif_active_addends", "lif_nonzero_edges", "median_probe_ns")
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for budget in report["budgets"]:
            lookup={(row["signal_family"],row["seed"],row["noise"],row["delay"]):row["events"]
                    for row in budget["probe_operations"]}
            for run in budget["runs"]:
                events=lookup[(run["signal_family"],run["seed"],run["noise"],run["delay"])]
                for name in MODELS:
                    ev=events[name]
                    writer.writerow({
                        "budget":budget["budget"],"signal_family":run["signal_family"],
                        "seed":run["seed"],"noise":run["noise"],"delay":run["delay"],
                        "model":name,"test_accuracy":run["accuracy"][name],
                        "recurrent_visits":budget["costs"][name]["recurrent_pair_visits_per_step"],
                        "recurrent_dense_products":ev["recurrent_dense_scalar_products"],
                        "readout_products":ev["readout_scalar_products"],
                        "lif_active_addends":ev["lif_active_weight_addends"],
                        "lif_nonzero_edges":ev["lif_nonzero_active_edges"],
                        "median_probe_ns":run["inference"][name]["median_elapsed_ns"],
                    })


def main(argv=None):
    parser=argparse.ArgumentParser(description="PhiNeuBio Phase 12 multi-budget CPU experiment")
    parser.add_argument("--budgets",type=int,nargs="+",default=[27,75,108])
    parser.add_argument("--seed-start",type=int,default=369)
    parser.add_argument("--seed-count",type=int,default=2)
    parser.add_argument("--signal-families",nargs="+",default=["iid","markov","periodic"])
    parser.add_argument("--noise-levels",type=float,nargs="+",default=[0.15])
    parser.add_argument("--delays",type=int,nargs="+",default=[2])
    parser.add_argument("--train-sequences",type=int,default=6)
    parser.add_argument("--validation-sequences",type=int,default=3)
    parser.add_argument("--test-sequences",type=int,default=3)
    parser.add_argument("--probe-sequences",type=int,default=3)
    parser.add_argument("--sequence-length",type=int,default=24)
    parser.add_argument("--burn-in",type=int,default=6)
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument("--inference-repeats",type=int,default=3)
    parser.add_argument("--json-output",type=Path)
    parser.add_argument("--csv-output",type=Path)
    a=parser.parse_args(argv)
    try:
        if not 1 <= a.seed_count <= 12:
            raise ValueError("seed_count must be in [1,12]")
        cfg=SweepConfig(
            budgets=tuple(a.budgets),seeds=tuple(range(a.seed_start,a.seed_start+a.seed_count)),
            signal_families=tuple(a.signal_families),noise_levels=tuple(a.noise_levels),
            delays=tuple(a.delays),train_sequences=a.train_sequences,
            validation_sequences=a.validation_sequences,test_sequences=a.test_sequences,
            probe_sequences=a.probe_sequences,sequence_length=a.sequence_length,
            burn_in=a.burn_in,epochs=a.epochs,inference_repeats=a.inference_repeats,
        )
        result=run_sweep(cfg)
    except ValueError as err:
        parser.error(str(err))
    content=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.json_output:
        a.json_output.parent.mkdir(parents=True,exist_ok=True)
        a.json_output.write_text(content,encoding="utf-8")
        print("Saved Phase 12 report: "+str(a.json_output))
    else:
        print(content,end="")
    if a.csv_output:
        write_csv(result,a.csv_output)
        print("Saved Phase 12 CSV: "+str(a.csv_output),file=sys.stderr)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
