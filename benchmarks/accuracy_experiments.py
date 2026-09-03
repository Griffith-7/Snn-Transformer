"""Run controlled accuracy experiments against the frozen baseline."""

import argparse
import json
import statistics
from pathlib import Path

from astrohebbian.benchmark import main

EXPERIMENTS = {
    "baseline": {"num_layers": 1, "v_levels": 1},
    "two_layers": {"num_layers": 2, "v_levels": 1},
    "multi_level_values": {"num_layers": 1, "v_levels": 2},
}


def summarize_runs(runs):
    model_names = runs[0]["models"]
    summary = {}
    for model_name in model_names:
        metrics = runs[0]["models"][model_name]
        summary[model_name] = {
            metric: {
                "mean": statistics.mean([run["models"][model_name][metric] for run in runs]),
                "std": statistics.stdev(
                    [run["models"][model_name][metric] for run in runs]
                ) if len(runs) > 1 else 0.0,
            }
            for metric in metrics
        }
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=["baseline"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/accuracy_experiments.json"))
    return parser


def run_experiments(argv=None):
    args = build_parser().parse_args(argv)
    results = {"experiments": {}, "status": "partial"}
    for experiment_name in args.experiments:
        config = EXPERIMENTS[experiment_name]
        experiment_runs = []
        for seed in args.seeds:
            benchmark_args = [
                "--seed", str(seed), "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--test-batch-size", str(args.test_batch_size),
                "--d-model", str(args.d_model), "--num-heads", str(args.num_heads),
                "--num-layers", str(config["num_layers"]),
                "--v-levels", str(config["v_levels"]), "--device", args.device,
            ]
            if args.train_size is not None:
                benchmark_args.extend(["--train-size", str(args.train_size)])
            if args.test_size is not None:
                benchmark_args.extend(["--test-size", str(args.test_size)])
            run = main(benchmark_args)
            run["experiment"] = experiment_name
            experiment_runs.append(run)
            results["experiments"][experiment_name] = {
                "config": config,
                "runs": experiment_runs,
                "summary": summarize_runs(experiment_runs),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {experiment_name} seed {seed} to {args.output}")
    results["status"] = "complete"
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    run_experiments()
