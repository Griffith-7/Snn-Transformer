"""Run and summarize the benchmark across explicit random seeds."""

import argparse
import json
import statistics
from pathlib import Path

from astrohebbian.benchmark import main


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "runs": len(values),
    }


def build_multi_seed_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--v-levels", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/multi_seed.json"))
    parser.add_argument(
        "--resume", action="store_true", help="Resume from completed seeds in the output file"
    )
    return parser


def write_results(output, seeds, run_results, complete):
    summary = {"status": "complete" if complete else "partial", "seeds": seeds, "runs": run_results}
    if run_results:
        summary["summary"] = {}
        for model_name in run_results[0]["models"]:
            metrics = run_results[0]["models"][model_name]
            summary["summary"][model_name] = {
                metric: summarize([run["models"][model_name][metric] for run in run_results])
                for metric in metrics
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main_multi_seed(argv=None):
    args = build_multi_seed_parser().parse_args(argv)
    run_results = []
    completed_seeds = set()
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        run_results = existing.get("runs", [])
        completed_seeds = {run["config"]["seed"] for run in run_results}

    for seed in args.seeds:
        if seed in completed_seeds:
            print(f"Skipping completed seed {seed}")
            continue
        benchmark_args = [
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--test-batch-size",
            str(args.test_batch_size),
            "--d-model",
            str(args.d_model),
            "--num-heads",
            str(args.num_heads),
            "--num-layers",
            str(args.num_layers),
            "--v-levels",
            str(args.v_levels),
            "--device",
            args.device,
        ]
        if args.train_size is not None:
            benchmark_args.extend(["--train-size", str(args.train_size)])
        if args.test_size is not None:
            benchmark_args.extend(["--test-size", str(args.test_size)])
        result = main(benchmark_args)
        run_results.append(result)
        write_results(args.output, args.seeds, run_results, complete=False)
        print(f"Saved completed seed {seed} to {args.output}")

    summary = write_results(args.output, args.seeds, run_results, complete=True)
    print(f"Wrote multi-seed summary to {args.output}")
    return summary


if __name__ == "__main__":
    main_multi_seed()
