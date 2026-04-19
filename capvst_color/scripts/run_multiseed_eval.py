from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from capvst_color.config_io import EVAL_CONFIG_DEFAULTS, load_yaml_config  # noqa: E402
from capvst_color.evaluate import run_photoreal_evaluation  # noqa: E402
from capvst_color.utils import ensure_dir, write_json  # noqa: E402


DATASET_CONFIGS = {
    "dpst": REPO_ROOT / "configs" / "eval_dpst.yaml",
    "pst50": REPO_ROOT / "configs" / "eval_pst50.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed CAP-VST evaluation and aggregate mean/std metrics.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIGS),
        default=["dpst", "pst50"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Training seeds to evaluate.",
    )
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        required=True,
        help="Checkpoint path template with a {seed} placeholder, e.g. ../outputs/train/seed_{seed}/checkpoints/best.pt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "eval" / "multiseed",
        help="Where to place per-seed eval outputs and aggregate JSON files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation, e.g. cuda or cuda:1.",
    )
    return parser.parse_args()


def aggregate_summaries(seed_summaries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            key
            for summary in seed_summaries.values()
            for key, value in summary.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    )
    aggregate: dict[str, Any] = {"seeds": sorted(seed_summaries)}
    for metric_name in metric_names:
        values = [
            float(summary[metric_name])
            for summary in seed_summaries.values()
            if metric_name in summary and math.isfinite(float(summary[metric_name]))
        ]
        if not values:
            continue
        aggregate[metric_name] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "values": values,
        }
    return aggregate


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    overall: dict[str, Any] = {}

    for dataset_name in args.datasets:
        config_path = DATASET_CONFIGS[dataset_name]
        dataset_output_root = ensure_dir(output_root / dataset_name)
        seed_summaries: dict[int, dict[str, Any]] = {}

        for seed in args.seeds:
            config = load_yaml_config(config_path, defaults=EVAL_CONFIG_DEFAULTS)
            config["seed"] = seed
            config["device"] = args.device
            config["checkpoint"] = args.checkpoint_template.format(seed=seed)
            config["evaluation"]["save_images"] = False
            config["evaluation"]["output_dir"] = str(dataset_output_root / f"seed_{seed}")

            result = run_photoreal_evaluation(config)
            summary = dict(result["summary"]["photoreal_benchmark"])
            seed_summaries[seed] = summary

        aggregate = {
            "dataset": dataset_name,
            "per_seed": {str(seed): summary for seed, summary in sorted(seed_summaries.items())},
            "aggregate": aggregate_summaries(seed_summaries),
        }
        write_json(dataset_output_root / "mean_std_summary.json", aggregate)
        overall[dataset_name] = aggregate

    write_json(output_root / "mean_std_summary.json", overall)
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
