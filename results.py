import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config import ExperimentConfig, TrainingHistory


def save_results(config: ExperimentConfig, results: list[TrainingHistory], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_config(config, output_dir)
    _save_results_json(config, results, output_dir)
    _save_summary(config, results, output_dir)

    print(f"Results saved to {output_dir}/")


def _save_config(config: ExperimentConfig, output_dir: Path):
    config_data = {
        "experiment": {"name": config.name, "seed": config.seed},
        "search_space": config.search_space,
        "execution": {"num_gpus": config.num_gpus, "workers_per_gpu": config.workers_per_gpu},
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)


def _save_results_json(config: ExperimentConfig, results: list[TrainingHistory], output_dir: Path):
    results_data = {
        "experiment_name": config.name,
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(results),
        "results": [
            {
                "params": asdict(h.params),
                "epoch_losses": h.epoch_losses,
                "val_loss": h.val_loss,
                "val_accuracy": h.val_accuracy,
            }
            for h in results
        ],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)


def _save_summary(config: ExperimentConfig, results: list[TrainingHistory], output_dir: Path):
    sorted_results = sorted(results, key=lambda x: x.val_accuracy, reverse=True)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Experiment: {config.name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write("=" * 60 + "\n")
        f.write("Top 10 Configurations by Validation Accuracy:\n")
        f.write("=" * 60 + "\n")
        for i, h in enumerate(sorted_results[:10], 1):
            f.write(f"{i:2d}. {h.val_accuracy * 100:.2f}% | {h.params}\n")
