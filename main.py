import sys
from pathlib import Path

from config import load_config
from data import load_dataset
from results import save_results
from search import generate_configs, run_hyperparameter_search
from visualization import generate_charts


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python main.py <config.toml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    print(f"Experiment: {config.name}")
    print(f"Seed: {config.seed}")
    print(f"Search space: {len(generate_configs(config.search_space))} combinations\n")

    print("Loading datasets...")
    train_images, train_labels = load_dataset("data/mnist/train_split.parquet")
    val_images, val_labels = load_dataset("data/mnist/val_split.parquet")
    print(f"Train: {len(train_images)}, Val: {len(val_images)}\n")

    results = run_hyperparameter_search(config, train_images, train_labels, val_images, val_labels)

    output_dir = Path("experiments") / config.name
    save_results(config, results, output_dir)
    generate_charts(results, output_dir)

    print("\n" + "=" * 60)
    print("Top 5 Configurations by Validation Accuracy:")
    print("=" * 60)
    sorted_results = sorted(results, key=lambda x: x.val_accuracy, reverse=True)
    for i, h in enumerate(sorted_results[:5], 1):
        print(f"{i}. {h.val_accuracy * 100:.2f}% | {h.params}")


if __name__ == "__main__":
    main()
