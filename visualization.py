from pathlib import Path

import matplotlib.pyplot as plt

from config import TrainingHistory


def generate_charts(results: list[TrainingHistory], output_dir: Path):
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    _plot_loss_curves(results, charts_dir)
    _plot_accuracy_vs_lr(results, charts_dir)
    _plot_accuracy_vs_channels(results, charts_dir)
    _plot_top_configurations(results, charts_dir)

    print(f"Charts saved to {charts_dir}/")


def _plot_loss_curves(results: list[TrainingHistory], charts_dir: Path):
    plt.figure(figsize=(12, 8))
    for i, history in enumerate(results):
        label = f"Exp {i+1}: lr={history.params.learning_rate}, ch={history.params.cnn_channels}"
        plt.plot(history.epoch_losses, label=label, alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curves")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
    plt.tight_layout()
    plt.savefig(charts_dir / "loss_curves.png", dpi=150)
    plt.close()


def _plot_accuracy_vs_lr(results: list[TrainingHistory], charts_dir: Path):
    plt.figure(figsize=(10, 6))
    lrs = [h.params.learning_rate for h in results]
    accs = [h.val_accuracy * 100 for h in results]
    plt.scatter(lrs, accs, alpha=0.6)
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Accuracy vs Learning Rate")
    plt.tight_layout()
    plt.savefig(charts_dir / "accuracy_vs_lr.png", dpi=150)
    plt.close()


def _plot_accuracy_vs_channels(results: list[TrainingHistory], charts_dir: Path):
    plt.figure(figsize=(10, 6))
    channels = [h.params.cnn_channels for h in results]
    accs = [h.val_accuracy * 100 for h in results]
    plt.scatter(channels, accs, alpha=0.6)
    plt.xlabel("CNN Channels")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Accuracy vs CNN Channels")
    plt.tight_layout()
    plt.savefig(charts_dir / "accuracy_vs_channels.png", dpi=150)
    plt.close()


def _plot_top_configurations(results: list[TrainingHistory], charts_dir: Path):
    sorted_results = sorted(results, key=lambda x: x.val_accuracy, reverse=True)[:10]
    plt.figure(figsize=(12, 6))
    labels = [
        f"lr={h.params.learning_rate}\nch={h.params.cnn_channels}\nfc={h.params.cnn_fc_hidden}"
        for h in sorted_results
    ]
    accuracies = [h.val_accuracy * 100 for h in sorted_results]
    bars = plt.bar(range(len(accuracies)), accuracies)
    plt.xticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel("Configuration")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Top 10 Configurations")
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{acc:.2f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(charts_dir / "top_configurations.png", dpi=150)
    plt.close()
