import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import ExperimentConfig, HyperParams, TrainingHistory, load_config
from data import load_dataset
from evolution.advisor import propose_next_generation_option_lists_with_llm
from evolution.mini_grid import expand_option_lists, propose_next_generation_option_lists
from results import save_results
from search import generate_configs, run_configs, run_hyperparameter_search
from visualization import generate_charts


def print_configs_table(configs: list[HyperParams], title: str = "Configurations") -> None:
    """Pretty-print configs as an aligned table."""
    if not configs:
        return
    
    # Define columns: (header, getter, width, align)
    columns = [
        ("#", lambda i, c: str(i), 3, "right"),
        ("Model", lambda i, c: c.model_type.upper(), 5, "left"),
        ("LR", lambda i, c: f"{c.learning_rate:.0e}", 8, "right"),
        ("BS", lambda i, c: str(c.batch_size), 4, "right"),
        ("Ep", lambda i, c: str(c.epochs), 3, "right"),
        ("Hidden", lambda i, c: str(c.hidden_size), 6, "right"),
        ("Layers", lambda i, c: str(c.num_layers), 6, "right"),
    ]
    
    # Box-drawing chars
    TL, TR, BL, BR = "╭", "╮", "╰", "╯"
    H, V = "─", "│"
    TJ, BJ, LJ, RJ, X = "┬", "┴", "├", "┤", "┼"
    
    # Build separator lines
    widths = [w + 2 for _, _, w, _ in columns]  # +2 for padding
    top = TL + TJ.join(H * w for w in widths) + TR
    mid = LJ + X.join(H * w for w in widths) + RJ
    bot = BL + BJ.join(H * w for w in widths) + BR
    
    def fmt_cell(val: str, width: int, align: str) -> str:
        if align == "right":
            return val.rjust(width)
        return val.ljust(width)
    
    def make_row(cells: list[tuple[str, int, str]]) -> str:
        return V + V.join(f" {fmt_cell(v, w, a)} " for v, w, a in cells) + V
    
    # Header row
    header_cells = [(h, w, a) for h, _, w, a in columns]
    
    print(f"\n  {title}")
    print(f"  {top}")
    print(f"  {make_row(header_cells)}")
    print(f"  {mid}")
    
    # Data rows
    for i, cfg in enumerate(configs, 1):
        row_cells = [(getter(i, cfg), w, a) for _, getter, w, a in columns]
        print(f"  {make_row(row_cells)}")
    
    print(f"  {bot}")


def print_results_table(results: list[TrainingHistory], title: str = "Results") -> None:
    """Pretty-print results with accuracy as an aligned table."""
    if not results:
        return
    
    # Sort by accuracy descending for display
    sorted_results = sorted(enumerate(results, 1), key=lambda x: x[1].val_accuracy, reverse=True)
    
    # Define columns: (header, getter, width, align)
    columns = [
        ("Rank", lambda rank, i, r: str(rank), 4, "right"),
        ("#", lambda rank, i, r: str(i), 3, "right"),
        ("Model", lambda rank, i, r: r.params.model_type.upper(), 5, "left"),
        ("LR", lambda rank, i, r: f"{r.params.learning_rate:.0e}", 8, "right"),
        ("BS", lambda rank, i, r: str(r.params.batch_size), 4, "right"),
        ("Hidden", lambda rank, i, r: str(r.params.hidden_size), 6, "right"),
        ("Acc", lambda rank, i, r: f"{r.val_accuracy * 100:.2f}%", 7, "right"),
        ("Loss", lambda rank, i, r: f"{r.val_loss:.4f}", 7, "right"),
        ("Time", lambda rank, i, r: f"{r.wall_time_seconds:.1f}s", 7, "right"),
    ]
    
    # Box-drawing chars
    TL, TR, BL, BR = "╭", "╮", "╰", "╯"
    H, V = "─", "│"
    TJ, BJ, LJ, RJ, X = "┬", "┴", "├", "┤", "┼"
    
    # Build separator lines
    widths = [w + 2 for _, _, w, _ in columns]
    top = TL + TJ.join(H * w for w in widths) + TR
    mid = LJ + X.join(H * w for w in widths) + RJ
    bot = BL + BJ.join(H * w for w in widths) + BR
    
    def fmt_cell(val: str, width: int, align: str) -> str:
        if align == "right":
            return val.rjust(width)
        return val.ljust(width)
    
    def make_row(cells: list[tuple[str, int, str]], highlight: bool = False) -> str:
        row = V + V.join(f" {fmt_cell(v, w, a)} " for v, w, a in cells) + V
        if highlight:
            return f"\033[1;32m{row}\033[0m"  # Bold green for best
        return row
    
    # Header row
    header_cells = [(h, w, a) for h, _, w, a in columns]
    
    print(f"\n  {title}")
    print(f"  {top}")
    print(f"  {make_row(header_cells)}")
    print(f"  {mid}")
    
    # Data rows
    for rank, (orig_idx, res) in enumerate(sorted_results, 1):
        row_cells = [(getter(rank, orig_idx, res), w, a) for _, getter, w, a in columns]
        print(f"  {make_row(row_cells, highlight=(rank == 1))}")
    
    print(f"  {bot}")


def save_progress(
    output_dir: Path,
    config: ExperimentConfig,
    generation: int,
    total_generations: int,
    current_configs: list[HyperParams],
    results_so_far: list[TrainingHistory],
    option_lists: dict[str, list] | None = None,
    status: str = "running",
) -> None:
    """Save incremental progress for real-time monitoring."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    progress_data = {
        "experiment_name": config.name,
        "mode": config.mode,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "generation": {
            "current": generation + 1,
            "total": total_generations,
        },
        "current_configs": [asdict(c) for c in current_configs],
        "option_lists": option_lists,
        "results_so_far": [
            {
                "params": asdict(h.params),
                "epoch_losses": h.epoch_losses,
                "val_loss": h.val_loss,
                "val_accuracy": h.val_accuracy,
                "wall_time_seconds": h.wall_time_seconds,
                "param_count": h.param_count,
            }
            for h in results_so_far
        ],
        "best_accuracy": max((h.val_accuracy for h in results_so_far), default=0.0),
        "total_configs_evaluated": len(results_so_far),
    }
    
    with open(output_dir / "progress.json", "w") as f:
        json.dump(progress_data, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python main.py <config.toml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    print(f"Experiment: {config.name}")
    print(f"Seed: {config.seed}")
    if config.mode == "grid":
        print(f"Search space: {len(generate_configs(config.search_space))} combinations\n")
    else:
        print(
            "Mode: evolution\n"
            f"Generations: {config.evolution.generations}\n"
            f"Cap per generation: {config.evolution.cap_per_generation}\n"
            f"LLM: {'enabled' if config.evolution.llm.enabled else 'disabled'}\n"
        )

    print("Loading datasets...")
    train_images, train_labels = load_dataset("data/mnist/train_split.parquet")
    val_images, val_labels = load_dataset("data/mnist/val_split.parquet")
    print(f"Train: {len(train_images)}, Val: {len(val_images)}\n")

    output_dir = Path("experiments") / config.name
    
    if config.mode == "grid":
        results = run_hyperparameter_search(config, train_images, train_labels, val_images, val_labels)
    else:
        results = []
        for gen in range(config.evolution.generations):
            gen_seed = config.seed + gen
            if config.evolution.llm.enabled and config.evolution.llm.model:
                option_lists = propose_next_generation_option_lists_with_llm(
                    base_search_space=config.search_space,
                    results_so_far=results,
                    cap=config.evolution.cap_per_generation,
                    seed=gen_seed,
                    model=config.evolution.llm.model,
                    temperature=config.evolution.llm.temperature,
                    output_dir=output_dir,
                    generation=gen,
                )
            else:
                option_lists = propose_next_generation_option_lists(
                    base_search_space=config.search_space,
                    results_so_far=results,
                    cap=config.evolution.cap_per_generation,
                    seed=gen_seed,
                )

            gen_configs = expand_option_lists(
                option_lists=option_lists,
                cap=config.evolution.cap_per_generation,
                seed=gen_seed,
            )
            
            # Save progress before running this generation
            save_progress(
                output_dir=output_dir,
                config=config,
                generation=gen,
                total_generations=config.evolution.generations,
                current_configs=gen_configs,
                results_so_far=results,
                option_lists=option_lists,
                status="running",
            )
            
            print(f"\n{'═' * 60}")
            print(f"  ⚡ GENERATION {gen + 1}/{config.evolution.generations}")
            print(f"{'═' * 60}")
            
            print_configs_table(gen_configs, title="📋 Training Queue")
            
            gen_results = run_configs(config, gen_configs, train_images, train_labels, val_images, val_labels)
            
            print_results_table(gen_results, title="📊 Generation Results")
            
            # Show best so far
            best_this_gen = max(gen_results, key=lambda r: r.val_accuracy)
            print(f"\n  🏆 Best this generation: {best_this_gen.val_accuracy * 100:.2f}% accuracy")
            
            results.extend(gen_results)
        
        # Mark as completed
        save_progress(
            output_dir=output_dir,
            config=config,
            generation=config.evolution.generations - 1,
            total_generations=config.evolution.generations,
            current_configs=[],
            results_so_far=results,
            option_lists=None,
            status="completed",
        )
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
