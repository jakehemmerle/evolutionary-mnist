from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from config import ExperimentConfig, HyperParams, TrainingHistory
from training import train_model


def run_single_experiment(args: tuple) -> TrainingHistory:
    params, train_images, train_labels, val_images, val_labels, device, experiment_id, seed = args

    history = train_model(
        params, train_images, train_labels, val_images, val_labels, device, experiment_id, seed
    )
    return history


def run_configs(
    config: ExperimentConfig,
    configs: list[HyperParams],
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
) -> list[TrainingHistory]:
    total_workers = config.num_gpus * config.workers_per_gpu
    print(f"Running {len(configs)} experiments with {total_workers} workers ({config.workers_per_gpu} per GPU)...\n")

    args_list = [
        (
            params,
            train_images,
            train_labels,
            val_images,
            val_labels,
            f"cuda:{i % config.num_gpus}",
            i + 1,
            config.seed,
        )
        for i, params in enumerate(configs)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=total_workers) as executor:
        futures = {executor.submit(run_single_experiment, args): args[0] for args in args_list}

        for future in as_completed(futures):
            history = future.result()
            results.append(history)
            print(f"[Done] {history.params} -> Accuracy: {history.val_accuracy * 100:.2f}%")

    return results
