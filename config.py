from dataclasses import dataclass, field

import tomli


@dataclass
class HyperParams:
    learning_rate: float
    epochs: int
    batch_size: int
    hidden_size: int
    num_layers: int

    def __str__(self) -> str:
        return (
            f"lr={self.learning_rate}, epochs={self.epochs}, "
            f"batch={self.batch_size}, hidden={self.hidden_size}, layers={self.num_layers}"
        )


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    search_space: dict[str, list]
    num_gpus: int
    workers_per_gpu: int


@dataclass
class TrainingHistory:
    params: HyperParams
    epoch_losses: list[float] = field(default_factory=list)
    val_loss: float = 0.0
    val_accuracy: float = 0.0


def load_config(path: str) -> ExperimentConfig:
    with open(path, "rb") as f:
        data = tomli.load(f)

    return ExperimentConfig(
        name=data["experiment"]["name"],
        seed=data["experiment"]["seed"],
        search_space=data["search_space"],
        num_gpus=data["execution"]["num_gpus"],
        workers_per_gpu=data["execution"]["workers_per_gpu"],
    )
