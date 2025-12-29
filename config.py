from dataclasses import dataclass, field

import tomli


@dataclass
class HyperParams:
    learning_rate: float
    epochs: int
    batch_size: int
    cnn_channels: int = 32
    cnn_kernel_size: int = 3
    cnn_dropout: float = 0.0
    cnn_fc_hidden: int = 128

    def __str__(self) -> str:
        return (
            f"lr={self.learning_rate}, epochs={self.epochs}, batch={self.batch_size}, "
            f"ch={self.cnn_channels}, k={self.cnn_kernel_size}, drop={self.cnn_dropout}, fc={self.cnn_fc_hidden}"
        )


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    schema: dict[str, dict]
    num_gpus: int
    workers_per_gpu: int
    evolution: "EvolutionConfig" = field(default_factory=lambda: EvolutionConfig())


@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = ""
    temperature: float = 0.2


@dataclass
class EvolutionConfig:
    generations: int = 5
    cap_per_generation: int = 8
    elite_k: int = 4
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class TrainingHistory:
    params: HyperParams
    epoch_losses: list[float] = field(default_factory=list)
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    wall_time_seconds: float = 0.0
    param_count: int = 0


def load_config(path: str) -> ExperimentConfig:
    with open(path, "rb") as f:
        data = tomli.load(f)

    evo = data.get("evolution", {}) or {}
    llm = evo.get("llm", {}) or {}

    return ExperimentConfig(
        name=data["experiment"]["name"],
        seed=data["experiment"]["seed"],
        schema=data["schema"],
        num_gpus=data["execution"]["num_gpus"],
        workers_per_gpu=data["execution"]["workers_per_gpu"],
        evolution=EvolutionConfig(
            generations=int(evo.get("generations", 5)),
            cap_per_generation=int(evo.get("cap_per_generation", 8)),
            elite_k=int(evo.get("elite_k", 4)),
            llm=LLMConfig(
                enabled=bool(llm.get("enabled", True)),
                model=str(llm.get("model", "")),
                temperature=float(llm.get("temperature", 0.2)),
            ),
        ),
    )
