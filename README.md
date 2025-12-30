# evolutionary-mnist

This repository demonstrates hyperparameter tuning via LLM for a two layer CNN on the MNIST dataset. It is also an example of something that can be done, but probably shouldn't.

Uses:
- [PyTorch](https://pytorch.org/) for training the model
- [OpenRouter](https://openrouter.ai/) for LLM API calls

## Example run

Validation accuracy:

![Accuracy per generation](experiments/evo-mini-v3/charts/accuracy_per_generation.png)

Reminder: no regression is used here. The improvements learned come strictly from the LLM's reasoning and analysis on the previous runs:

![Accuracy vs learning rate](experiments/evo-mini-v3/charts/accuracy_vs_lr.png)

Example LLM reasoning (which leaves a lot of room for improvement):

![Screenshot](experiments/evo-mini-v3/charts/Screenshot%202025-12-30%20at%203.51.59%E2%80%AFPM.png)

## Setup

```bash
uv sync
```

The MNIST dataset is downloaded from Hugging Face: https://huggingface.co/datasets/ylecun/mnist

```bash
uvx --from huggingface_hub hf download ylecun/mnist --repo-type dataset --local-dir data
```

Use the `scripts/prepare_data.py` script to split the dataset into train and validation sets.

## Running Experiments

This example runs for 5 generations with 4 training runs per generation.

```bash
uv run evolutionary-mnist experiments/evo-mini-v3.toml
```

## Future work:
- Improve system prompt.
- Neural Architecture Search (NAS) for on-the-fly architecture exploration.
- Keep training time constant per run within each generation.
