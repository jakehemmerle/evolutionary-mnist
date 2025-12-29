import time

import torch
import torch.nn as nn

from config import HyperParams, TrainingHistory
from model import build_model


def _device_type(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def _maybe_add_channel(params: HyperParams, images: torch.Tensor) -> torch.Tensor:
    # CNN expects NCHW; MLP expects NHW.
    if params is not None and getattr(params, "model_type", "mlp") == "cnn":
        return images.unsqueeze(1)
    return images


def evaluate(model, images, labels, criterion, device, params: HyperParams | None = None) -> tuple[float, float]:
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=_device_type(device)):
        images = images.to(device)
        labels = labels.to(device)
        images = _maybe_add_channel(params, images)
        outputs = model(images)
        loss = criterion(outputs, labels).item()
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
    return loss, accuracy


def train_model(
    params: HyperParams,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    device: str,
    experiment_id: int,
    seed: int,
) -> TrainingHistory:
    torch.manual_seed(seed)

    start_t = time.perf_counter()
    model = build_model(params).to(device)
    model = torch.compile(model)
    param_count = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    scaler = torch.amp.GradScaler(enabled=_device_type(device) == "cuda")

    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    num_samples = len(train_images)

    history = TrainingHistory(params=params)
    history.param_count = param_count

    for epoch in range(params.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, params.batch_size):
            batch_images = train_images[i : i + params.batch_size]
            batch_labels = train_labels[i : i + params.batch_size]
            batch_images = _maybe_add_channel(params, batch_images)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=_device_type(device)):
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        history.epoch_losses.append(avg_loss)
        print(f"[Exp {experiment_id:2d}] [{device}] Epoch {epoch + 1}/{params.epochs} - Loss: {avg_loss:.4f}")

    # Validation
    val_images = val_images.to(device)
    val_labels = val_labels.to(device)
    val_images = _maybe_add_channel(params, val_images)
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=_device_type(device)):
        outputs = model(val_images)
        history.val_loss = criterion(outputs, val_labels).item()
        predictions = outputs.argmax(dim=1)
        history.val_accuracy = (predictions == val_labels).float().mean().item()

    history.wall_time_seconds = time.perf_counter() - start_t
    return history
