import torch
import torch.nn as nn

from config import HyperParams, TrainingHistory
from model import MNISTNet


def evaluate(model, images, labels, criterion, device) -> tuple[float, float]:
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        images = images.to(device)
        labels = labels.to(device)
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

    model = MNISTNet(params.hidden_size, params.num_layers).to(device)
    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    scaler = torch.amp.GradScaler()

    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    num_samples = len(train_images)

    history = TrainingHistory(params=params)

    for epoch in range(params.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, params.batch_size):
            batch_images = train_images[i : i + params.batch_size]
            batch_labels = train_labels[i : i + params.batch_size]

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
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

    history.val_loss, history.val_accuracy = evaluate(model, val_images, val_labels, criterion, device)
    return history
