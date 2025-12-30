import io

import pandas as pd
import torch
from PIL import Image


def load_image(image_data: dict) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_data["bytes"])).convert("L")
    tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(28, 28)
    return tensor / 255.0


def load_dataset(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_parquet(path)
    images = torch.stack([load_image(row["image"]) for _, row in df.iterrows()])
    labels = torch.tensor(df["label"].tolist())
    return images, labels
