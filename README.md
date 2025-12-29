# MNIST

## Dataset

The MNIST dataset is downloaded from Hugging Face: https://huggingface.co/datasets/ylecun/mnist

### Download Command

```bash
uvx --from huggingface_hub hf download ylecun/mnist --repo-type dataset --local-dir data
```

### Dataset Information

- **Source**: [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist) on Hugging Face
- **License**: MIT
- **Format**: Parquet files
- **Size**: 70,000 images total (60,000 train, 10,000 test)
- **Image dimensions**: 28x28 grayscale
- **Classes**: 10 (digits 0-9)

### Files

- `data/mnist/train-00000-of-00001.parquet` - Training set (60,000 images)
- `data/mnist/test-00000-of-00001.parquet` - Test set (10,000 images)
