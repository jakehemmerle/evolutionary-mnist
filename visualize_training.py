"""Visualize MNIST training metrics across epochs."""

import matplotlib.pyplot as plt

# Training data from 50 epochs
epochs = list(range(1, 51))

train_loss = [
    0.9179, 0.4269, 0.3514, 0.3088, 0.2836, 0.2661, 0.2529, 0.2428, 0.2347, 0.2281,
    0.2224, 0.2177, 0.2136, 0.2099, 0.2065, 0.2033, 0.2004, 0.1976, 0.1952, 0.1928,
    0.1906, 0.1887, 0.1868, 0.1849, 0.1833, 0.1817, 0.1802, 0.1787, 0.1774, 0.1761,
    0.1749, 0.1737, 0.1726, 0.1716, 0.1706, 0.1697, 0.1687, 0.1678, 0.1667, 0.1660,
    0.1651, 0.1643, 0.1635, 0.1628, 0.1623, 0.1615, 0.1608, 0.1603, 0.1596, 0.1590,
]

val_loss = [
    0.4711, 0.3878, 0.3436, 0.3162, 0.2966, 0.2818, 0.2710, 0.2592, 0.2510, 0.2457,
    0.2423, 0.2387, 0.2367, 0.2353, 0.2345, 0.2336, 0.2325, 0.2312, 0.2305, 0.2296,
    0.2297, 0.2290, 0.2283, 0.2278, 0.2276, 0.2275, 0.2272, 0.2267, 0.2267, 0.2267,
    0.2264, 0.2268, 0.2266, 0.2270, 0.2270, 0.2273, 0.2283, 0.2284, 0.2282, 0.2284,
    0.2288, 0.2295, 0.2298, 0.2303, 0.2304, 0.2302, 0.2316, 0.2319, 0.2342, 0.2348,
]

val_accuracy = [
    86.72, 89.70, 90.38, 91.27, 91.72, 92.08, 92.20, 92.73, 92.88, 92.90,
    92.85, 93.07, 93.10, 93.15, 93.12, 93.23, 93.32, 93.35, 93.45, 93.38,
    93.38, 93.50, 93.48, 93.55, 93.60, 93.47, 93.57, 93.65, 93.70, 93.60,
    93.58, 93.57, 93.58, 93.63, 93.60, 93.65, 93.63, 93.67, 93.72, 93.70,
    93.68, 93.65, 93.67, 93.67, 93.72, 93.67, 93.67, 93.62, 93.60, 93.67,
]

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Training and Validation Loss
axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Validation Accuracy
axes[1].plot(epochs, val_accuracy, 'g-', label='Val Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150)
plt.show()

print("Plot saved to training_metrics.png")
