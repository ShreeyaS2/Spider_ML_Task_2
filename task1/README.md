# CIFAR-10 Image Classification — Custom ResNet-CNN

A custom ResNet-CNN trained from scratch on CIFAR-10, achieving **~91% validation accuracy** over 100 epochs.


## Dataset

**CIFAR-10** — 60,000 32×32 colour images, 10 classes.

Normalised using precomputed CIFAR-10 per-channel mean and std. Training augmentation: RandomHorizontalFlip, RandomCrop(32, padding=4), ColorJitter, RandomErasing. Validation uses normalisation only.


## Architecture

3-stage residual network, `2,3,3` block configuration (8 residual blocks, 16 conv layers).

**BasicBlock:**
```
Conv 3×3 → BN → ReLU → Conv 3×3 → BN → +shortcut → ReLU → Dropout2d
```
The second conv has no ReLU before the residual add — ReLU is applied after, following the standard ResNet BasicBlock design. Projection shortcuts (1×1 conv + BN) used wherever channels change.

| Stage | Blocks | Channels | Downsampling |
|---|---|---|---|
| 1 | 2 | 32 | MaxPool2d |
| 2 | 3 | 64 | MaxPool2d |
| 3 | 3 | 128 | AdaptiveAvgPool2d(1,1) |

Classifier: `Flatten → Linear(128, 256) → BN → ReLU → Dropout(0.3) → Linear(256, 10)`

`bias=False` on all Conv2d layers — BatchNorm's mean subtraction makes the conv bias redundant. Kaiming/He normal initialisation throughout.


## Training

| Setting | Value |
|---|---|
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| Loss | CrossEntropyLoss, label_smoothing=0.1 |
| Scheduler | CosineAnnealingLR, T_max=100 |
| Batch size | 64 |
| Epochs | 100 |


## Results

**90.77% validation accuracy.** Best performing classes: automobile (96.7%), truck (94.4%). Hardest: cat (78.9%) and dog (84.5%) — frequently confused due to similar textures, a known CIFAR-10 challenge.

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Run: `python cnn.py` — CIFAR-10 downloads automatically to `./data`.
