# Weather Forecasting — LSTM vs Transformer Comparison
 
A from-scratch implementation and comparative study of an LSTM and Transformer for 12-hour temperature forecasting on the Jena Climate Dataset.
 
---
 
## Task Overview
 
Both models are trained to forecast the next **12 hours of temperature** given the previous **72 hours of weather observations** (14 features, hourly resolution after downsampling every 6 timesteps).
 
---
 
## Dataset
 
**Jena Climate Dataset** — meteorological readings from a weather station in Jena, Germany, originally recorded at 10-minute intervals. Downsampled to hourly by averaging every 6 consecutive rows.
 
**Features (14 total):** T (degC), p (mbar), Tpot (K), Tdew (degC), rh (%), VPmax (mbar), VPact (mbar), VPdef (mbar), sh (g/kg), H2OC (mmol/mol), rho (g/m³), wv (m/s), max. wv (m/s), wd (deg)
 
**Preprocessing:**
- Date Time column dropped
- Downsampled from 10-minute to hourly by averaging every 6 rows
- Z-score normalised per feature (mean 0, std 1), std clipped to minimum 1 to avoid division by zero
- 80/20 chronological train/val split
**Windowing:**
- Input: 72 consecutive hourly observations (all 14 features)
- Target: next 12 hours of temperature only (column index 1)
---
 
## Architecture
 
### LSTM (from scratch)
 
A 2-layer stacked LSTM implemented entirely using `nn.Parameter` and PyTorch tensor operations — no `nn.LSTM`, `nn.GRU`, or any built-in recurrent module.
 
**Gates implemented manually:**
- Forget gate: `f = σ(W_f · [x, h] + b_f)`: Decides what to erase from cell state
- Input gate: `i = σ(W_i · [x, h] + b_i)`: Decides what new information to write
- Candidate cell state: `ĉ = tanh(W_c · [x, h] + b_c)`: Proposed new values
- Cell state update: `c_t = f ⊙ c_{t-1} + i ⊙ ĉ`: Core memory equation
- Output gate: `o = σ(W_o · [x, h] + b_o)`: Controls what is exposed as hidden state
- Hidden state: `h_t = o ⊙ tanh(c_t)`


**Design choices:**
- Forget gate bias initialised to 1 to encourage remembering early in training
- Xavier uniform initialisation for all gate weight matrices
- Gate weights fused into single matrix multiply per layer (4 gates → 1 matmul) for efficiency
- Dropout of 0.2 applied to hidden state after each layer
- Final hidden state `h_t` at the last timestep passed through a linear projection to produce all 12 predictions simultaneously.


| Hyperparameter | Value |
|---|---|
| Hidden size | 64 |
| Layers | 2 |
| Dropout | 0.2 |
| Input steps | 72 |
| Output steps | 12 |
 
### Transformer
 
This is an encoder-only Transformer using self-attention, implemented from scratch.
 
**Components implemented manually:**
- Multi-head self-attention with Q/K/V projections (`nn.Linear`, bias=False)
- Scaled dot-product attention via `torch.einsum`
- Two separate `LayerNorm` instances per encoder block (post-attention, post-feedforward)
- Feedforward network: Linear → ReLU → Dropout → Linear (expansion factor 4)
- Residual connections around both attention and feedforward sub-layers
- Sinusoidal positional encoding injected after input projection
- Input projection: `nn.Linear(input_size, d_model)` to project 14 continuous features into model dimension
 
**Output:** mean pooling across all 72 timestep representations → linear projection to 12 predictions.
 
| Hyperparameter | Value |
|---|---|
| d_model | 64 |
| Heads | 4 (16 dims per head) |
| Encoder layers | 3 |
| Feedforward expansion | 4× (256) |
| Dropout | 0.2 |
| Input steps | 72 |
| Output steps | 12 |
 
---
 
## Training Setup
 
Both models trained under identical conditions for a fair comparison:
 
| Setting | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | LSTM: 1e-3 / Transformer: 5e-4 |
| Weight decay | 5e-4 |
| Batch size | 256 |
| Epochs | 40 |
| Loss function | MSELoss |
| Gradient clipping | max_norm=1.0 |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Best model tracking | Saved on lowest val loss |
 
The transformer uses a lower learning rate than the LSTM since transformer training is more sensitive to large initial learning rates. Weight decay was increased to `5e-4` from `1e-4` because the downsampled hourly dataset is smaller than the original 10-minute dataset (~126,000 windows), creating a higher overfitting risk for both architectures.
Also, multiple combinations were tested to see which gave the most accuracy. It was seen that a warmup helped improve prediction. The hidden_size was also kept as 64 to match with the LSTM and also because a bigger value would have caused excessive expansion and compression for a smaller portion of the dataset. The warmup was also tested for both a weight decay of 1e-3 and 5e-4, and it supported 5e-4.
 
---
 
## Results
 
| Metric | LSTM | Transformer |
|---|---|---|
| Best val loss (normalised MSE) | 0.0542 | 0.0736 |
| MSE (°C²) | 3.84 | 5.21 |
| MAE (°C) | 1.45 | 1.63 |
| Huber loss | 1.04 | 1.21 |
| RMSE (°C) | ~1.96 | ~2.28 |
 
The LSTM outperforms the Transformer across every metric on this task.


As a note, the 720/24 input window was also tried.
 
---
 
## Analysis
 
### Why attention helps sequence modeling
 
Attention computes a direct, learned weighted connection between any two positions in a sequence due to parallel computation. In an LSTM, information from early timesteps must propagate through every intermediate hidden state to influence later ones — each step applies sigmoid and tanh gates that mitigate vanishing gradients. Attention sidesteps this by letting any timestep directly attend to any other, regardless of distance, with no intermediate decay.


### Why LSTM outperforms Transformer here
 
At `input_steps=72`, the LSTM's built-in inductive bias (recency, sequential locality) aligns well with the structure of hourly weather data with short-range dependencies, and the LSTM's sequential hidden state naturally encodes this without needing to learn it from data. The Transformer has to learn these relationships entirely through attention weights, which requires more training data and longer sequences for its flexibility to pay off.
 
At `input_steps=720`, the attention matrix grows to 720×720 per head, consuming substantially more memory and computation, making the LSTM computationally more practical at longer sequence lengths that have short-range dependencies. Despite parallelised processing, the attention cost can exceed the benefit of parallelism at long sequences, where the LSTM's linear scaling per step becomes advantageous.
 
gh examples and training data to learn relationships from scratch
- When training speed matters, as parallel computation means faster predictions.


### Training stability
 
The LSTM trained more stably from epoch 1, with smooth loss curves and close train/val tracking throughout. The Transformer showed a higher initial loss (0.51 vs 0.19 at epoch 1) and slightly more validation noise in early epochs — characteristic of attention-based models without a warmup phase, where randomly initialised attention weights interact poorly with a fixed learning rate early in training. Gradient clipping (`max_norm=1.0`) was applied to both models to guard against gradient spikes, which is particularly important for the LSTM, where backpropagation through 72 sequential timesteps accumulates gradients multiplicatively across the recurrent chain.
 
### Forecasting challenges
 
- Temperature extremes (very cold or very hot days) are rare in training data; both models showed more scattered plots at the very extremes, with the transformer predictions scattering more than the LSTM.
- Not all features are integer values such as date-time, so before processing the dataset, this column had to be omitted, as dividing by std during normalisation would cause an error. Due to this, any value of std becoming 0 is also changed to 1.
- Normalising all 14 features to comparable scales is critical — without normalisation, features with large absolute values (e.g. pressure ~1000 mbar) would dominate the loss and prevent other features from contributing meaningfully to predictions
---
 
## Files
 
```
lstm.py                        — from-scratch LSTM implementation and training
transformer.py                 — from-scratch Transformer implementation and training
compare.py                     — side-by-side comparison plot and metrics
lstm_weights.pkl               — best LSTM model weights
transformer_weights.pkl        — best Transformer model weights
lstm_eval.npz                  — LSTM predictions and targets on val set (denormalised, °C)
transformer_eval.npz           — Transformer predictions and targets on val set (denormalised, °C)
training_curve_lstm.png        — LSTM loss curves
training_curve_transformer.png — Transformer loss curves
comparison.png                 — combined scatter plot, both models on same val windows
```
 
---
 
## Requirements
 
```
torch
numpy
pandas
matplotlib
scikit-learn
```
---
 
## How to Run
 
```bash
# Train LSTM
python lstm.py
 
# Train Transformer
python transformer.py
 
# Generate comparison plot and metrics
python compare.py

