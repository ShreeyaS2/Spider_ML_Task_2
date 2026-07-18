import numpy as np
import matplotlib.pyplot as plt

lstm_data = np.load(r"C:\Personal\comp_proj\spider_ml_task_2\task2\lstm_eval.npz")
trans_data = np.load(r"C:\Personal\comp_proj\spider_ml_task_2\task3\transformer_eval.npz")

lstm_preds, lstm_targets = lstm_data["preds"], lstm_data["targets"]
trans_preds, trans_targets = trans_data["preds"], trans_data["targets"]

print(f"LSTM preds shape: {lstm_preds.shape}")
print(f"Transformer preds shape: {trans_preds.shape}")
print()
print(f"LSTM normalized MSE: {np.mean((lstm_preds - lstm_targets)**2):.4f}")
print(f"Transformer normalized MSE: {np.mean((trans_preds - trans_targets)**2):.4f}")
print()
print(f"LSTM preds range: {lstm_preds.min():.2f} to {lstm_preds.max():.2f}")
print(f"Transformer preds range: {trans_preds.min():.2f} to {trans_preds.max():.2f}")
print()
print(f"Targets range (LSTM file): {lstm_targets.min():.2f} to {lstm_targets.max():.2f}")
print(f"Targets range (transformer file): {trans_targets.min():.2f} to {trans_targets.max():.2f}")

# Aggregate scatter
plt.figure(figsize=(7,7))
plt.scatter(trans_targets.flatten(), trans_preds.flatten(), alpha=0.11, s=1, color="green", label="Transformer")
plt.scatter(lstm_targets.flatten(), lstm_preds.flatten(), alpha=0.08, s=1, color="blue", label="LSTM")
lims = [lstm_targets.min(), lstm_targets.max()]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual (°C)")
plt.ylabel("Predicted (°C)")
plt.title("Predicted vs Actual Temperatures — LSTM vs Transformer")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("comparison.png")
plt.show()
print("Plot saved successfully: comparison.png")
