import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

LR= 1e-3 # Adam learning rate
WEIGHT_DECAY  = 1e-4  # L2 regularisation coefficient
input_steps= 72
output_steps=12
input_size=14
hidden_size=128
target_index= 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=pd.read_csv(r"C:\Personal\comp_proj\spider_ml_task_2\jena_climate_dataset.csv")
data = data.drop(columns=["Date Time"])
data = data.astype(np.float32)
data=data.to_numpy()
data=np.array(data)

#normalisation
mean= data.mean(axis=0)
std= data.std(axis=0)
std[std==0]=1
data= (data-mean)/std

train_data= data[:int(len(data)*0.3)]
val_data= data[int(len(data)*0.8):]

class JenaClimateForecastDataset(Dataset):
    def __init__(self, data, input_steps, output_steps, target_index=1):
        """
        data: Numpy array of shape (num_samples, num_features)
        target_index: Column index of the feature you want to predict (e.g., Temperature)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.target_index = target_index

    def __len__(self):
        # Subtract total window size to prevent out-of-bounds slicing
        return len(self.data) -self.input_steps - self.output_steps + 1

    def __getitem__(self, i):
        # Past 72 hours (all features)
        x = self.data[i : i + self.input_steps]

        # Future 12 hours (only target feature, e.g., temperature)
        y = self.data[i + self.input_steps : i + self.input_steps + self.output_steps, self.target_index]

        return x, y

train_dataset= JenaClimateForecastDataset(train_data, input_steps, output_steps, target_index)
val_dataset= JenaClimateForecastDataset(val_data, input_steps, output_steps, target_index)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class TempPrediction(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_steps = output_steps
        self.dropout= nn.Dropout(0.2)

        limit1= np.sqrt(6 /(input_size + hidden_size))
        limit2= np.sqrt(6/(2*hidden_size))

        self.W1 = nn.Parameter(torch.FloatTensor(4*hidden_size, input_size+hidden_size).uniform_(-limit1, limit1))
        self.W2 = nn.Parameter(torch.FloatTensor(4*hidden_size, 2*hidden_size).uniform_(-limit2, limit2))
        self.b1= nn.Parameter(torch.cat([
            torch.ones(hidden_size, 1),    # forget bias = 1
            torch.zeros(hidden_size, 1),
            torch.zeros(hidden_size, 1),
            torch.zeros(hidden_size, 1)
        ]))

        self.b2 = nn.Parameter(torch.cat([
            torch.ones(hidden_size, 1),
            torch.zeros(hidden_size, 1),
            torch.zeros(hidden_size, 1),
            torch.zeros(hidden_size, 1)
        ]))

        # output projection
        self.out = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size = x.size(0)
        h1 = torch.zeros(batch_size, self.hidden_size, 1).to(x.device)
        c1 = torch.zeros(batch_size, self.hidden_size, 1).to(x.device)
        h2= torch.zeros(batch_size, self.hidden_size, 1).to(x.device)
        c2 = torch.zeros(batch_size, self.hidden_size, 1).to(x.device)

        for t in range(x.size(1)):
            x_t1 = x[:, t, :].unsqueeze(2)          # (batch, input_size, 1)
            xh1= torch.cat([x_t1, h1], dim=1)        # (batch, input_size+hidden_size, 1)

            gates1 = self.W1 @ xh1 + self.b1          # ONE matmul instead of 4
            f_t1, i_t1, cand1, o_t1 = gates1.chunk(4, dim=1)
            f_t1= torch.sigmoid(f_t1) #matrix multiplication
            i_t1= torch.sigmoid(i_t1)
            cand1 = torch.tanh(cand1)
            o_t1= torch.sigmoid(o_t1)

            c1= f_t1*c1 + i_t1 * cand1
            h1 =o_t1*torch.tanh(c1)
            h1= self.dropout(h1)

            xh2= torch.cat([h1, h2], dim=1)        # (batch, input_size+hidden_size, 1)

            gates2 = self.W2 @xh2 + self.b2         # ONE matmul instead of 4
            f_t2, i_t2, cand2, o_t2 = gates2.chunk(4, dim=1)
            f_t2= torch.sigmoid(f_t2) #matrix multiplication
            i_t2= torch.sigmoid(i_t2)
            cand2 = torch.tanh(cand2)
            o_t2= torch.sigmoid(o_t2)

            c2= f_t2* c2 + i_t2 * cand2
            h2 =o_t2*torch.tanh(c2)

        # h shape: (batch, hidden_size, 1) → squeeze → (batch, hidden_size)
        return self.out(h2.squeeze(2))             # (batch, output_steps)

def train(model, loader, criterion, optimizer, device):
    """Run one full pass over the training set; return avg loss & accuracy."""
    model.train()
    total_loss,total = 0.0, 0

    for input_batch, output_batch in loader:
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)

        optimizer.zero_grad()
        preds = model.forward(input_batch)
        loss = criterion(preds, output_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()*input_batch.size(0)
        total+= input_batch.size(0)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0

    for input_batch, output_batch in loader:
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)

        preds = model.forward(input_batch)
        loss = criterion(preds, output_batch)

        total_loss += loss.item()*input_batch.size(0)
        total+= input_batch.size(0)

    return total_loss / total


model = TempPrediction(input_size, hidden_size, output_steps).to(DEVICE)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

history = {
    "train_loss": [], "val_loss": []
}

best_val_loss= float("inf")

for epoch in range(1, 41):
    train_loss=train(model, train_loader, criterion, optimizer, DEVICE)
    val_loss=evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    if epoch%5==0 or epoch==1:
        print(f"Epoch: {epoch} | Training Loss: {train_loss} | Validation Loss: {val_loss}")

    # Track best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {}
        for k, v in model.state_dict().items():
            best_state[k] = v.cpu().clone()
print("\n")
print(f"Best validation loss: {best_val_loss}")

#Save model weights
weights_path = "lstm_weights.pkl"
with open(weights_path, "wb") as f:
    pickle.dump(best_state, f)
print(f"Best model weights saved: {weights_path}")

model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

#Training curves
epochs_range = range(1, 41)

plt.figure(figsize=(6, 6))
plt.plot(epochs_range, history["train_loss"], "b-o", markersize=4, label="Train Loss")
plt.plot(epochs_range, history["val_loss"],   "r-s", markersize=4, label="Val Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("training_curve_lstm.png")
plt.show()
print("Plot saved successfully: training_curve_lstm.png")

#Scatter plot
model.eval()
all_preds, all_targets= [], []

with torch.no_grad():
    for input_batch, output_batch in val_loader:
        input_batch= input_batch.to(DEVICE)
        preds = model.forward(input_batch)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(output_batch.cpu().numpy())

# stack all batches together
all_preds = np.concatenate(all_preds, axis=0)    # (num_windows, 12)
all_targets = np.concatenate(all_targets, axis=0)  # (num_windows, 12)

temp_mean= mean[target_index]
temp_std= std[target_index]
all_preds= all_preds*temp_std+temp_mean #denormalise back to celsius
all_targets= all_targets*temp_std+temp_mean

def mae_loss(pred, true):
    return np.mean(np.abs(pred - true))

def huber_loss(pred, true, delta=1.0):
    err = pred - true
    abs_err = np.abs(err)
    quad = 0.5 * err**2
    lin = delta * (abs_err - 0.5 * delta)
    return np.mean(np.where(abs_err <= delta, quad, lin))

mse_val = np.mean((all_preds - all_targets) ** 2)
mae_val = mae_loss(all_preds, all_targets)
huber_val = huber_loss(all_preds, all_targets)

print(f"LSTM | MSE: {mse_val:.4f}  MAE: {mae_val:.4f}  Huber: {huber_val:.4f}")

# save for compare.py
np.savez("lstm_eval.npz", preds=all_preds, targets=all_targets)

#Final evaluation on validation set
val_loss= evaluate(model, val_loader, criterion, DEVICE)
print(f"Validation Loss: {val_loss}")