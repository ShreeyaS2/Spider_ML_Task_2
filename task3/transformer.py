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

LR= 5e-4 # Adam learning rate
WEIGHT_DECAY  = 5e-4  # L2 regularisation coefficient
input_steps= 72
output_steps=12
heads= 4
num_layers=3
input_size=14
d_model= 64 #reasonable hidden size for the input steps 
forward_exp= d_model*4
target_index= 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=pd.read_csv('/content/jena_climate_dataset.csv')
data = data.drop(columns=["Date Time"])
data = data.reset_index(drop=True)          # ensure index starts at 0
data= data.groupby(data.index//6).mean()
data = data.astype(np.float32)
data=data.to_numpy()
data=np.array(data)

#normalisation
mean= data.mean(axis=0)
std= data.std(axis=0)
std[std==0]=1
data= (data-mean)/std

train_data= data[:int(len(data)*0.8)]
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
        return len(self.data) -self.input_steps - self.output_steps + 1

    def __getitem__(self, i):
        # Past 72 hours (all features)
        x = self.data[i : i + self.input_steps]

        # Future 12 hours (only target feature, e.g., temperature)
        y = self.data[i + self.input_steps : i + self.input_steps + self.output_steps, self.target_index]

        return x, y

train_dataset= JenaClimateForecastDataset(train_data, input_steps, output_steps, target_index)
val_dataset= JenaClimateForecastDataset(val_data, input_steps, output_steps, target_index)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std= x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta #applying a learnt scale and shift that will actually be more useful on a per-feature basis than mean 0 and std 1- applyed across hidden_size for every timestep of every batch

class Attention(nn.Module):
    def __init__(self, d_model, heads):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        if(self.d_model%self.heads==0):
            self.d_k = self.d_model//self.heads
        else:
            print("d_model must be divisible by heads")

        self.values= nn.Linear(self.d_model, self.d_model, bias=False)
        self.keys= nn.Linear(self.d_model, self.d_model, bias=False)
        self.queries= nn.Linear(self.d_model, self.d_model, bias=False)
        self.out= nn.Linear(heads*self.d_k, d_model)

    def forward(self, q, k, v, mask):
        n=q.shape[0]
        v=self.values(v)
        k=self.keys(k)
        q=self.queries(q)
        key_len, query_len, val_len = k.shape[1], q.shape[1], v.shape[1]

        v=v.reshape(n, val_len, self.heads, self.d_k)
        q=q.reshape(n, query_len, self.heads, self.d_k)
        k=k.reshape(n, key_len, self.heads, self.d_k)
        attention= torch.einsum("nqhd, nkhd->nhqk", [q, k])

        if mask is not None:
            attention= attention.masked_fill(mask==0, float("-1e20"))

        att_fin = torch.softmax(attention/(self.d_k**0.5), dim=3) 

        out = torch.einsum("nhqk, nkhd->nqhd", [att_fin, v]).reshape(n, query_len, self.heads*self.d_k)
        out_fin= self.out(out)
        return out_fin


class Encoder(nn.Module):
    def __init__(self, heads, d_model, forward_exp):
        super(Encoder, self).__init__()
        self.attention= Attention(d_model, heads)
        self.feedforward= nn.Sequential(
            nn.Linear(d_model, forward_exp),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(forward_exp, d_model)
        )
        self.dropout= nn.Dropout(0.2)
        self.layernorm1= LayerNorm(d_model, eps=1e-6)
        self.layernorm2= LayerNorm(d_model, eps=1e-6)

    def forward(self, q,k,v,mask):
        attention= self.attention.forward(q, k, v, mask)
        x1= self.layernorm1.forward(q+attention)
        x2=self.dropout(x1)
        f= self.feedforward(x2)
        x= f+x2
        x3= self.layernorm2.forward(x)
        x4= self.dropout(x3)
        return x4

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=input_steps):
        super(PositionalEncoding, self).__init__()
        pe= torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float() #(max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x):
        return x +self.pe[:, :x.size(1), :].to(x.device) 


class Transformer(nn.Module):
    def __init__(self, input_size, d_model, heads, num_layers, output_steps):
        super(Transformer, self).__init__()
        self.input= nn.Linear(input_size, d_model)
        self.pe= PositionalEncoding(d_model)
        self.dropout= nn.Dropout(0.1)
        self.layers= nn.ModuleList([
            Encoder(heads, d_model, forward_exp)
            for _ in range(num_layers)
        ])
        self.out= nn.Linear(d_model, output_steps)

    def forward(self, x):
        x = self.input(x)    
        x = self.pe(x)        
        x = self.dropout(x)
        for layer in self.layers:
            x = layer.forward(x, x, x, None)       
        x = x[:, -1, :]            # take last timestep (batch, d_model)
        return self.out(x)         # (batch, 12)

def train(model, loader, criterion, optimizer, device):
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


model = Transformer(input_size, d_model, heads, num_layers, output_steps).to(DEVICE)
criterion = nn.MSELoss()

warmup_epochs= 5
target_lr=5e-4

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

history = {
    "train_loss": [], "val_loss": []
}

best_val_loss= float("inf") 

for epoch in range(1, 41):
    if epoch <= warmup_epochs:
        warmup_lr = target_lr* (epoch/ warmup_epochs)
        set_lr(optimizer, warmup_lr)

    train_loss=train(model, train_loader, criterion, optimizer, DEVICE)
    val_loss=evaluate(model, val_loader, criterion, DEVICE)
    if epoch > warmup_epochs:
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
weights_path = "transformer_weights.pkl"
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
plt.savefig("training_curve_transformer.png")
plt.show()
print("Plot saved successfully: training_curve_transformer.png")

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
all_preds = np.concatenate(all_preds, axis=0)    
all_targets = np.concatenate(all_targets, axis=0)  

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

print(f"Transformer | MSE: {mse_val:.4f}  MAE: {mae_val:.4f}  Huber: {huber_val:.4f}")

# save for compare.py
np.savez("transformer_eval.npz", preds=all_preds, targets=all_targets)

#Final evaluation on validation set
val_loss= evaluate(model, val_loader, criterion, DEVICE)
print(f"Validation Loss: {val_loss}")
