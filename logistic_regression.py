import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
import numpy as np

wandb.login()
config = {
    "learning_rate": 0.00005,
    "epochs": 80
}
wandb.init(project="logistic_regression", config=config)

embeddings = np.load("/om/user/oliviat/embeddings640.npy")
labels = np.load("/om/user/oliviat/labels640.npy")

train_data, rest_data, train_labels, rest_labels = train_test_split(embeddings, labels, test_size=0.3, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(rest_data, rest_labels, test_size=0.5, random_state=42)

# experiment with scaling the embeddings
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

train_data = torch.tensor(train_data, dtype=torch.float32)
val_data = torch.tensor(val_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=8, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=8, shuffle=False)
test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=8, shuffle=False)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_data.shape[1]
model = LogisticRegression(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

num_epochs = 80
for epoch in range(num_epochs):
    model.train()
    training_loss, train_correct, train_total = 0,0,0
    for batch_data, batch_labels in train_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_data).squeeze()
        loss = criterion(outputs, batch_labels.float())
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
        predicted = (outputs >= 0.5).long()
        train_total += batch_labels.size(0)
        train_correct += (predicted == batch_labels).sum().item()

    model.eval()
    val_loss = 0
    eval_correct, eval_total = 0,0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data).squeeze()
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            predicted = (outputs >= 0.5).long()
            eval_total += labels.size(0)
            eval_correct += (predicted == labels).sum().item()
    wandb.log({"epoch": epoch, "train_loss": training_loss/len(train_loader), "train_accuracy": train_correct/train_total, "validation_loss": val_loss/len(val_loader), "validation_accuracy": eval_correct/eval_total})

model.eval()
test_correct, test_total = 0,0
with torch.no_grad():
    for test_data, test_labels in test_loader:
        test_data, test_labels = test_data.to(device), test_labels.to(device)
        outputs = model(test_data).squeeze()
        predicted = (outputs >= 0.5).long()
        test_total += test_labels.size(0)
        test_correct += (predicted == test_labels).sum().item()
wandb.log({"test_accuracy": test_correct/test_total})
