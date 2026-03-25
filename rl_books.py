import csv
import random
import os 
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

dir_path = os.path.dirname(os.path.realpath(__file__))

filename = dir_path + "/books.csv"
fields = []
rows = []

with open(filename, 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

    #print("Total no. of rows: %d" % csvreader.line_num)  # Row count
#print('Field names are: ' + ', '.join(fields))
random.shuffle(rows)

# clean up data, put fields were going to use as features into
# an array and the average rating into an array to use as labels

# ratings index = 3
# integer features indices = 7, 8, 9

features = []
labels = []
for row in rows:
    try:
        feature = [float(row[7]), float(row[8]), float(row[9])]
        label = float(row[3])
        features.append(feature)
        labels.append(label)
    except:
        continue

features = np.array(features, dtype=np.float32)
features = np.array(features, dtype=np.float32)
features = (features - features.mean(axis=0)) / features.std(axis=0)

class GoodreadsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        # to test im just going to go off of the numerical fields for
        # now, going to change it to the other ones later
        feature = self.features[index]
        label = self.labels[index]

        return (torch.tensor(feature, dtype=torch.float32), torch.tensor([label], dtype=torch.float32))

# should use the first 80% of the list for training
# and final 20% for testing
dividing_index = int(len(features) * .8)
train_features = features[:dividing_index]
train_labels = labels[:dividing_index]
test_features = features[dividing_index:]
test_labels = labels[dividing_index:]

train_dataset = GoodreadsDataset(train_features, train_labels)
test_dataset = GoodreadsDataset(test_features, test_labels)

train_dataloader = DataLoader(train_dataset, 32, True)
test_dataloader = DataLoader(test_dataset, 32, False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

device = "cpu"
model = NeuralNetwork().to(device=device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    test_loss = 0
    num_samples = 0
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            batch_size = X.size(0)
            pred = model(X)
            test_loss += loss_fn(pred,y).item() * batch_size
            num_samples += batch_size
    print(test_loss/num_samples)

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")