import csv
import argparse
import random
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

dir_path = os.path.dirname(os.path.realpath(__file__))

filename = dir_path + "/books.csv"
fields = []
rows = []

with open(filename, 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

random.shuffle(rows)

# clean up data, put fields were going to use as features into
# an array and the average rating into an array to use as labels

# ratings index = 3
# integer features indices = 7, 8, 9
# title 1
# author 2
# num_pages 7
# language_code 6

# numeric features
features = []

# labels
average_ratings = []

# text features
titles = []
authors = []

for row in rows:
    try:
        # get the numeric features and put them into an array
        numeric_features = [len(row[1]), len(row[1].split()), float(row[7]), int.from_bytes(row[6].encode('utf-8'), 'big')]
        features.append(numeric_features)

        # string features we have to deal with later
        titles.append(row[1])
        authors.append(row[2]) 

        # labels
        average_ratings.append(float(row[3]))

    except:
        continue

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

title_vectorizer = TfidfVectorizer(max_features=7000) 
author_vectorizer = CountVectorizer(max_features=500)

dividing_index = int(len(features) * .8)
train_features = features[:dividing_index]
train_average_ratings = average_ratings[:dividing_index]
train_titles = titles[:dividing_index]
train_authors = authors[:dividing_index]
train_titles_text = title_vectorizer.fit_transform(train_titles).toarray()
train_authors_text = author_vectorizer.fit_transform(train_authors).toarray()

test_features = features[dividing_index:]
test_average_ratings = average_ratings[dividing_index:]
test_titles = titles[dividing_index:]
test_authors = authors[dividing_index:]
test_titles_text = title_vectorizer.transform(test_titles).toarray()
test_authors_text = author_vectorizer.transform(test_authors).toarray()

train_features = np.concatenate([train_features, train_titles_text, train_authors_text], axis=1)
test_features = np.concatenate([test_features, test_titles_text, test_authors_text], axis=1)

avg_train_dataset = GoodreadsDataset(train_features, train_average_ratings)
avg_test_dataset = GoodreadsDataset(test_features,  test_average_ratings)
avg_train_dataloader = DataLoader(avg_train_dataset, 32, True)
avg_test_dataloader = DataLoader(avg_test_dataset, 32, False)

class AllTogether(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4 + 7000 + 500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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
avg_model = AllTogether().to(device=device)

loss_fn = nn.MSELoss()
avg_optimizer = torch.optim.SGD(avg_model.parameters(), lr=1e-3)


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
    train(avg_train_dataloader, avg_model, loss_fn, avg_optimizer)
    test(avg_test_dataloader, avg_model, loss_fn)
print("Done!")