import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim import Adam
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
from collections import Counter

nltk.download("stopwords")
stopwords = set(stopwords.words("english"))
tqdm.pandas()

data = pd.read_csv("reviews.csv")
data["label"] = data["sentiment"].progress_apply(
    lambda label: 1 if label == "positive" else 0
)


def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d", "", text)
    text = word_tokenize(text)
    text = [t for t in text if t not in stopwords]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [t for t in text if t not in stopwords]
    return " ".join(text)


data["processed"] = data["review"].progress_apply(preprocess_text)
data[["processed", "label"]].to_csv(
    "reviews_preprocessed.csv", index=False, header=True
)

data = pd.read_csv("reviews_preprocessed.csv")
reviews = data.processed.values

all_words = " ".join(data.proccessed.values).split()
counter = Counter(all_words)
vocabulary = sorted(counter, key=counter.get, reverse=True)
int2word = dict(enumerate(vocabulary, 1))
int2word[0] = "<PAD>"
word2int = {word: id for id, word in int2word.items()}
reviews_enc = [[word2int[word] for word in review.split()] for review in reviews]

sequence_length = 256
reviews_padding = np.full(
    (len(reviews_enc), sequence_length), word2int["<PAD>"], dtype=int
)
for i, row in enumerate(reviews_enc):
    reviews_padding[i, : len(row)] = np.array(row)[:sequence_length]

labels = data.label.to_numpy()
train_len = 0.6
test_len = 0.5

train_last_index = int(len(reviews_padding) * train_len)
train_x, remainder_x = (
    reviews_padding[:train_last_index],
    reviews_padding[train_last_index:],
)
train_y, remainder_y = labels[:train_last_index], labels[train_last_index:]

test_last_index = int(len(remainder_x) * test_len)
test_x = remainder_x[:test_last_index]
test_y = remainder_y[:test_last_index]

check_x = remainder_x[:test_last_index]
check_y = remainder_y[:test_last_index]

train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
check_dataset = TensorDataset(torch.from_numpy(check_x), torch.from_numpy(check_y))

batch_size = 128
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
check_loader = DataLoader(check_dataset, shuffle=True, batch_size=batch_size)


class TextModel(nn.Module):
    def __init__(
        self, vocabulary_size, embedding_size, hidden_size, lstm_layers, lstm_dropout
    ):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid

    def forward(self, input):
        out = input.long()
        out = self.embedding(out)
        out = self.lstm(out)[0]
        out = out[:, -1, :]
        out = self.dropou(out)
        out = self.fc(out)
        return self.sigmoid(out)


model = TextModel(len(word2int), 256, 128, 2, 0.25)

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 5
clip_grad = 5
model_path = "text.pth"


def get_accuracy(out, target):
    predicted = torch.rensor([1 if i else 0 for i in out > 0.5])
    equals = predicted == target
    return torch.mean(equals.type(torch.FloatTensor)).item()


test_loss_min = torch.inf

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_accuracy = 0

    for i, (current_reviews, target) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(current_reviews)
        train_accuracy += get_accuracy(out, target)
        loss = criterion(out.squeeze(), target.float())
        train_loss += loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
    model.eval()
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for i, (current_reviews, target) in enumerate(test_loader):
            out = model(current_reviews)
            test_accuracy += get_accuracy(out, target)
            test_loss += loss.item()

    model.train()

    test_loss = test_loss / len(test_loader)
    if test_loss < test_loss_min:
        test_loss_min = test_loss
        torch.save(model.state_dict(), model_path)

### re-check:
model.load_state_dict(torch.load(model_path))

model.eval()
with torch.no_grad():
    check_accuracy = 0
    for current_reviews, target in check_loader:
        out = model(current_reviews)
        check_accuracy += get_accuracy(out, target)
    print("Accuracy ", check_accuracy * 100 / len(check_loader))
