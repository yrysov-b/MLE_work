import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd

# Define the neural network model
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)

def load_data():
    # Load training data
    train_data = pd.read_csv('data/train_data.csv')
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values

    # Load inference data
    inference_data = pd.read_csv('data/inference_data.csv')
    X_inference = inference_data.drop('label', axis=1).values
    y_inference = inference_data['label'].values

    return X_train, y_train, X_inference, y_inference

def train_model(X_train, y_train):
    model = IrisClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        inputs = torch.Tensor(X_train).float()
        labels = torch.LongTensor(y_train)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model

def evaluate_model(model, X, y):
    inputs = torch.Tensor(X).float()
    labels = torch.LongTensor(y)

    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    accuracy = accuracy_score(y, predicted.numpy())
    print(f'Accuracy: {accuracy * 100:.2f}%')

def save_model(model, directory='models', filename='iris_model.pth'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    torch.save(model.state_dict(), filepath)

def main():
    # Load data
    X_train, y_train, X_inference, y_inference = load_data()

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_inference, y_inference)

    # Save model in the "models" directory
    save_model(model)

if __name__ == "__main__":
    main()
