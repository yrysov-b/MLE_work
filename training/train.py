import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import logging
import os
import time

RESULTS_DIR = 'results'
LOG_FILE = os.path.join(RESULTS_DIR, 'training_log.txt')

# Create the "results" folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'w', 'utf-8')])

class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)

def load_data():
    logging.info("Loading training data...")
    train_data = pd.read_csv('data/train_data.csv')
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    logging.info(f"Training dataset size: {len(train_data)}")
    return X_train, y_train

def train_model(X_train, y_train):
    logging.info("Training model...")
    model = IrisClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(1000):
        inputs = torch.Tensor(X_train).float()
        labels = torch.LongTensor(y_train)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss.item()}")
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    end_time = time.time()
    training_time = end_time - start_time
    logging.info(f"Training time: {training_time} seconds")

    return model

def save_model(model, filepath='models/model.pickle'):
    if not os.path.exists('models'):
        os.makedirs('models')

    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

def run_training():
    try:
        X_train, y_train = load_data()
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Empty training dataset.")

        model = train_model(X_train, y_train)

        # Save the model
        save_model(model)

        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    run_training()
