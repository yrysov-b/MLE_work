import torch
import torch.nn as nn
import pandas as pd
import pickle
import logging
import os
import time
from sklearn.metrics import accuracy_score

RESULTS_DIR = 'results'
LOG_FILE = os.path.join(RESULTS_DIR, 'inference_log.txt')

# Create the "results" folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE, 'w', 'utf-8')])

class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)

def load_inference_data():
    logging.info("Loading inference data...")
    inference_data = pd.read_csv('data/inference_data.csv')
    X_inference = inference_data.drop('label', axis=1).values
    y_inference = inference_data['label'].values
    logging.info(f"Inference dataset size: {len(inference_data)}")
    return X_inference, y_inference

def load_model(filepath='models/model.pickle'):
    logging.info("Loading model...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' not found.")
    
    with open(filepath, 'rb') as file:
        logging.info("Model loaded.")
        model = pickle.load(file)
    return model

def inference(model, X):
    logging.info("Making predictions...")
    inputs = torch.Tensor(X).float()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    logging.info("Predictions made.")
    return predicted.numpy()

def run_inference():
    try:
        X_inference, y_inference = load_inference_data()
        if len(X_inference) == 0 or len(y_inference) == 0:
            raise ValueError("Empty inference dataset.")

        model = load_model()

        start_time = time.time()
        predictions = inference(model, X_inference)
        end_time = time.time()

        # Evaluate predictions
        accuracy = (predictions == y_inference).mean() * 100
        logging.info(f'Accuracy on inference data: {accuracy:.2f}%')

        inference_time = end_time - start_time
        logging.info(f"Inference time: {inference_time} seconds")
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    run_inference()
