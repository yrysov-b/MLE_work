import torch
import pandas as pd
import pickle

from training.train import IrisClassifier, inference

def load_inference_data():
    # Load inference data
    inference_data = pd.read_csv('data/inference_data.csv')
    X_inference = inference_data.drop('label', axis=1).values
    y_inference = inference_data['label'].values
    return X_inference, y_inference

def load_model(filepath='models/model.pickle'):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Load inference data
    X_inference, y_inference = load_inference_data()

    # Load trained model
    model = load_model()

    # Make predictions
    predictions = inference(model, X_inference)

    # Evaluate predictions
    accuracy = (predictions == y_inference).mean() * 100
    print(f'Accuracy on inference data: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
