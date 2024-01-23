import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def create_data_repository():
    if not os.path.exists('data'):
        os.makedirs('data')

def split_and_save_data():
    iris = load_iris()
    X_train, X_inference, y_train, y_inference = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Create DataFrames
    train_df = pd.DataFrame(data=X_train, columns=iris.feature_names)
    train_df['label'] = y_train

    inference_df = pd.DataFrame(data=X_inference, columns=iris.feature_names)
    inference_df['label'] = y_inference

    # Save as CSV
    train_df.to_csv('data/train_data.csv', index=False)
    inference_df.to_csv('data/inference_data.csv', index=False)

def main():
    create_data_repository()
    split_and_save_data()

if __name__ == "__main__":
    main()
