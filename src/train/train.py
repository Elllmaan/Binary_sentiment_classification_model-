import argparse
import os
import sys
import pickle
import json
import logging
import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.utils import gen_batches

import nltk
from dotenv import load_dotenv
import mlflow

# Calculate the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up NLTK data directory
nltk_data_dir = os.path.join(PROJECT_ROOT, 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)


def download_nltk_package(package):
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_dir)

for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    download_nltk_package(pkg)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)


load_dotenv()


mlflow.autolog(log_datasets=False)

# Add project root to system path
sys.path.append(PROJECT_ROOT)


from src.utils import configure_logging
from src.nlp_utils import select_preprocessor, select_vectorizer


# Load configuration settings from settings.json
CONF_FILE = os.getenv('CONF_PATH', os.path.join(PROJECT_ROOT, 'settings.json'))
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define directories based on configuration
DATA_DIR = os.path.join(PROJECT_ROOT, conf['general']['data_dir'])
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'outputs', conf['general']['models_dir'])
VECTORIZER_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'vectorizers')


os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VECTORIZER_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(RAW_DATA_DIR, conf['train']['table_name'])
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv')

# Initialize parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", help="Specify training data file", default=conf['train']['table_name'])
parser.add_argument("--model_path", help="Specify the path for the output model")


class DataProcessor:

    def __init__(self) -> None:
        # Use config-driven preprocessor and vectorizer
        method = conf['train']['preprocessing_method']     # e.g. "basic"
        vectorizer_type = conf['train']['vectorizer']       # e.g. "TfidfVectorizer"
        
        self.preprocessor = select_preprocessor(method=method)
        self.vectorizer = select_vectorizer(vectorizer_type=vectorizer_type)
        
        # Log the chosen approach
        logging.info(f"Selected text processing method: {method}")
        logging.info(f"Selected vectorizer: {vectorizer_type}")

    def prepare_data(self, max_rows: int = None):

        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)

        # Remove duplicate rows by 'review' only
        initial_count = df.shape[0]
        df.drop_duplicates(subset='review', inplace=True)
        logging.info(f"Removed {initial_count - df.shape[0]} duplicates by 'review'. New shape: {df.shape}")
        max_rows = df.shape
        # Basic validation for required columns
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            logging.error("Input CSV must contain 'review' and 'sentiment' columns.")
            sys.exit(1)

        # Convert sentiment to numeric values: 1 for positive, 0 for negative
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

        # Apply data sampling (reduce dataset size) if needed
        dataset_size = conf['train']['dataset_size']  # Get dataset size from settings.json
        if dataset_size and dataset_size > 0 and len(df) > dataset_size:
            df = df.sample(n=dataset_size, random_state=conf['general']['random_state'])
            logging.info(f"Random sampling performed. Sample size: {max_rows}")

        # Preprocess text using the chosen preprocessor
        df['review'] = df['review'].apply(self.preprocessor)

        # Save processed data
        df.to_csv(PROCESSED_TRAIN_PATH, index=False)
        logging.info(f"Processed data saved to {PROCESSED_TRAIN_PATH}. Shape: {max_rows}")

        # Vectorize text data
        X = self.vectorizer.fit_transform(df['review'])
        y = df['sentiment']
        return X, y
    
    def data_extraction(self, path: str) -> pd.DataFrame:

        logging.info(f"Loading data from {path}...")
        try:
            df = pd.read_csv(path)
            logging.info(f"Data loaded. Shape: {df.shape}")
            return df
        except FileNotFoundError as e:
            logging.error(f"Could not find file: {path}. Error: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error reading file {path}. Exception: {e}")
            sys.exit(1)

    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:

        if max_rows and max_rows > 0 and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=conf['general']['random_state'])
            logging.info(f"Random sampling performed. Sample size: {max_rows}")
        return df


class Training:

    def __init__(self) -> None:
        model_type = conf['train']['model']  # e.g. "Linear SVC"
        logging.info(f"Initializing Training with model type: {model_type}")

        if model_type == "Logistic Regression":
            self.model = LogisticRegression(max_iter=1000, random_state=conf['general']['random_state'])
        elif model_type == "Multinomial Naive Bayes":
            self.model = MultinomialNB()
        elif model_type == "Linear SVC":
            self.model = LinearSVC(random_state=conf['general']['random_state'])
        else:
            raise ValueError(f"Model '{model_type}' not recognized.")

        logging.info(f"Model details: {self.model}")

    def run_training(self, X, y, vectorizer, out_path: str = None, test_size: float = None):
        logging.info("Running training...")

        # Convert X to dense format only if it's sparse
        X_dense = X.toarray() if hasattr(X, "toarray") else X

        if isinstance(y, pd.Series):
            y_array = y.to_numpy(dtype=np.int32)  # Convert Pandas Series to NumPy array
        else:
            y_array = np.array(y, dtype=np.int32)  # Convert if not already an array

        if len(y_array.shape) > 1:
            y_array = y_array.ravel()

        # ✅ Now log dataset information without issues
        mlflow.log_param("dataset_size", X_dense.shape[0])
        mlflow.log_param("feature_count", X_dense.shape[1])
        mlflow.log_param("label_distribution", str(np.unique(y_array, return_counts=True)))

        if test_size is not None and 0 < test_size < 1:
            mlflow.log_param("test_size", test_size)
            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=conf['general']['random_state']
            )
            start_time = time.time()
            self.model.fit(X_train, y_train)  
            end_time = time.time()
            logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")
            self.test(X_test, y_test)
        else:
            mlflow.log_param("test_size", "Not used")
            logging.info("Training on the full dataset (test_size not provided).")
            start_time = time.time()
            self.model.fit(X, y)  
            end_time = time.time()
            logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")
            logging.info("Skipping evaluation since test_size is not valid.")

        self.save(out_path, vectorizer)


    def test(self, X_test, y_test) -> float:
 
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        mlflow.log_metric("test_accuracy", accuracy)
        return accuracy

    def save(self, path: str, vectorizer) -> None:

        model_filename = datetime.now().strftime(conf['general']['datetime_format']) + '.pickle'
        model_path = os.path.join(MODEL_DIR, model_filename)
        vectorizer_path = os.path.join(VECTORIZER_DIR, model_filename)

        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logging.info(f"Model saved to {model_path}")

        # Save the vectorizer
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logging.info(f"Vectorizer saved to {vectorizer_path}")


def main():
    configure_logging()

    # Parse command-line arguments (if any)
    args = parser.parse_args()

    data_proc = DataProcessor()
    trainer = Training()

    # Prepare data (optionally sampling a subset if specified in settings)
    X, y = data_proc.prepare_data(max_rows=conf['train']['data_sample'])


    # Run training. This will only split if conf['train']['test_size'] is in (0,1)
    trainer.run_training(
        X, 
        y, 
        vectorizer=data_proc.vectorizer, 
        test_size=conf['train']['test_size']
    )

if __name__ == "__main__":
    main()