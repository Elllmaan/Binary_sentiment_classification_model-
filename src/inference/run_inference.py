import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Optional, Any

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, roc_auc_score,
                             ConfusionMatrixDisplay)

import nltk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Set up NLTK data directory and download required resources
nltk_data_dir = os.path.join(PROJECT_ROOT, 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

# Import common utilities from src/utils.py.

from src.utils import configure_logging, get_project_dir
from src.nlp_utils import process_basic, select_preprocessor



CONF_FILE = os.getenv('CONF_PATH', os.path.join(PROJECT_ROOT, 'settings.json'))
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define directories
DATA_DIR = os.path.join(PROJECT_ROOT, conf['general']['data_dir'], 'raw')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'outputs', conf['general']['models_dir'])
VECTORIZER_DIR = os.path.join(PROJECT_ROOT, 'outputs', conf['general']['vectorizers_dir'])
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, 'outputs', conf['general']['results_dir'])
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')

# Ensure the figures directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)

# Initialize parser for command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--infer_file",
    help="Specify inference data file",
    default=conf['inference']['inp_table_name']
)
parser.add_argument(
    "--out_path",
    help="Specify the path to the output table"
)

def get_latest_model_path() -> str:

    latest_model = None
    latest_datetime = None

    for dirpath, _, filenames in os.walk(MODEL_DIR):
        for filename in filenames:
            if filename.endswith('.pickle'):
                try:
                    # Remove the '.pickle' extension before parsing.
                    timestamp_str = filename[:-7]
                    model_datetime = datetime.strptime(timestamp_str, conf['general']['datetime_format'])
                except ValueError as e:
                    logging.warning(f"Filename '{filename}' not parsed with datetime format: {e}")
                    continue
                if latest_datetime is None or model_datetime > latest_datetime:
                    latest_datetime = model_datetime
                    latest_model = filename

    if latest_model is None:
        logging.error("No valid model file found in MODEL_DIR.")
        sys.exit(1)

    latest_path = os.path.join(MODEL_DIR, latest_model)
    logging.info(f"Latest model determined to be: {latest_path}")
    return latest_path

def get_model_by_path(path: str) -> Any:

    if not os.path.exists(path):
        logging.error(f"Model file does not exist at: {path}")
        sys.exit(1)
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            logging.info(f"Loaded model from: {path}")
            return model
    except Exception as e:
        logging.error(f"An error occurred while loading the model from {path}: {e}")
        sys.exit(1)

def get_vectorizer_by_path(path: str) -> Any:

    if not os.path.exists(path):
        logging.error(f"Vectorizer file does not exist at: {path}")
        sys.exit(1)
    try:
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)
            logging.info(f"Loaded vectorizer from: {path}")
            return vectorizer
    except Exception as e:
        logging.error(f"An error occurred while loading the vectorizer: {e}")
        sys.exit(1)

def get_inference_data(path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded inference data from: {path} (shape={df.shape})")
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)

def store_results(results: pd.DataFrame, path: Optional[str] = None) -> None:

    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)
    if not path:
        path = os.path.join(PREDICTIONS_DIR, 'predictions.csv')
    results.to_csv(path, index=False)
    logging.info(f"Results saved to {path}, shape={results.shape}")

def predict_results(model: Any, vectorizer: Any, infer_data: pd.DataFrame) -> pd.DataFrame:

    if 'review' not in infer_data.columns:
        logging.error("The input DataFrame for inference must contain a 'review' column.")
        sys.exit(1)
    
    # Select the same preprocessor used during training.
    preprocessor = select_preprocessor()
    infer_data['processed'] = infer_data['review'].apply(preprocessor)
    X_infer = vectorizer.transform(infer_data['processed'])
    numeric_predictions = model.predict(X_infer)
    label_mapping = {0: 'negative', 1: 'positive'}
    predictions = pd.Series(numeric_predictions).map(label_mapping)
    infer_data['predictions'] = predictions
    return infer_data[['review', 'predictions']]

def main() -> None:

    configure_logging()
    args = parser.parse_args()

    # Determine model and vectorizer paths.
    model_name = conf['inference'].get('model_name')
    if not model_name:
        logging.info("No model_name specified in settings.json; using the latest model.")
        model_path = get_latest_model_path()
        # The vectorizer is assumed to have the same filename in the vectorizer directory.
        vectorizer_path = os.path.join(VECTORIZER_DIR, os.path.basename(model_path))
    else:
        model_path = os.path.join(MODEL_DIR, model_name)
        vectorizer_path = os.path.join(VECTORIZER_DIR, model_name)

    # Load model and vectorizer.
    model = get_model_by_path(model_path)
    vectorizer = get_vectorizer_by_path(vectorizer_path)

    # Load inference data.
    infer_file = args.infer_file
    inference_path = os.path.join(DATA_DIR, infer_file)
    infer_data = get_inference_data(inference_path)

    # Predict results.
    results = predict_results(model, vectorizer, infer_data)

    # Determine the predictions file name based on the model file name if no custom path is provided.
    model_basename = os.path.basename(model_path)
    base_name, _ = os.path.splitext(model_basename)
    if not args.out_path:
        predictions_file = os.path.join(PREDICTIONS_DIR, f"predictions_{base_name}.csv")
    else:
        predictions_file = args.out_path

    # Save predictions.
    store_results(results, predictions_file)

    # Optionally, if test data is available, calculate additional metrics and save figures.
    test_data_path = os.path.join(DATA_DIR, 'test.csv')
    if os.path.exists(test_data_path):
        test_data = pd.read_csv(test_data_path)
        if 'sentiment' in test_data.columns:
            # Map true labels and predictions to numeric values.
            true_labels = test_data['sentiment'].map({'positive': 1, 'negative': 0})
            numeric_predictions = results['predictions'].map({'positive': 1, 'negative': 0})
            
            # Calculate metrics.
            accuracy = accuracy_score(true_labels, numeric_predictions)
            precision = precision_score(true_labels, numeric_predictions)
            recall = recall_score(true_labels, numeric_predictions)
            f1 = f1_score(true_labels, numeric_predictions)
            cm = confusion_matrix(true_labels, numeric_predictions)
            report = classification_report(true_labels, numeric_predictions)
            
            # Create a metrics information string.
            metrics_info = f"Test Accuracy: {accuracy * 100:.2f}%\n"
            metrics_info += f"Precision: {precision * 100:.2f}%\n"
            metrics_info += f"Recall: {recall * 100:.2f}%\n"
            metrics_info += f"F1 Score: {f1 * 100:.2f}%\n"
            
            # Write metrics to a text file.
            metrics_file = os.path.join(PREDICTIONS_DIR, f"metrics_{base_name}.txt")
            with open(metrics_file, 'w') as f:
                f.write(metrics_info)
            logging.info(f"Metrics (Precision, Recall, F1 Score, Confusion Matrix) written to {metrics_file}")
            
            # Print key metrics to console.
            print("Key Evaluation Metrics:")
            print(metrics_info)
            logging.info(f"Final Test Accuracy: {accuracy * 100:.2f}%")
            
            # For ROC curve, preprocess test data similarly to inference.
            preprocessor = select_preprocessor()
            test_data['processed'] = test_data['review'].apply(preprocessor)
            X_test = vectorizer.transform(test_data['processed'])
            
            # Try to obtain prediction scores (probabilities or decision function) for ROC calculation.
            if hasattr(model, 'predict_proba'):
                scores = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
            else:
                scores = None
                logging.warning("Model does not support predict_proba or decision_function. ROC curve cannot be plotted.")
            
            if scores is not None:
                fpr, tpr, _ = roc_curve(true_labels, scores)
                roc_auc = roc_auc_score(true_labels, scores)
                plt.figure()
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                roc_path = os.path.join(FIGURES_DIR, f"roc_auc_{base_name}.png")
                plt.savefig(roc_path)
                plt.close()
                logging.info(f"ROC AUC curve saved to {roc_path}")
            
            # Plot and save confusion matrix figure.
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            cm_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{base_name}.png")
            plt.savefig(cm_path)
            plt.close()
            logging.info(f"Confusion matrix figure saved to {cm_path}")
        else:
            logging.info("No 'sentiment' column found in test data; skipping additional metrics calculation.")
    else:
        logging.info("Test data not found; skipping additional metrics calculation.")

if __name__ == "__main__":
    main()