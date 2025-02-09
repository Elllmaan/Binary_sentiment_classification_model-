# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json
import zipfile
import requests

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Append the repository root (one level above src) to system path
sys.path.append(os.path.dirname(ROOT_DIR))
from src.utils import singleton, get_project_dir, configure_logging

# Initially create a default DATA_DIR relative to this file (will be overridden later)
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data/raw'))
if not os.path.exists(DEFAULT_DATA_DIR):
    os.makedirs(DEFAULT_DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH', os.path.join(ROOT_DIR, '../settings.json'))


# Load configuration settings from JSON
if CONF_FILE is None:
    logger.error("CONF_PATH environment variable is not set. Please set it to the path of your configuration file.")
    sys.exit(1)

logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths using the repository root instead of get_project_dir to fix the file-not-found error.
logger.info("Defining paths...")

# Compute the repository root (one level above the src folder)
project_dir = os.path.abspath(os.path.join(ROOT_DIR, '..'))
# Build the raw data directory path: <repo_root>/<data_dir from conf>/raw
DATA_DIR = os.path.join(project_dir, conf['general']['data_dir'], 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Dataset URLs from config
TRAIN_URL = conf['train']['data_url']
TEST_URL = conf['inference']['data_url']

# Function to download dataset from a URL
def download_file(url, save_path):
    logger.info(f"Downloading dataset from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    
    with open(save_path, 'wb') as file:
        file.write(response.content)
    logger.info(f"File saved to {save_path}")

# Singleton class for handling dataset
@singleton
class DatasetHandler():
    def __init__(self):
        self.train_df = None
        self.test_df = None

    # Method to download or load datasets
    def load(self, train_url: str, test_url: str, save_train_path: str, save_test_path: str):
        logger.info("Processing dataset files...")

        # Check if the train file exists; if not, download it.
        if not os.path.exists(save_train_path):
            download_file(train_url, save_train_path)
        else:
            logger.info(f"Train file already exists at {save_train_path}, skipping download.")

        # Check if the test file exists; if not, download it.
        if not os.path.exists(save_test_path):
            download_file(test_url, save_test_path)
        else:
            logger.info(f"Test file already exists at {save_test_path}, skipping download.")
        
        # Load data into pandas DataFrames
        self.train_df = pd.read_csv(save_train_path)
        self.test_df = pd.read_csv(save_test_path)
        
        logger.info(f"Training data loaded from {save_train_path}")
        logger.info(f"Test data loaded from {save_test_path}")
        return self.train_df, self.test_df

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    dataset_handler = DatasetHandler()
    dataset_handler.load(
        train_url=TRAIN_URL,
        test_url=TEST_URL,
        save_train_path=TRAIN_PATH,
        save_test_path=INFERENCE_PATH
    )
    logger.info("Loading script completed successfully.")