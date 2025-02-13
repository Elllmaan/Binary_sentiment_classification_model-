# Final Project

This guide provides clear, step-by-step instructions to set up, train, and run inference for the project. Whether you’re working locally or using Docker, follow the instructions below to get started quickly.

This repository demonstrates a complete Sentiment Analysis pipeline for movie reviews, covering:

- Data Science (DS) Part:
    - Exploratory Data Analysis (EDA)
    - Feature Engineering (tokenization, stopword filtering, stemming vs. lemmatization, vectorization approaches)
    - Modeling (baseline, multiple models, best model selection)
    - Business Value Discussion
- Machine Learning Engineering (MLE) Part:
    - Dockerized training and inference containers
    - Automated data downloading and processing
    - Model packaging and reproducible predictions

## Table of Contents

1. Repository Overview
2. Data Science Summary
    2.1 Exploratory Data Analysis (EDA)
    2.2 Feature Engineering
    2.3 Modeling
    2.4 Potential Business Value
3. Quickstart Guide
    3.1 Cloning the Repository
    3.2 Generating Data (Local Runs Only)
    3.3 Local Setup
    3.3.1 Creating and Activating a Virtual Environment
    3.3.2 Installing Dependencies
    3.3.3 Configuring Environment Variables
    3.4 Running the Project Locally
    3.4.1 Training the Model Locally
    3.4.2 Running Inference Locally
4. Docker Guide
    4.1 Dockerized Training
    4.2 Dockerized Inference
5. Configuration with settings.json
6. Final Test Metrics
7. Project Structure


---
## 1. Repository Overview
This repository is dedicated to analyzing movie reviews and classifying their sentiment (positive/negative). It contains:

- Data Files (in data/):
    - raw/ - unprocessed train/test data
    - processed/ - cleaned and preprocessed data
- Notebooks (in notebooks/) with in-depth EDA and experimentation
- Source Code (in src/):
    - train/ - Dockerfile and scripts for training
    - inference/ - Dockerfile and scripts for inference
    - data_loader.py - Utility for downloading and preparing data
    - nlp_utils.py, utils.py - Helper functions
- Outputs (in outputs/):
    - models/ - Trained model artifacts
    - vectorizers/ - Vectorizer artifacts
    - predictions/ - Inference predictions and metrics
    - figures/ - Plots (e.g., confusion matrices, ROC curves)
- requirements.txt - Required Python packages
- settings.json - Configuration file for training/inference parameters
- .env - Optional environment file for referencing settings

## 2. Data Science Summary
2.1 Exploratory Data Analysis (EDA)
- Dataset Size:
    - Training: 40,000 movie reviews (20k positive / 20k negative)
    - Test: 10,000 reviews (5k positive / 5k negative)
- Missing Values & Duplicates:
    - No missing values in either set.
    - 272 duplicates in training; 13 in test. These were removed.
- Sentiment Distribution:
    - Balanced (50% positive, 50% negative), so accuracy is suitable.
- Review Length:
    - Right-skewed; most are medium length, but some reviews are extremely long (up to ~13,704 characters).
2.2 Feature Engineering
- Tokenization & Stopwords:
    - Used NLTK’s word tokenizer plus standard English stopwords.
    - Added domain-specific stopwords like “film,” “movie.”
- Stemming vs. Lemmatization:
    - Stemming (Porter) can truncate words (less context).
    - Lemmatization (WordNet) retains more linguistically correct forms.
- Vectorization:
    - CountVectorizer vs. TfidfVectorizer
    - TF-IDF outperformed Count in experiments.
2.3 Modeling
- Baseline: Logistic Regression + CountVectorizer + basic preprocessing → ~88.84% accuracy
- Experimental Grid:
    - Classifiers: Logistic Regression, Multinomial Naive Bayes, Linear SVC
    - Vectorizers: CountVectorizer, TfidfVectorizer
    - Processing: basic, stem, lemma, preserve_exclamations
- Top Performer: Linear SVC + TfidfVectorizer + basic preprocessing → ~90% accuracy
2.4 Potential Business Value
1. Customer Insights & Recommendation Systems

- Identifies sentiment to tailor recommendations, boosting user engagement.
2. Brand Reputation & Crisis Management

- Tracks negative sentiment trends for early mitigation.
3. Competitive Intelligence

- Monitors sentiment across platforms for data-driven market positioning.

## 3. Quickstart Guide
### Cloning the Repository

Clone the repository to your local machine and navigate into the project directory:

```bash
git clone <repository-url>
cd <repository-name>
```

---

### Generating Data (Local Runs Only)

Before training or running inference locally, you must generate the necessary data by running:

bash
python src/data_loader.py


**Note:**  
If you are using Docker, the data is generated automatically as part of the container workflow. You do not need to run this step manually.

---

###  Local Setup

#### Creating and Activating a Virtual Environment

1. **Create a Virtual Environment:**

   
bash
   python -m venv myenv


2. **Activate the Virtual Environment:**

   - **On macOS/Linux:**
     
bash
     source myenv/bin/activate

   - **On Windows:**
     
bash
     myenv\Scripts\activate


#### Installing Dependencies

Install all required Python packages using:

bash
pip install -r requirements.txt


#### Configuring Environment Variables

Create a .env file to set the path to your settings file:

bash
echo "CONF_PATH=settings.json" > .env


---

### Running the Project Locally

#### Training the Model

Train the model by running the training script:

bash
python src/train/train.py


During training, you will see informative log messages (e.g., data preparation, training progress, model saving) that help you track the process.

#### Running Inference

After training, run the inference script to evaluate your model and generate predictions:

bash
python src/inference/run_inference.py


The inference process will load your trained model, perform predictions, and save the results.

---

### 4. Docker Setup

Docker workflows automatically generate data, so you do not need to run the data generation step separately.

#### Training with Docker

1. **Create folders that are used as Volumes**

   
mkdir -p data/raw data/processed outputs/models outputs/vectorizers outputs/predictions outputs/figures


2. **Run the Training Container:**
  
   1. **Build the Training Docker Image:**

   
bash
   docker build -f src/train/Dockerfile -t training_image .


   
bash
   docker run -v "$(pwd)/outputs":/app/outputs -v "$(pwd)/data":/app/data training_image
   
   Now you should be able to see model and vectorizer in output folder

### Inference with Docker

1. **Build the Inference Docker Image:**

   
bash
   docker build -f src/inference/Dockerfile -t inference_image .


2. **Run the Inference Container:**

   
bash
   docker run -v "$(pwd)/outputs":/app/outputs -v "$(pwd)/data":/app/data inference_image

Key Evaluation Metrics:
Test Accuracy: 89.35%
Precision: 89.20%
Recall: 89.54%
F1 Score: 89.37%

## 5. Configuration with settings.json

The file settings.json defines important parameters for both training and inference stages. 

Here you can describe possible options for parameters, e.g.
```
"model": "Linear SVC", "Logistic Regression", ...
```

Below is an example:
```
{
    "general": {
        "random_state": 42,
        "status": "test",
        "datetime_format": "%d.%m.%Y_%H.%M",
        "data_dir": "data",
        "models_dir": "models",
        "vectorizers_dir": "vectorizers",
        "results_dir": "predictions"
    },
    "train": {
        "table_name": "train.csv",
        "data_sample": null,
        "dataset_size": 32000, 
        "test_size": 0.2,
        "preprocessing_method": "basic",
        "vectorizer": "TfidfVectorizer",
        "model": "Linear SVC",
        "data_url": "https://media.githubusercontent.com/media/Elllmaan/Epam/refs/heads/main/train.csv"
    },
    "inference": {
        "inp_table_name": "test.csv",
        "model_name": null,
        "vectorizer_name": null,
        "data_url": "https://media.githubusercontent.com/media/Elllmaan/Epam/refs/heads/main/test.csv"
    }
}
```

You can override any of these settings to experiment with different preprocessing methods, vectorizers, or models. The code in src/train/train.py and src/inference/run_inference.py references these settings to configure the entire pipeline.

## 6. Final Test Metrics

When running inference via Docker with the best model (Linear SVC + TfidfVectorizer, basic cleaning), you can expect results such as:

    Test Accuracy: 89.35%
    Precision: 89.20%
    Recall: 89.54%
    F1 Score: 89.37%

You can find the corresponding metrics in outputs/predictions/metrics_<timestamp>.txt, along with ROC AUC and confusion matrix figures in outputs/figures/.

## 7. Project Structure

A high-level overview of the main folders and files:
```
.
├── README.md              
├── notebooks
│   └── FinalProject.ipynb
├── data
│   ├── raw
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed
│       └── train_processed.csv
├── outputs
│   ├── models
│   ├── vectorizers
│   ├── predictions
│   └── figures
├── src
│   ├── __init__.py
│   ├── data_loader.py
│   ├── nlp_utils.py
│   ├── utils.py
│   ├── train
│   │   ├── Dockerfile
│   │   └── train.py
│   └── inference
│       ├── Dockerfile
│       └── run_inference.py
├── requirements.txt
├── settings.json
└── .env (optional)
```

## Review

Good job, Elman! All tasks are done, good exploration of the data, clear readme. Some suggestments could be found in merge request. Keep going! Good luck!

#### Evaluation criteria

Following things need to be checked (100 in total):

1. Repository exists, doesn’t contain unnecessary files or trained models, and can be cloned using the “git clone” command
   Important! your repository should be private, closer to checking we will inform you which usernames need to be added as collaborators); (5%) 5/5

2. DS part (max 40%):

   Artifacts: DS part report in README.md, notebooks;
   Required steps with comments and conclusions:

   all this should be reflected in DS part report in README.md

   a) EDA (5%) 5/5

   b) Feature engineering (10%) 10/10
   - tokenization;
   - stop-words filtering;
   - compare stemming vs lemmatization;
   - vectorization: compare at least 2 approaches;

   c) Modeling (15%): 15/15
   - Build baseline ML model, explore different models: at least 3 models;
   - Reasoning on best model selection;
   - Overall performance evaluation (best accuracy>=0.85);

   d) Potential business applications and value for business; (10%) 10/10

3. MLE part (max 40%):

   Docker image can be built using dockerfile:
   *[Outputs] All the necessary outputs (serialized models, predictions, plots, train results etc) should be accessible after container execution i.e., provisioning volume to container is required;
   a) [Training] from the training directory. It should have all the necessary data and code for the training. The running container should automatically start the training process, save the trained model (in the container), and show the quality calculated on the test sample;
   Important! final metrics for the test should be added to the README.md, and it should be reproducible and visible; (20%) 20/20
   b) [Inference] from the inference directory. It should have all the necessary data, a trained model from the previous step and the code for inference. The running container should automatically run the trained model on the inference data and save the result in the additional file; (20%) 20/20

4. README.md file is well-structured, includes DS part report and ML part (how to run guide, quickstart instruction); (10%) 9/10

5. Well-structured ML project and clean code. (5%) 4/5

Total 98/100