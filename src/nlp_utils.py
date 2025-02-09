# src/nlp_utils.py
import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Calculate the project root (you could also import get_project_dir from utils if preferred)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up NLTK data directory
nltk_data_dir = os.path.join(PROJECT_ROOT, 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources.
# (If these downloads are time‐consuming, you might choose to perform them once during the container build or in an initialization step.)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

# Preprocessing: set up stopwords
stop_words = set(stopwords.words("english"))
stop_words.update(["film", "movie"])

def preprocess_text_basic(text: str) -> list:
    """
    Remove HTML tags, lowercase, remove punctuation, tokenize, and filter stopwords.
    
    :param text: The raw text.
    :return: A list of tokens.
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    return [w for w in tokens if w not in stop_words]

def process_basic(text: str) -> str:
    """
    Join tokens produced by preprocess_text_basic into a single string.
    
    :param text: The raw text.
    :return: A preprocessed string.
    """
    return " ".join(preprocess_text_basic(text))

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Map NLTK's POS tags to WordNet's format.
    
    :param treebank_tag: POS tag from nltk.pos_tag.
    :return: Corresponding WordNet POS tag.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def select_preprocessor(method: str = "basic"):
    """
    Select a text preprocessing lambda based on the method.
    
    :param method: The preprocessing method to use.
    :return: A lambda function that processes a text string.
    """
    if method == "stem":
        ps = PorterStemmer()
        return lambda text: " ".join([ps.stem(token) for token in preprocess_text_basic(text)])
    elif method == "lemma":
        lemmatizer = WordNetLemmatizer()
        return lambda text: " ".join([
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in nltk.pos_tag(preprocess_text_basic(text))
        ])
    elif method == "preserve_exclamations":
        return lambda text: " ".join(
            w for w in word_tokenize(
                text.lower().translate(
                    str.maketrans("", "", "".join([c for c in string.punctuation if c not in {"!", "?"}]))
                )
            ) if w not in stop_words
        )
    else:  # default to basic
        return lambda text: " ".join(preprocess_text_basic(text))






def select_vectorizer(vectorizer_type="TfidfVectorizer"):
    if vectorizer_type == "TfidfVectorizer":
        return TfidfVectorizer(
            max_features=50000,  # Reduce feature size
            dtype=np.float32,  # Use lower memory float format
            sublinear_tf=True
        )
    else:
        raise ValueError(f"Vectorizer '{vectorizer_type}' not recognized.")