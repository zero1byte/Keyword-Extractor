#  Keyword Extractor

A TF-IDF based keyword extraction pipeline trained on the [LDKP10K](https://huggingface.co/datasets/midas/ldkp10k) scientific keyphrase dataset from HuggingFace.

---

##  Features

- End-to-end NLP pipeline: data loading → preprocessing → training → evaluation
- TF-IDF vectorizer with **unigram, bigram, and trigram** support
- Consistent preprocessing applied to both documents and ground-truth keyphrases
- Evaluation with **Precision, Recall, and F1 Score** on a held-out test split
- Interactive **Jupyter widget UI** — paste any text and extract keywords instantly
- Clean, modular code that can be used as a Python library or run standalone

---

##  Project Structure

```
keyword-extractor/
├── keyword_extractor.py   # Core pipeline: loading, preprocessing, model, evaluation
├── app.py                 # Jupyter widget UI (run inside a notebook)
├── requirements.txt       # Python dependencies
└── README.md
```

---

##  Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/keyword-extractor.git
cd keyword-extractor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline (terminal)

```bash
python keyword_extractor.py
```

This will:
- Fetch the LDKP10K dataset from HuggingFace
- Preprocess text and keyphrases
- Fit the TF-IDF model on the training split
- Evaluate on 100 test samples and display a performance chart

### 4. Run the interactive UI (Jupyter)

Open a Jupyter notebook and run:

```python
%run app.py
```

Or import directly:

```python
from app import launch_app
launch_app()
```

---

## 🧠 How It Works

### Dataset

The pipeline uses the **LDKP10K Small** split from HuggingFace, a large-scale dataset of scientific papers with human-annotated abstractive keyphrases.

### Preprocessing

Both document text and keyphrase tags pass through the same pipeline to ensure fair comparison:

1. Lowercase the text
2. Remove all non-alphabetic characters
3. Filter out tokens shorter than 4 characters

### Feature Extraction

A `TfidfVectorizer` is fitted only on the **training split** (no data leakage) with:

| Parameter | Value |
|---|---|
| `max_features` | 30,000 |
| `ngram_range` | (1, 3) — unigrams to trigrams |
| `stop_words` | English |

### Keyword Extraction

For a given document, the model:
1. Transforms it into a TF-IDF vector
2. Ranks all n-gram features by their TF-IDF score
3. Takes the top-N n-grams and expands them to individual word tokens
4. Returns the deduplicated set of keywords

### Evaluation

Each test document is scored against its ground-truth keyphrases:

```
Precision  = |predicted ∩ true| / |predicted|
Recall     = |predicted ∩ true| / |true|
F1         = 2 × (Precision × Recall) / (Precision + Recall)
```

Averages are computed across all valid test samples.

---

##  Usage as a Module

```python
from keyword_extractor import KeywordExtractor, load_dataset, prepare_data
from sklearn.model_selection import train_test_split

# Load and prepare data
df = load_dataset()
df = prepare_data(df)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Fit model
extractor = KeywordExtractor(max_features=30_000, ngram_range=(1, 3))
extractor.fit(train_df["clean_text"])

# Extract keywords from any text
keywords = extractor.extract("Deep learning has revolutionized natural language processing.", top_n=15)
print(keywords)

# Evaluate on test set
precision, recall, f1 = extractor.evaluate(test_df, sample_size=100)
```

---

##  Limitations

TF-IDF is a strong baseline but has inherent constraints:

- **No semantic understanding** — treats words as independent frequency signals
- **Context-blind** — cannot resolve polysemy (e.g., "bank" as finance vs. geography)
- **No synonymy** — "neural network" and "deep learning" are unrelated to the model
- **Domain-dependent** — performance drops on text far from the training distribution

For improved extraction, consider transformer-based models such as **KeyBERT**, **YAKE**, or **PKE**.

---

##  Dependencies

| Library | Purpose |
|---|---|
| `nltk` | Stopword corpus |
| `requests` | HuggingFace dataset fetch |
| `pandas` | DataFrame handling |
| `scikit-learn` | TF-IDF vectorizer, train/test split |
| `matplotlib` | Evaluation chart |
| `ipywidgets` | Jupyter interactive UI |

---


