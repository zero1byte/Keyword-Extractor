"""
keyword_extractor.py
--------------------
TF-IDF based keyword extraction pipeline trained on the LDKP10K dataset.

Usage (standalone):
    python keyword_extractor.py

Usage (as a module):
    from keyword_extractor import KeywordExtractor, load_dataset, prepare_data
"""

import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_URL = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=midas%2Fldkp10k&config=small&split=train"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(url: str = DATASET_URL, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch the LDKP10K dataset from HuggingFace and return a DataFrame
    with 'Summary' (full section text) and 'Tags' (keyphrase list) columns.

    Raises:
        RuntimeError: if the HTTP request fails.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch dataset: {exc}") from exc

    rows = [item["row"] for item in data["rows"]]
    df = pd.DataFrame(rows)

    df["Summary"] = df["sec_text"].apply(
        lambda sections: " ".join(" ".join(sec) for sec in sections)
    )
    df["Tags"] = df["abstractive_keyphrases"]
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    """
    Normalize text for TF-IDF:
      - Lowercase
      - Remove non-alphabetic characters
      - Drop tokens shorter than 4 characters
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    tokens = [w for w in text.split() if len(w) > 3]
    return " ".join(tokens)


def process_tags(tags: list) -> list:
    """
    Apply the same preprocessing pipeline to a list of keyphrase tags,
    then flatten and deduplicate to individual word tokens.
    """
    words = []
    for tag in tags:
        cleaned = preprocess(tag)
        words.extend(cleaned.split())
    return list(set(words))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'clean_text' and 'true_keywords' columns to the DataFrame.
    Returns a new DataFrame (does not mutate the original).
    """
    df = df.copy()
    df["clean_text"] = df["Summary"].apply(preprocess)
    df["true_keywords"] = df["Tags"].apply(process_tags)
    return df


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class KeywordExtractor:
    """
    TF-IDF keyword extractor supporting unigrams, bigrams, and trigrams.

    Example:
        extractor = KeywordExtractor()
        extractor.fit(train_df["clean_text"])
        keywords = extractor.extract("Deep learning has transformed NLP.", top_n=10)
    """

    def __init__(self, max_features: int = 30_000, ngram_range: tuple = (1, 3)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self._fitted = False

    def fit(self, texts) -> None:
        """Fit the TF-IDF vocabulary on an iterable of preprocessed texts."""
        self.vectorizer.fit(texts)
        self._fitted = True

    def extract(self, text: str, top_n: int = 30) -> list:
        """
        Return up to top_n keywords for the given raw text.

        N-gram phrases are expanded to their constituent words so that
        single-word results can be compared against ground-truth tokens.

        Raises:
            RuntimeError: if called before fit().
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before extract().")

        cleaned = preprocess(text)
        tfidf_vector = self.vectorizer.transform([cleaned])

        feature_names = self.vectorizer.get_feature_names_out()
        scores = zip(feature_names, tfidf_vector.toarray()[0])
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)

        words = set()
        for phrase, score in ranked[:top_n]:
            if score == 0.0:
                break
            words.update(phrase.split())

        return list(words)

    def evaluate(
        self,
        test_df: pd.DataFrame,
        sample_size: int = 100,
        top_n: int = 30,
    ) -> tuple:
        """
        Compute average Precision, Recall, and F1 over test_df rows.

        Skips rows where either true_keywords or predicted keywords are empty.

        Returns:
            (avg_precision, avg_recall, avg_f1) — all floats in [0, 1].
            Returns (0.0, 0.0, 0.0) and prints a warning if no rows are valid.
        """
        precision_list, recall_list, f1_list = [], [], []
        n = min(sample_size, len(test_df))

        for i in range(n):
            row = test_df.iloc[i]
            true_kw = set(row["true_keywords"])
            pred_kw = set(self.extract(row["Summary"], top_n=top_n))

            if not true_kw or not pred_kw:
                continue

            tp = len(true_kw & pred_kw)
            precision = tp / len(pred_kw)
            recall = tp / len(true_kw)
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        # BUG FIX: guard against empty list before computing mean
        if not precision_list:
            print("Warning: no valid samples found for evaluation.")
            return 0.0, 0.0, 0.0

        avg_p = sum(precision_list) / len(precision_list)
        avg_r = sum(recall_list) / len(recall_list)
        avg_f = sum(f1_list) / len(f1_list)

        print(f"Evaluated on {len(precision_list)}/{n} samples")
        print(f"  Precision : {avg_p:.4f}")
        print(f"  Recall    : {avg_r:.4f}")
        print(f"  F1 Score  : {avg_f:.4f}")

        return avg_p, avg_r, avg_f


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metrics(precision: float, recall: float, f1: float) -> None:
    """Render a bar chart of Precision / Recall / F1 Score."""
    labels = ["Precision", "Recall", "F1 Score"]
    values = [precision, recall, f1]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("Keyword Extractor — Model Performance", fontsize=13, pad=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading dataset …")
    df = load_dataset()
    df = prepare_data(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df)} samples  |  Test: {len(test_df)} samples")

    extractor = KeywordExtractor(max_features=30_000, ngram_range=(1, 3))
    extractor.fit(train_df["clean_text"])
    print("Model fitted.\n")

    precision, recall, f1 = extractor.evaluate(test_df, sample_size=100)
    plot_metrics(precision, recall, f1)
