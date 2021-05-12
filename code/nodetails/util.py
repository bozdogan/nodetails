import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nodetails import DatasetResult


_INCLUDE_DIR = f"{ os.path.dirname(__file__)}/../../include".replace("\\", "/")

with open(f"{_INCLUDE_DIR}/stopwords/english") as f:
    _stopwords_en = tuple([line for line in f.read().split("\n") if line])

with open(f"{_INCLUDE_DIR}/contraction_mapping_en.txt") as f:
    _contractions_en = dict([(line.split(",")) for line in f.read().split("\n")])


def clean_text(articletext, stopwords=_stopwords_en, contractions=_contractions_en):

    # NOTE(bora): Remove any HTML tags, content inside parantheses,
    # quotation marks, and word-final "'s"s. Also expand contractions
    # to their full form.
    txt = BeautifulSoup(articletext.lower(), "lxml").text
    txt = re.sub(r"\([^)]*\)", "", txt)
    txt = re.sub("\"", "", txt)
    txt = " ".join([contractions[it] if it in contractions else it for it in txt.split(" ")])
    txt = re.sub(r"'s\b", "", txt)
    txt = re.sub(r"[^a-zA-z]", " ", txt)

    tokens = [it for it in txt.split() if it not in stopwords]
    long_words = [it for it in tokens if len(it) > 3]
    return (" ".join(long_words)).strip()


def clean_summary(summary, contractions=_contractions_en):
    summary = re.sub("\"", "", summary)  # NOTE(bora): Remove quotation marks
    summary = " ".join([contractions[it] if it in contractions else it for it in
                        summary.split(" ")])  # NOTE(bora): Expand contractions
    summary = re.sub(r"'s\b", "", summary)  # NOTE(bora): Remove trailing "'s"s
    summary = re.sub(r"[^a-zA-z]", " ", summary)
    summary = summary.lower()

    tokens = summary.split()

    result = " ".join([it for it in tokens if len(it) > 1])
    return result


def clean_dataset(dataset, keep_original=False, verbose=False):
    """Removes punctuation and converts to lowercase.
    
    :param dataset: A mapping with equal length lists in 
        its "text" and "sum" fields.
    :param keep_original: Whether or not to keep original text
        in the result. (CURRENTLY NOT IMPLEMENTED)
    """
    assert "text" in dataset and "sum" in dataset

    text_cleaned = [clean_text(it) for it in dataset["text"]]
    sum_cleaned = [clean_summary(it) for it in dataset["sum"]]
    result = pd.DataFrame({"text_cleaned": text_cleaned,
                           "sum_cleaned": sum_cleaned})

    result["sum_cleaned"] = result["sum_cleaned"].replace("", np.nan)
    result = result.dropna()
    result["sum_cleaned"].apply(lambda it: f"__START__ {it} __END__")

    return result


# TODO(bora): This function is not universal. Needs to be modified before
# processing different datasets.
def preprocess_dataset(datafile, nrows=None, verbose=False, show_histogram=False):
    cachefile_name = f"{datafile}-cache-{nrows}.gz"
    
    if os.path.exists(cachefile_name):
        if verbose:
            print("Cached data found")
            print("Loading preprocessed file")
        data = pd.read_pickle(cachefile_name)
    else:
        data = (pd.read_csv(datafile, nrows=10000)
                  .drop_duplicates(subset=["Text"])
                  .dropna()
                  .rename(columns={"Text": "text", "Summary": "sum"}))

        data = clean_dataset(data)

        if verbose:
            print("Saving preprocessed data to cache file")
        data.to_pickle(cachefile_name)

    if verbose:
        print("\nCounting words")

    if show_histogram:
        text_word_count = [len(it.split()) for it in data["text_cleaned"]]
        summary_word_count = [len(it.split()) for it in data["sum_cleaned"]]
        
        length_df = pd.DataFrame({"text": text_word_count, "summary": summary_word_count})
        print("Here are histograms")
        length_df.hist(bins=30)
        plt.show()

    return data


def prepare_for_training(data: pd.DataFrame, max_len_text, max_len_sum, verbose=False):
    if verbose:
        print("Spliting train/test sets")
    x_train, x_val, y_train, y_val = \
        train_test_split(data["text_cleaned"], data["sum_cleaned"],
                         test_size=.1, random_state=0, shuffle=True)

    if verbose:
        print("Tokenizing")

    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_train))

    x_train = pad_sequences(x_tokenizer.texts_to_sequences(x_train),
                            maxlen=max_len_text, padding="post")
    x_val = pad_sequences(x_tokenizer.texts_to_sequences(x_val),
                          maxlen=max_len_text, padding="post")

    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_train))

    y_train = pad_sequences(y_tokenizer.texts_to_sequences(y_train),
                            maxlen=max_len_sum, padding="post")
    y_val = pad_sequences(y_tokenizer.texts_to_sequences(y_val),
                          maxlen=max_len_sum, padding="post")

    if verbose:
        print("Preprocessing is done\n")

    return DatasetResult(x_train, y_train, x_val, y_val,
                         x_tokenizer, y_tokenizer)


def prepare_dataset(datafile, max_len_text, max_len_sum, nrows=None, verbose=False, show_histogram=False):
    data = preprocess_dataset(datafile, nrows, verbose, show_histogram)
    return prepare_for_training(data, max_len_text, max_len_sum, verbose)

# END OF util.py
