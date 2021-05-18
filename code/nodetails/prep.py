import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from bs4 import BeautifulSoup
import sklearn.model_selection
from tensorflow import keras

from nodetails import util
from nodetails import DatasetResult
from nodetails import _DEBUG


_INCLUDE_DIR = f"{ os.path.dirname(__file__)}/../../include".replace("\\", "/")

with open(f"{_INCLUDE_DIR}/stopwords/english") as f:
    _stopwords_en = tuple([line for line in f.read().split("\n") if line])

with open(f"{_INCLUDE_DIR}/contraction_mapping_en.txt") as f:
    _contractions_en = dict([(line.split(",")) for line in f.read().split("\n")])


def clean_dataset(dataset, keep_original=False):
    """Removes punctuation and converts to lowercase.
    
    :param dataset: A mapping with equal length lists in 
        its "text" and "sum" fields.
    :param keep_original: Whether or not to keep original text
        in the result. (CURRENTLY NOT IMPLEMENTED)
    """
    assert "text" in dataset and "sum" in dataset

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

    result = pd.DataFrame()
    if keep_original:
        result["text_orig"] = dataset["text"]
        result["sum_orig"] = dataset["sum"]
    result["text_cleaned"] = [clean_text(it) for it in dataset["text"]]
    result["sum_cleaned"] = [clean_summary(it) for it in dataset["sum"]]

    result["sum_cleaned"] = result["sum_cleaned"].replace("", np.nan)
    result = result.dropna()
    result["sum_cleaned"].apply(lambda it: f"__START__ {it} __END__")

    return result


def convert_to_padded_sequences(data, maxlen, tokenizer=None):
    if isinstance(data, str):
        data = [data]

    if tokenizer is None:
        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(list(data) + ["start", "end"])
    
    data_seq = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(data), maxlen=maxlen, padding="post")

    return data_seq, tokenizer


@util.cached
def preprocess_dataset(datafile, nrows=None, show_histogram=False):
    data = (pd.read_csv(datafile, nrows=nrows)
              .drop_duplicates(subset=["Text"])
              .dropna()
              .rename(columns={"Text": "text", "Summary": "sum"}))

    data = clean_dataset(data, keep_original=True)

    if _DEBUG:
        print("\nCounting words")

    if show_histogram:
        text_word_count = [len(it.split()) for it in data["text_cleaned"]]
        summary_word_count = [len(it.split()) for it in data["sum_cleaned"]]
        
        length_df = pd.DataFrame({"text": text_word_count, "summary": summary_word_count})
        print("Here are histograms")
        length_df.hist(bins=30)
        plt.show()

    return data


def split_train_test(data: pd.DataFrame, textlen, sumlen):
    if _DEBUG:
        print("Spliting train/test sets")

    (x_train_str, x_val_str,
     y_train_str, y_val_str) = sklearn.model_selection.train_test_split(
        data["text_cleaned"], data["sum_cleaned"],
        test_size=.1, random_state=0, shuffle=True)

    if _DEBUG:
        print("Tokenizing")

    x_train, x_tokenizer = convert_to_padded_sequences(x_train_str, textlen)
    x_val, _ = convert_to_padded_sequences(x_val_str, textlen, x_tokenizer)

    y_train, y_tokenizer = convert_to_padded_sequences(y_train_str, sumlen)
    y_val, _ = convert_to_padded_sequences(y_val_str, sumlen, y_tokenizer)

    if _DEBUG:
        print("Preprocessing is done\n")

    return DatasetResult(x_train, y_train, x_val, y_val,
                         x_tokenizer, y_tokenizer)

# END OF preprocess.py
