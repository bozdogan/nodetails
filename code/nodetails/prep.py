import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import sklearn.model_selection
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences 

from nodetails.types import *
from nodetails import (contraction_map_en as contraction_map, 
                       stopwords_en as stopwords)

_long_word_threshold = 3


def clean_text(txt, plain_input=False):
    if not plain_input:
        txt = BeautifulSoup(txt.lower(), "lxml").text
    txt = re.sub(r"\([^)]*\)", "", txt)
    txt = re.sub("\"", "", txt)

    txt = " ".join([contraction_map[it] if it in contraction_map else it
                    for it in txt.split(" ")])

    txt = re.sub(r"'s\b", "", txt)
    txt = re.sub(r"[^a-zA-z]", " ", txt)

    tokens = [it for it in txt.split() if it not in stopwords]
    result = [it for it in tokens if len(it) > _long_word_threshold]
    return (" ".join(result)).strip()


def clean_dataset(dataset, plain_input=False,
                  keep_original=False) -> pd.DataFrame:
    """Prepare a dataset to processed in a NLP environment

    In a given dataset, do these to all texts;
      - Convert to lowercase,
      - Remove HTML tags(*),
      - Expand contraction_map,
      - Remove extra whitespace, and
      - Remove special characters.
    Optionally keep the original text in the resulted object.

    (*) If `plain_input` is set to True, HTML tags will not be parsed. 

    The `dataset` variable must contain "text" and "sum" keys which
    corresponds the text body and summary, respectively. `dataset` is
    intended to be a pandas DataFrame but a dictionary with list of
    strings in "text" and "sum" fields would work too.

    Args:
        dataset: Input data. Must be a DataFrame or `dict`
        plain_input: If set True, HTML tags will not be parsed.
        keep_original: Whether to keep the unprocessed data

    Returns:
        pandas.DataFrame: A DataFrame with "text_cleaned" and "sum_cleaned" columns

        If `keep_original` is True, "text_orig" and "sum_orig" columns
        will be present.
    """

    # NOTE(bora): These won't work with non-English languages
    # out-of-the-box because non-English letters will be removed
    # as they're seen as special characters.

    def clean_summary(sum):
        sum = re.sub("\"", "", sum)
        sum = " ".join([contraction_map[it] if it in contraction_map else it
                        for it in sum.split(" ")])
        sum = re.sub(r"'s\b", "", sum)
        sum = re.sub(r"[^a-zA-z]", " ", sum)
        sum = sum.lower()

        tokens = sum.split()

        result = [it for it in tokens if len(it) > 1]
        return " ".join(result)

    result = pd.DataFrame()
    if keep_original:
        result["text_orig"] = dataset["text"]
        result["sum_orig"] = dataset["sum"]
    result["text_cleaned"] = [clean_text(it, plain_input) for it in dataset["text"]]
    result["sum_cleaned"] = [clean_summary(it) for it in dataset["sum"]]

    result["sum_cleaned"] = result["sum_cleaned"].replace("", np.nan)
    result = result.dropna()
    return result


def prepare_training_set(dataset, x_len=150, y_len=12, split=.1) -> (TrainingSet, Lexicon):
    assert "text_cleaned" in dataset and "sum_cleaned" in dataset,\
           "Dataset is not correctly formatted."

    dataset["sum_cleaned"] =  dataset["sum_cleaned"].apply(lambda it: f"<start> {it} <end>")
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        dataset["text_cleaned"], dataset["sum_cleaned"],
        test_size=split, random_state=17, shuffle=True)
    
    _tokenizer_filters = "!\"#$%&()*+,-./:;=?[\\]^_`{|}~\t\n"  # NOTE(bora): Keras defaults, except "<" and ">"
    x_tkn = keras.preprocessing.text.Tokenizer(filters=_tokenizer_filters)
    y_tkn = keras.preprocessing.text.Tokenizer(filters=_tokenizer_filters)

    x_tkn.fit_on_texts(x_train)
    x_train = pad_sequences(x_tkn.texts_to_sequences(x_train), maxlen=x_len, padding="post")
    x_val = pad_sequences(x_tkn.texts_to_sequences(x_val), maxlen=x_len, padding="post")

    y_tkn.fit_on_texts(y_train)
    y_train = pad_sequences(y_tkn.texts_to_sequences(y_train), maxlen=y_len + 2, padding="post")
    y_val = pad_sequences(y_tkn.texts_to_sequences(y_val), maxlen=y_len + 2, padding="post")

    # NOTE(bora): Summaries will include the start token, so we set
    # `maxlen` for y-tokenizer as `y_len + 2`. Decoder will also use
    # one word slot for the end token. In order to make the final length
    # accurate, we increment `y_len` before returning.
    y_len += 1

    return TrainingSet(x_train, y_train, x_val, y_val), Lexicon(x_tkn, y_tkn, x_len, y_len)

# END OF prep.py
