import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import bs4
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("../include/stopwords/english") as f:
    _stopwords_en = tuple([line for line in f.read().split("\n") if line])
with open("../include/contraction_mapping_en.txt") as f:
    _contractions_en = dict([(line.split(",")) for line in f.read().split("\n")])


def clean_text(articletext, stopwords=_stopwords_en, contractions=_contractions_en):
    articletext = bs4.BeautifulSoup(articletext.lower(), "lxml").text
    articletext = re.sub(r"\([^)]*\)", "", articletext)  # NOTE(bora): Remove any parentheses and text between them
    articletext = re.sub("\"", "", articletext)  # NOTE(bora): Remove quotation marks
    articletext = " ".join([contractions[it] if it in contractions else it for it in
                            articletext.split(" ")])  # NOTE(bora): Remove contractions
    articletext = re.sub(r"'s\b", "", articletext)  # NOTE(bora): Remove trailing "'s"s

    articletext = re.sub(r"[^a-zA-z]", " ", articletext)
    tokens = [it for it in articletext.split() if it not in stopwords]

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


def prepare_dataset(datafile, max_len_text, max_len_sum, nrows=None, verbose=False, show_histogram=False):
    cachefile_name = f"{datafile}-cache-{nrows}.gz"
    
    if os.path.exists(cachefile_name):
        if verbose:
            print("Cached data found")
            print("Loading preprocessed file")
        data = pd.read_pickle(cachefile_name)
    else:
        data = pd.read_csv(datafile, nrows=nrows)

        data.drop_duplicates(subset=["Text"], inplace=True)
        data.dropna(inplace=True)

        cleaned_text = []
        for text in data["Text"]:
            cleaned_text.append(clean_text(text))

        cleaned_summary = []
        for summary in data["Summary"]:
            cleaned_summary.append(clean_summary(summary))

        data["cleaned_text"] = cleaned_text
        data["cleaned_summary"] = cleaned_summary

        # NOTE(bora): Drop any rows with empty values
        data["cleaned_summary"].replace("", np.nan, inplace=True)
        data.dropna(inplace=True)

        # NOTE(bora): Mark start and end of each summary
        data["cleaned_summary"] = data["cleaned_summary"].apply(lambda x: f"__START__ {x} __END__")

        if verbose:
            print("Saving preprocessed data to cache file")
        data.to_pickle(cachefile_name)

    if verbose:
        print("\nCounting words")

    text_word_count = []
    for it in data["cleaned_text"]:
        text_word_count.append(len(it.split()))

    summary_word_count = []
    for it in data["cleaned_summary"]:
        summary_word_count.append(len(it.split()))

    length_df = pd.DataFrame({"text": text_word_count, "summary": summary_word_count})
    if verbose > 2 or show_histogram:
        print("Here are histograms")
        length_df.hist(bins=30)
        plt.show(block=True)

        print("Tokenizing")

    x_train, x_val, y_train, y_val = \
        train_test_split(data["cleaned_text"], data["cleaned_summary"],
                         test_size=.1, random_state=0, shuffle=True)

    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_train))

    x_train = x_tokenizer.texts_to_sequences(x_train)
    x_val = x_tokenizer.texts_to_sequences(x_val)
    x_train = pad_sequences(x_train, maxlen=max_len_text, padding="post")
    x_val = pad_sequences(x_val, maxlen=max_len_text, padding="post")

    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_train))

    y_train = y_tokenizer.texts_to_sequences(y_train)
    y_val = y_tokenizer.texts_to_sequences(y_val)
    y_train = pad_sequences(y_train, maxlen=max_len_sum, padding="post")
    y_val = pad_sequences(y_val, maxlen=max_len_sum, padding="post")

    if verbose:
        print("Preprocessing is done\n")

    return (x_train, y_train, x_val, y_val), (x_tokenizer, y_tokenizer)

# END OF util.py
