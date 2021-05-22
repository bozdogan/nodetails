import os
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup

from nodetails import contraction_map_en, stopwords_en


def clean_text(articletext, stopwords=stopwords_en, contractions=contraction_map_en):

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


def clean_summary(summary, contractions=contraction_map_en):
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

    result = pd.DataFrame()
    if keep_original:
        result["text_orig"] = dataset["text"]
        result["sum_orig"] = dataset["sum"]
    result["text_cleaned"] = [clean_text(it) for it in dataset["text"]]
    result["sum_cleaned"] = [clean_summary(it) for it in dataset["sum"]]

    result["sum_cleaned"] = result["sum_cleaned"].replace("", np.nan)
    result = result.dropna()
    result["sum_cleaned"] = result["sum_cleaned"].apply(lambda it: f"__START__ {it} __END__")

    return result


# END OF preprocess.py