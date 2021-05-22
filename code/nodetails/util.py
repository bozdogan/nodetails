import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nodetails import extractive, abstractive
from nodetails.preprocess import clean_dataset

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
        data = (pd.read_csv(datafile, nrows=nrows)
                  .drop_duplicates(subset=["Text"])
                  .dropna()
                  .rename(columns={"Text": "text", "Summary": "sum"}))

        data = clean_dataset(data, keep_original=True)

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


# def prepare_for_training(data: pd.DataFrame, max_len_text, max_len_sum, verbose=False):
#     if verbose:
#         print("Spliting train/test sets")
#     x_train, x_val, y_train, y_val = \
#         train_test_split(data["text_cleaned"], data["sum_cleaned"],
#                          test_size=.1, random_state=0, shuffle=True)
#
#     if verbose:
#         print("Tokenizing")
#
#     x_tokenizer = Tokenizer()
#     x_tokenizer.fit_on_texts(list(x_train))
#
#     x_train = pad_sequences(x_tokenizer.texts_to_sequences(x_train),
#                             maxlen=max_len_text, padding="post")
#     x_val = pad_sequences(x_tokenizer.texts_to_sequences(x_val),
#                           maxlen=max_len_text, padding="post")
#
#     y_tokenizer = Tokenizer()
#     y_tokenizer.fit_on_texts(list(y_train))
#
#     y_train = pad_sequences(y_tokenizer.texts_to_sequences(y_train),
#                             maxlen=max_len_sum, padding="post")
#     y_val = pad_sequences(y_tokenizer.texts_to_sequences(y_val),
#                           maxlen=max_len_sum, padding="post")
#
#     if verbose:
#         print("Preprocessing is done\n")
#
#     return DatasetResult(x_train, y_train, x_val, y_val,
#                          x_tokenizer, y_tokenizer)


# def prepare_dataset(datafile, max_len_text, max_len_sum, nrows=None, verbose=False, show_histogram=False):
#     data = preprocess_dataset(datafile, nrows, verbose, show_histogram)
#     return prepare_for_training(data, max_len_text, max_len_sum, verbose)

# def summary_from_wikipedia(article_url, abst_model: InferenceParameters):
#     extsum = extractive.get_summary_from_url(article_url, 10, preset="wikipedia")
#     abstsum = abstractive.make_inference(abst_model, extsum.summary, debug_output=True)
#
#     return abstsum

# END OF util.py