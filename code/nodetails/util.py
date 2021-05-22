import os
import pandas as pd
import matplotlib.pyplot as plt

from nodetails import ext
from nodetails import InferenceModel
from nodetails.prep import clean_dataset


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

def summary_from_wikipedia(article_url, abs_infr_model: InferenceModel):
    extsum = ext.get_summary_from_url(article_url, 10, preset="wikipedia")
    abstsum = abs.make_inference(abs_infr_model, extsum.summary, debug_output=True)

    return abstsum

# END OF util.py