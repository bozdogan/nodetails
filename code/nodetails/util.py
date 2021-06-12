import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt

from nodetails.types import *
from nodetails import is_debug
from nodetails import ndabs, ndext, prep


def cached(fn):
    def cached_wrapper(datafile, renaming_map=None, nrows=None,
                       cache_dir=None):
        if renaming_map is None:
            renaming_map = {"Text": "text", "Summary": "sum"}
        if cache_dir is None:
            cache_dir = osp.dirname(datafile)

        cache_filename = osp.join(cache_dir, f"{osp.basename(datafile)}-cache-{nrows}.gz")

        if osp.exists(cache_filename):
            if is_debug():
                print("Cached data found")
                print("Loading preprocessed file")
            data = pd.read_pickle(cache_filename)
        else:
            data = fn(datafile, renaming_map=renaming_map, nrows=nrows)

            if is_debug():
                print("Saving preprocessed data to cache file")
            data.to_pickle(cache_filename)

        if is_debug():
            print("\nCounting words")

        return data

    return cached_wrapper


def read_dataset_csv(input_file, renaming_map: dict, nrows=None):
    data = (pd.read_csv(input_file, nrows=nrows)
              .rename(columns=renaming_map)
              .drop_duplicates(subset=["text"])
              .dropna())

    data = prep.clean_dataset(data, keep_original=True)
    return data


cached_read_dataset_csv = cached(read_dataset_csv)


def show_word_count_graphs(data: pd.DataFrame, hist_bins=30):
    txt_word_count = [len(it.split()) for it in data["text_cleaned"]]
    sum_word_count = [len(it.split()) for it in data["sum_cleaned"]]

    (pd.DataFrame({"Text": txt_word_count,
                   "Summary": sum_word_count})
       .hist(bins=hist_bins))

    plt.show()


def summary_from_wikipedia(article_url, abs_model: AbstractiveModel):
    extsum = ndext.get_summary_from_url(article_url, 10, preset="wikipedia")
    abstsum = ndabs.make_inference(abs_model, extsum.summary)

    return abstsum

# END OF util.py
