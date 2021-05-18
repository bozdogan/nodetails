import os
import pandas as pd

from nodetails import InferenceParameters, extractive, abstractive
from nodetails import _DEBUG


def cached(fn):
    def cached_wrapper(datafile, nrows, *args, **kwargs):
        cachefile_name = f"{datafile}-cache-{nrows}.gz"
        
        if os.path.exists(cachefile_name):
            if _DEBUG:
                print("Cached data found")
                print("Loading preprocessed file")
            data = pd.read_pickle(cachefile_name)
        else:
            data= fn(datafile, nrows, *args, **kwargs)

            if _DEBUG:
                print("Saving preprocessed data to cache file")
            data.to_pickle(cachefile_name)

        return data

    return cached_wrapper


def summary_from_wikipedia(article_url, abst_model: InferenceParameters):
    extsum = extractive.get_summary_from_url(article_url, 10, preset="wikipedia")
    abstsum = abstractive.make_inference(abst_model, extsum.summary)

    return abstsum

# END OF util.py
