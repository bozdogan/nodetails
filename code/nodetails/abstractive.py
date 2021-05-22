"""
High-level API for NoDetails abstractive summarization model
"""

import nodetails.util
import nodetails.prep
import nodetails.preprocess
import nodetails.nn.sequence_model
from nodetails.nn.sequence_model import (InferenceModel)


def save(infr_params: InferenceModel, save_directory, name, verbose=True):
    nodetails.nn.sequence_model.save_model(
        infr_params, f"{save_directory}/{name}.model", verbose)


def load(save_directory, name, verbose=True) -> InferenceModel:
    return nodetails.nn.sequence_model.load_model(
        f"{save_directory}/{name}.model", verbose)

# END OF abstractive.py
