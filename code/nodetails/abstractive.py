"""
High-level API for NoDetails abstractive summarization model
""" 

import os.path
import pandas
import nodetails.util
import nodetails.prep
import nodetails.preprocess
import nodetails.nn.sequence_model
from nodetails.nn.sequence_model import (TrainingSet, Lexicon,
                                         TrainingModel, InferenceModel)


def create_model(data: pandas.DataFrame, x_len, y_len, latent_dim=500, batch_size=128,
                 show_epoch_graph=True, print_model_summary=True) -> (TrainingModel, InferenceModel):
    """Takes a pandas data frame, returns an InferenceParameters object"""

    training_set, lexicon = nodetails.prep.prepare_training_set(
        data, x_len=x_len, y_len=y_len, split=.1)

    training_model, infr_model = nodetails.nn.sequence_model.define_model(
        lexicon, latent_dim)
    
    training_model.model = nodetails.nn.sequence_model.train_model(
        training_model, training_set, batch_size,
        show_graph=show_epoch_graph)

    if print_model_summary:
        training_model.model.summary()

    return training_model, infr_model


def model_exists(save_directory, name):
    return os.path.isdir(f"{save_directory}/{name}.model")


def save(infr_params: InferenceModel, save_directory, name, verbose=True):
    nodetails.nn.sequence_model.save_sequence_model(
        infr_params, f"{save_directory}/{name}.model", verbose)


def load(save_directory, name, verbose=True) -> InferenceModel:
    return nodetails.nn.sequence_model.load_sequence_model(
        f"{save_directory}/{name}.model", verbose)


def make_inference(infr_params: InferenceModel, query: str, debug_output=False):
    (encoder_model, decoder_model,
     y_index_word, x_index_word, y_word_index,
     max_len_text, max_len_sum) = infr_params
    
    x_word_index = {v: k for k, v in x_index_word.items()}
    
    def convert_to_sequences(words):
        result = []
        for it in words:
            it = it.strip()
            if it in x_word_index:
                result.append(x_word_index[it])
            elif debug_output:
                print("Token doesn't exist on lexicon: %s" % it)

        return nodetails.util.pad_sequences([result],
                                            maxlen=max_len_text,
                                            padding="post")[0]

    query_cleaned = nodetails.preprocess.clean_text(query)

    query_seq = convert_to_sequences(query_cleaned.split())
    prediction = nodetails.nn.sequence_model.decode_sequence(query_seq.reshape(1, max_len_text),
                                                             infr_params)

    if debug_output:
        print("\n == INFERENCE ==\n")
        
        print("  Query:", query)
        print("  query_cleaned:", query_cleaned)
        print("  Summary:", prediction)

    return prediction

# END OF abstractive.py
