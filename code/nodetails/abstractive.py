"""
High-level API for NoDetails abstractive summarization model
""" 

import os.path
import pandas
import nodetails.util
import nodetails.prep
import nodetails.nn.sequence_model
from nodetails import DatasetResult, InferenceParameters
from nodetails import _DEBUG


def create_model(data: DatasetResult, textlen, sumlen, latent_dim=500, batch_size=128,
                 show_epoch_graph=False, print_model_summary=_DEBUG) -> InferenceParameters:
    """Takes a pandas data frame, returns an InferenceParameters object"""

    (x_train, y_train, x_val, y_val,
     x_tokenizer, y_tokenizer) = data

    model_params = nodetails.nn.sequence_model.define_model(
        x_tokenizer, y_tokenizer, textlen, sumlen, latent_dim)
    
    model = nodetails.nn.sequence_model.train_model(
        model_params, (x_train, y_train), (x_val, y_val),
        batch_size=batch_size,
        show_graph=show_epoch_graph)

    if print_model_summary:
        model.summary()

    return nodetails.nn.sequence_model.prep_model_for_inference(model_params)


def prepare_dataset(datafile, textlen, sumlen,
                    nrows=None, show_histogram=False) -> DatasetResult:
    data = nodetails.prep.preprocess_dataset(datafile, nrows, show_histogram)
    return nodetails.prep.split_train_test(data, textlen, sumlen)


def save(infr_params: InferenceParameters, savepath):
    nodetails.nn.sequence_model.save_sequence_model(infr_params, savepath)


def load(savepath):
    return nodetails.nn.sequence_model.load_sequence_model(savepath)


def make_inference(infr_params: InferenceParameters, query: str):
    (encoder_model, decoder_model,
     y_index_word, x_index_word, y_word_index,
     textlen, sumlen) = infr_params
    
    x_word_index = {v: k for k, v in x_index_word.items()}
    
    def convert_to_sequences(words):
        result = []
        for it in words:
            it = it.strip()
            if it in x_word_index:
                result.append(x_word_index[it])
            elif _DEBUG:
                print("Token doesn't exist on lexicon: %s" % it)

        return nodetails.util.pad_sequences([result],
                                            maxlen=textlen,
                                            padding="post")[0]

    query_cleaned = nodetails.preprocess.clean_text(query)

    query_seq = convert_to_sequences(query_cleaned.split())
    prediction = nodetails.nn.sequence_model.decode_sequence(query_seq.reshape(1, textlen),
                                                             infr_params)

    if _DEBUG:
        print("\n == INFERENCE ==\n")
        
        print("  Query:", query)
        print("  query_cleaned:", query_cleaned)
        print("  Summary:", prediction)

    return prediction

# END OF abstractive.py
