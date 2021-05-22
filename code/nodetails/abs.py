import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.callbacks import EarlyStopping

import nodetails.nn
import nodetails.prep
import nodetails.util

from nodetails import *
from nodetails import InferenceModel
from nodetails.nn.attention import Attention


def create_models(lexicon: Lexicon, latent_dim=500):
    x_tkn, y_tkn, x_len, y_len = lexicon

    encoder_vocab = len(x_tkn.word_index) + 1
    decoder_vocab = len(y_tkn.word_index) + 1

    encoder_input = layers.Input(shape=(x_len,))
    embedded1 = layers.Embedding(encoder_vocab, latent_dim, trainable=True, input_shape=(x_len,))(encoder_input)

    hidden_layer1 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    hidden1, _, _ = hidden_layer1(embedded1)

    hidden_layer2 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    hidden2, _, _ = hidden_layer2(hidden1)

    hidden_layer3 = layers.LSTM(latent_dim, return_state=True, return_sequences=True)
    encoder_output, state_h, state_c = hidden_layer3(hidden2)
    encoder_state = [state_h, state_c]

    decoder_input = layers.Input(shape=(None,))
    decoder_embedding = layers.Embedding(decoder_vocab, latent_dim, trainable=True)
    embedded2 = decoder_embedding(decoder_input)

    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder_lstm(embedded2, initial_state=encoder_state)

    attention = Attention()
    hidden4, _ = attention([encoder_output, decoder_output])

    hidden5 = layers.Concatenate(axis=-1)([decoder_output, hidden4])

    decoder_dense = layers.TimeDistributed(
        layers.Dense(decoder_vocab, activation="softmax"))
    output = decoder_dense(hidden5)

    training_model = Model([encoder_input, decoder_input], output)

    infr_encoder_model = Model(inputs=encoder_input,
                               outputs=[encoder_output, state_h, state_c])

    infr_prev_h = layers.Input(shape=(latent_dim,))
    infr_prev_c = layers.Input(shape=(latent_dim,))
    infr_prev_hidden = layers.Input(shape=(x_len, latent_dim))
    infr_embedded = decoder_embedding(decoder_input)

    infr_hidden1, infr_state_h, infr_state_c = decoder_lstm(
        infr_embedded, initial_state=[infr_prev_h, infr_prev_c])
    infr_hidden2, _ = attention([infr_prev_hidden, infr_hidden1])
    infr_hidden3 = layers.Concatenate(axis=-1)([infr_hidden1, infr_hidden2])

    infr_output = decoder_dense(infr_hidden3)
    infr_decoder_model = Model([decoder_input] + [infr_prev_hidden, infr_prev_h, infr_prev_c],
                          [infr_output] + [infr_state_h, infr_state_c])

    return (TrainingModel(training_model, latent_dim),
            InferenceModel(infr_encoder_model, infr_decoder_model, lexicon))


def train_model(training_model: TrainingModel, training_set: TrainingSet,
                batch_size=64, show_graph=False):
    x_train, y_train, x_val, y_val, = training_set
    model = training_model.model

    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1)

    history = model.fit([x_train, y_train[:, :-1]],
                        y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:],
                        epochs=50, callbacks=[es], batch_size=batch_size,
                        validation_data=([x_val, y_val[:, :-1]],
                                         y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))

    if show_graph:
        plt.figure()
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="test")
        plt.legend()
        plt.show(block=True)

    return model


def save_model(infr_model: InferenceModel, save_location, verbose=True):
    encoder_model, decoder_model, lexicon = infr_model
    if verbose:
        print(f"Saving model at {save_location}")

    encoder_model.save(f"{save_location}/encoder.h5")
    decoder_model.save(f"{save_location}/decoder.h5")
    if verbose:
        print(f"Encoder and decoder is saved.")

    params = lexicon

    with open(f"{save_location}/parameters.pkl", "wb") as fp:
        pickle.dump(params, fp)
    if verbose:
        print(f"Model saved")


def load_model(save_location, verbose=True) -> InferenceModel:
    if verbose:
        print(f"Loading model from {save_location}")

    encoder_model = keras_load_model(f"{save_location}/encoder.h5",
                                     custom_objects={"Attention": Attention},
                                     compile=False)
    decoder_model = keras_load_model(f"{save_location}/decoder.h5",
                                     custom_objects={"Attention": Attention},
                                     compile=False)
    if verbose:
        print(f"Encoder and decoder is loaded.")

    with open(f"{save_location}/parameters.pkl", "rb") as fp:
        lexicon = pickle.load(fp)
    if verbose:
        print(f"Model loaded")

    return InferenceModel(encoder_model, decoder_model, lexicon)


def decode_sequence(input_seq, infr_model: InferenceModel):
    encoder_model, decoder_model, (_, y_tkn, _, y_len) = infr_model

    encoder_output, state_h, state_c = encoder_model.predict(input_seq)
    encoder_state = [state_h, state_c]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = y_tkn.word_index["<start>"]

    _done = False
    result = []
    while not _done:
        output, state_h, state_c = decoder_model.predict(
            [target_seq, encoder_output] + encoder_state)

        sampled_index = np.argmax(output[0, -1, :])
        if sampled_index == 0:
            _done = True
        else:
            sampled_token = y_tkn.index_word[sampled_index]
            # print("sampled_token", sampled_token)
            if sampled_token != "<end>":
                result.append(sampled_token)

            if sampled_token == "<end>" or len(result) >= y_len - 1:
                _done = True
            else:
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = sampled_index
                encoder_state = [state_h, state_c]

    return " ".join(result)


def seq2text(input_seq, tkn):
    result = [tkn.index_word[it] for it in input_seq if it != 0]
    return " ".join(result)


def test_validation_set(infr_model: InferenceModel, x_val, y_val, lexicon, item_range=(0, 1)):
    x_tkn, y_tkn, x_len, y_len = lexicon

    def decode_validation_seq(it):
        result = decode_sequence(it.reshape(1, x_len), infr_model)
        assert result, f"Empty result of type {type(result)} at item #{it}"
        return result

    for i in range(*item_range):
        review = seq2text(x_val[i], x_tkn)
        sum_orig = (seq2text(y_val[i], y_tkn).replace("<start>", "")
                                             .replace("<end>", "")
                                             .strip())
        sum_pred = decode_validation_seq(x_val[i])
        print("\nReview #%s: %s" % (i, review))
        print("Original summary:", sum_orig)
        print("Predicted summary:", sum_pred)


def make_inference(infr_model: InferenceModel, query: str, debug_output=False):
    (encoder_model, decoder_model,
     (x_tkn, y_tkn, x_len, y_len)) = infr_model

    def convert_to_sequences(words):
        result = []
        for it in words:
            it = it.strip()
            if it in x_tkn.word_index:
                result.append(x_tkn.word_index[it])
            elif debug_output:
                print("Token doesn't exist on lexicon: %s"%it)

        return nodetails.prep.pad_sequences([result],
                                            maxlen=x_len,
                                            padding="post")[0]

    query_cleaned = nodetails.prep.clean_text(query)

    query_seq = convert_to_sequences(query_cleaned.split())
    prediction = decode_sequence(query_seq.reshape(1, x_len), infr_model)
    if debug_output:
        print("\n == INFERENCE ==\n")

        print("  Query:", query)
        print("  query_cleaned:", query_cleaned)
        print("  Summary:", prediction)

    return prediction

# END OF abs.py
