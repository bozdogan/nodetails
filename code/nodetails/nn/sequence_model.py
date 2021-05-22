import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.callbacks import EarlyStopping

from nodetails import ModelSpecs, InferenceParameters
from nodetails.nn.attention import Attention


def define_model(x_tkn, y_tkn, x_len, y_len, latent_dim=500):
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

    return ModelSpecs(x_tkn, y_tkn, x_len, y_len, latent_dim,
                      encoder_input, encoder_output, state_h, state_c,
                      decoder_input, output, decoder_embedding, decoder_lstm, decoder_dense, attention)


def train_model(model_params: ModelSpecs, training_data, validation_data, batch_size=128, show_graph=True):
    (x_train, y_train), (x_val, y_val) = training_data, validation_data

    (_, _, _, _, _,
     enc_input, _, _, _,
     dec_input, dec_output, _, _, _, _) = model_params

    model = Model([enc_input, dec_input], dec_output)

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


def prep_model_for_inference(model_params: ModelSpecs):
    (x_tokenizer, y_tokenizer, max_len_text, max_len_sum, latent_dim,
     enc_input, enc_output, state_h, state_c,
     decoder_input, dec_output, dec_embedding, decoder_lstm, decoder_dense, attention) = model_params

    infr_encoder_model = Model(inputs=enc_input,
                               outputs=[enc_output, state_h, state_c])

    infr_prev_h = layers.Input(shape=(latent_dim,))
    infr_prev_c = layers.Input(shape=(latent_dim,))
    infr_prev_hidden = layers.Input(shape=(max_len_text, latent_dim))
    infr_embedded = dec_embedding(decoder_input)

    infr_hidden1, infr_state_h, infr_state_c = decoder_lstm(
        infr_embedded, initial_state=[infr_prev_h, infr_prev_c])
    infr_hidden2, _ = attention([infr_prev_hidden, infr_hidden1])
    infr_hidden3 = layers.Concatenate(axis=-1)([infr_hidden1, infr_hidden2])

    infr_output = decoder_dense(infr_hidden3)
    decoder_model = Model([decoder_input] + [infr_prev_hidden, infr_prev_h, infr_prev_c],
                          [infr_output] + [infr_state_h, infr_state_c])

    y_index_word = y_tokenizer.index_word
    x_index_word = x_tokenizer.index_word
    y_word_index = y_tokenizer.word_index

    return InferenceParameters(infr_encoder_model, decoder_model,
                               y_index_word, x_index_word, y_word_index,
                               max_len_text, max_len_sum)


def _seq2summary(input_seq, y_word_index, y_index_word):
    result = []
    for it in input_seq:
        if (it != 0 and it != y_word_index["start"]) and it != y_word_index["end"]:
            result.append(y_index_word[it])
    return " ".join(result)


def _seq2text(input_seq, x_index_word):
    result = []
    for it in input_seq:
        if it != 0:
            result.append(x_index_word[it])
    return " ".join(result)


def decode_sequence(input_seq, infr_params: InferenceParameters, debug_output=False):
    (encoder_model, decoder_model,
     y_index_word, x_index_word, y_word_index,
     max_len_text, max_len_sum) = infr_params

    # Encode the input as state vectors.
    enc_out, enc_h, enc_c = encoder_model.predict(input_seq)
    if debug_output:
        print("(input_seq, e_out)", (input_seq, enc_out))

    target_seq = np.zeros((1, 1))  # Generate empty target sequence
    target_seq[0, 0] = y_word_index["start"]  # Set "start" as the first word of the target sequence

    _done = False
    decoded_sentence = ""
    while not _done:
        output_tokens, h, c = decoder_model.predict([target_seq] + [enc_out, enc_h, enc_c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            # NOTE(bora): 0 is selected as sequence index. Probably because model is
            # under-trained. Exit the loop to prevent KeyError. Result will be an
            # empty string.
            break

        sampled_token = y_index_word[sampled_token_index]
        if debug_output:
            print("sampled_token", sampled_token)
        if sampled_token != "end":
            decoded_sentence += " " + sampled_token

            # Exit condition. Either hit max length or find "end"
        if sampled_token == "end" or len(decoded_sentence.split()) >= (max_len_sum - 1):
            _done = True

        # Update the target sequence and internal states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        enc_h, enc_c = h, c

    return decoded_sentence


def save_sequence_model(infr_params: InferenceParameters, save_location, verbose=True):
    (encoder_model, decoder_model,
     y_index_word, x_index_word, y_word_index,
     max_len_text, max_len_sum) = infr_params
    if verbose:
        print(f"Saving model at {save_location}")

    encoder_model.save(f"{save_location}/encoder")
    decoder_model.save(f"{save_location}/decoder")
    if verbose:
        print(f"Encoder and decoder is saved.")

    params = (y_index_word, x_index_word, y_word_index,
              max_len_text, max_len_sum)

    with open(f"{save_location}/parameters.pkl", "wb") as fp:
        pickle.dump(params, fp)
    if verbose:
        print(f"Model saved")


def load_sequence_model(save_location, verbose=True):
    if verbose:
        print(f"Loading model from {save_location}")

    encoder_model = keras_load_model(f"{save_location}/encoder",
                                     custom_objects={"Attention": Attention},
                                     compile=False)
    decoder_model = keras_load_model(f"{save_location}/decoder",
                                     custom_objects={"Attention": Attention},
                                     compile=False)
    if verbose:
        print(f"Encoder and decoder is loaded.")

    with open(f"{save_location}/parameters.pkl", "rb") as fp:
        (y_index_word, x_index_word, y_word_index,
         max_len_text, max_len_sum) = pickle.load(fp)
    if verbose:
        print(f"Model loaded")

    return InferenceParameters(encoder_model, decoder_model,
                               y_index_word, x_index_word, y_word_index,
                               max_len_text, max_len_sum)


def test_validation_set(x_val, y_val, infr_params: InferenceParameters, item_range=(0, 1),
                        debug_output=False, silent=False):
    if silent: debug_output = False
    (encoder_model, decoder_model,
     y_index_word, x_index_word, y_word_index,
     max_len_text, max_len_sum) = infr_params

    def decode_validation_seq(it):
        result = decode_sequence(x_val[it].reshape(1, max_len_text), infr_params, debug_output)
        assert result, f"Empty result of type {type(result)} at item #{it}"
        return result

    range_lo = max(0, min(item_range[0], len(x_val)))
    range_hi = min(item_range[1], len(x_val))

    for item in range(range_lo, range_hi):
        review = _seq2text(x_val[item], x_index_word)
        sum_orig = _seq2summary(y_val[item], y_word_index, y_index_word)
        sum_pred = decode_validation_seq(item)
        if not silent:
            print("\nReview #%s: %s" % (item, review))
            print("Original summary:", sum_orig)
            print("Predicted summary:", sum_pred)

# END OF sequence_model.py
