import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model as keras_load_model

from attention_layer import Attention

_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    for it in _gpus:
        tf.config.experimental.set_memory_growth(it, True)


def define_model(x_tokenizer, y_tokenizer, max_len_text, max_len_sum,latent_dim=500):
    x_voc_size = len(x_tokenizer.word_index) + 1
    y_voc_size = len(y_tokenizer.word_index) + 1

    tf.keras.backend.clear_session()

    enc_input = layers.Input(shape=(max_len_text,))
    enc_embedding = layers.Embedding(x_voc_size, latent_dim, trainable=True)(enc_input)

    enc_lstm1 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    enc_lstm1_out, _, _ = enc_lstm1(enc_embedding)

    enc_lstm2 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    enc_lstm2_out, state_h2, state_c2 = enc_lstm2(enc_lstm1_out)

    enc_lstm3 = layers.LSTM(latent_dim, return_state=True, return_sequences=True)
    enc_output, state_h, state_c = enc_lstm3(enc_lstm2_out)

    dec_input = layers.Input(shape=(None,))
    dec_embedding = layers.Embedding(y_voc_size, latent_dim, trainable=True)
    dec_embedding_out = dec_embedding(dec_input)

    # NOTE(bora): Decoder LSTM is initialized with encoder states
    dec_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    dec_output, _fwd_state, _bkw_state = dec_lstm(dec_embedding_out, initial_state=[state_h, state_c])

    attn = Attention(name="attention_layer")
    attn_out, attn_states = attn([enc_output, dec_output])

    dec_concat_input = layers.Concatenate(axis=-1, name="concat_layer")([dec_output, attn_out])

    dec_dense = layers.TimeDistributed(layers.Dense(y_voc_size, activation="softmax"))
    dec_output = dec_dense(dec_concat_input)

    model = Model([enc_input, dec_input], dec_output)
    parameters = (x_tokenizer, y_tokenizer, max_len_text, max_len_sum, enc_input, enc_output, state_h, state_c,
                  dec_input, dec_output, dec_embedding, dec_lstm, dec_dense, attn)

    return model, parameters


def train_model(model, model_params, training_data, validation_data,
                latent_dim=500, batch_size=512, show_graph=True):
    (x_train, y_train), (x_val, y_val) = training_data, validation_data
    (x_tokenizer, y_tokenizer, max_len_text, max_len_sum, enc_input, enc_output, state_h, state_c,
     dec_input, dec_output, dec_embedding, dec_lstm, dec_dense, attn) = model_params

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
        plt.show()

    # NOTE(bora): Encoder inference
    encoder_model = Model(inputs=enc_input, outputs=[enc_output, state_h, state_c])

    # NOTE(bora): Decoder inference
    # NOTE(bora): Keep track of states from the previous time step
    decoder_state_input_h = layers.Input(shape=(latent_dim,))
    decoder_state_input_c = layers.Input(shape=(latent_dim,))
    decoder_hidden_state_input = layers.Input(shape=(max_len_text, latent_dim))

    # NOTE(bora): Get the embeddings of the decoder sequence
    dec_emb2 = dec_embedding(dec_input)

    # NOTE(bora): Set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = dec_lstm(dec_emb2,
                                                    initial_state=[decoder_state_input_h, decoder_state_input_c])

    # NOTE(bora): Attention inference
    attn_out_inf, attn_states_inf = attn([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = layers.Concatenate(axis=-1, name="concat")([decoder_outputs2, attn_out_inf])

    # NOTE(bora): A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = dec_dense(decoder_inf_concat)

    decoder_model = Model([dec_input] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
                          [decoder_outputs2] + [state_h2, state_c2])

    reverse_target_word_index = y_tokenizer.index_word
    reverse_source_word_index = x_tokenizer.index_word
    target_word_index = y_tokenizer.word_index

    return (encoder_model, decoder_model, reverse_target_word_index, reverse_source_word_index, target_word_index,
            max_len_text, max_len_sum)


def _seq2summary(input_seq, target_word_index, reverse_target_word_index):
    result = []
    for it in input_seq:
        if (it != 0 and it != target_word_index["start"]) and it != target_word_index["end"]:
            result.append(reverse_target_word_index[it])
    return " ".join(result)


def _seq2text(input_seq, reverse_source_word_index):
    result = []
    for it in input_seq:
        if it != 0:
            result.append(reverse_source_word_index[it])
    return " ".join(result)


def _decode_sequence(input_seq, model_params, debug_output=False):
    (encoder_model, decoder_model,
     reverse_target_word_index, reverse_source_word_index, target_word_index,
     max_len_text, max_len_sum) = model_params

    # Encode the input as state vectors.
    enc_out, enc_h, enc_c = encoder_model.predict(input_seq)
    if debug_output:
        print("(input_seq, e_out)", (input_seq, enc_out))
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Chose the "start" word as the first word of the target sequence
    target_seq[0, 0] = target_word_index["start"]

    _done = False
    decoded_sentence = ""
    while not _done:
        output_tokens, h, c = decoder_model.predict([target_seq] + [enc_out, enc_h, enc_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        if debug_output:
            print("sampled_token", sampled_token)
        if sampled_token != "end":
            decoded_sentence += " " + sampled_token

            # Exit condition: either hit max length or find stop word.
        if sampled_token == "end" or len(decoded_sentence.split()) >= (max_len_sum - 1):
            _done = True

        # NOTE(bora): Update the target sequence and internal states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        enc_h, enc_c = h, c

    return decoded_sentence


def test_validation_set(x_val, y_val, model_params, item_range=None, debug_output=False):
    (encoder_model, decoder_model,
     reverse_target_word_index, reverse_source_word_index, target_word_index,
     max_len_text, max_len_sum) = model_params

    def decode_seq(it):
        return _decode_sequence(it.reshape(1, max_len_text), model_params, debug_output)

    if item_range:
        range_lo, range_hi = min(item_range[0], 0), min(item_range[1], len(x_val))

        for item in range(range_lo, range_hi):
            print("\nReview:", _seq2text(x_val[item], reverse_source_word_index))
            print("Original summary:", _seq2summary(y_val[item], target_word_index, reverse_target_word_index))
            print("Predicted summary:", decode_seq(x_val[item]))
    else:
        from random import randint
        item = randint(0, len(x_val) - 1)
        print("\nItem #%d" % item)
        print("-"*len("Item #%d" % item))
        print("Review:", _seq2text(x_val[item], reverse_source_word_index))
        print("Original summary:", _seq2summary(y_val[item], target_word_index, reverse_target_word_index))
        print("Predicted summary:", decode_seq(x_val[item]))


def save_nodetails_model(model_params, save_location, debug_output=False):
    (encoder_model, decoder_model,
     reverse_target_word_index, reverse_source_word_index, target_word_index,
     max_len_text, max_len_sum) = model_params
    if debug_output:
        print(f"Saving model at {save_location}")

    # TODO(bora): Below methods still don't work as intended.
    encoder_model.save(f"{save_location}/encoder")
    decoder_model.save(f"{save_location}/decoder")
    if debug_output:
        print(f"Encoder and decoder is saved.")

    params = (reverse_target_word_index, reverse_source_word_index, target_word_index,
              max_len_text, max_len_sum)

    with open(f"{save_location}/parameters.pkl", "wb") as fp:
        pickle.dump(params, fp)
    if debug_output:
        print(f"Model saved")


def load_nodetails_model(save_location, debug_output=False):
    if debug_output:
        print(f"Loading model from {save_location}")

    # TODO(bora): Below methods still don't work as intended.
    encoder_model = keras_load_model(f"{save_location}/encoder",
                                     custom_objects={"attention_layer": Attention},
                                     compile=False)
    decoder_model = keras_load_model(f"{save_location}/decoder",
                                     custom_objects={"attention_layer": Attention},
                                     compile=False)
    if debug_output:
        print(f"Encoder and decoder is loaded.")

    with open(f"{save_location}/parameters.pkl", "rb") as fp:
        (reverse_target_word_index, reverse_source_word_index, target_word_index,
         max_len_text, max_len_sum) = pickle.load(fp)
    if debug_output:
        print(f"Model loaded")

    return (encoder_model, decoder_model,
            reverse_target_word_index, reverse_source_word_index, target_word_index,
            max_len_text, max_len_sum)

# END OF nodetails_model.py
