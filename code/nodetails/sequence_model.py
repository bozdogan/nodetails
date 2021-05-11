import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.callbacks import EarlyStopping

from nodetails import ModelParameters, InferenceParameters
from nodetails.attention import Attention

_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    for it in _gpus:
        tf.config.experimental.set_memory_growth(it, True)


def define_model(x_tokenizer, y_tokenizer, max_len_text, max_len_sum, latent_dim=500):
    x_voc_size = len(x_tokenizer.word_index) + 1
    y_voc_size = len(y_tokenizer.word_index) + 1

    tf.keras.backend.clear_session()

    enc_input = layers.Input(shape=(max_len_text,))
    enc_embedding = layers.Embedding(x_voc_size, latent_dim, trainable=True, input_shape=(max_len_text,))(enc_input)

    enc_lstm1 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    enc_lstm1_out, _, _ = enc_lstm1(enc_embedding)

    enc_lstm2 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    enc_lstm2_out, _, _ = enc_lstm2(enc_lstm1_out)

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

    return ModelParameters(x_tokenizer, y_tokenizer, max_len_text, max_len_sum, latent_dim,
                           enc_input, enc_output, state_h, state_c,
                           dec_input, dec_output, dec_embedding, dec_lstm, dec_dense, attn)


def train_model(model_params: ModelParameters, training_data, validation_data, batch_size=128, show_graph=True):
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


def prep_model_for_inference(model_params: ModelParameters):
    (x_tokenizer, y_tokenizer, max_len_text, max_len_sum, latent_dim,
     enc_input, enc_output, state_h, state_c,
     dec_input, dec_output, dec_embedding, dec_lstm, dec_dense, attn) = model_params

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

    y_index_word = y_tokenizer.index_word
    x_index_word = x_tokenizer.index_word
    y_word_index = y_tokenizer.word_index

    return InferenceParameters(encoder_model, decoder_model,
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
