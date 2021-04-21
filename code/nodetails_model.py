import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from attention_layer import Attention
import numpy as np
import matplotlib.pyplot as plt

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


def train_model(model, parameters, training_data, validation_data,
                latent_dim=500, batch_size=512, verbose=True):
    (x_train, y_train), (x_val, y_val) = training_data, validation_data
    (x_tokenizer, y_tokenizer, max_len_text, max_len_sum, enc_input, enc_output, state_h, state_c,
     dec_input, dec_output, dec_embedding, dec_lstm, dec_dense, attn) = parameters

    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1)

    history = model.fit([x_train, y_train[:, :-1]],
                        y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:],
                        epochs=50, callbacks=[es], batch_size=batch_size,
                        validation_data=([x_val, y_val[:, :-1]],
                                         y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))

    if verbose:
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

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        enc_out, enc_h, enc_c = encoder_model.predict(input_seq)
        if verbose > 2:
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
            if verbose > 2:
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

    def seq2summary(input_seq):
        result = []
        for i in input_seq:
            if (i != 0 and i != target_word_index["start"]) and i != target_word_index["end"]:
                result.append(reverse_target_word_index[i])
        return " ".join(result)

    def seq2text(input_seq):
        result = []
        for i in input_seq:
            if i != 0:
                result.append(reverse_source_word_index[i])
        return " ".join(result)

    if verbose:
        for i in range(len(x_val)):
            print("Review:", seq2text(x_val[i]))
            print("Original summary:", seq2summary(y_val[i]))
            print("Predicted summary:", decode_sequence(x_val[i].reshape(1, max_len_text)))
            print("\n")

        print("Review:", seq2text(x_val[3]))
        print("Original summary:", seq2summary(y_val[3]))
        print("Predicted summary:", decode_sequence(x_val[3].reshape(1, max_len_text)))

# END OF nodetails_model.py
