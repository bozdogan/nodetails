_README.md_  
_a.k.a. Vize Raporu_

No Details: Essence of the text
===============================

- We developed a deep learning model that learns how to summarize an article, and an accompanying application to the model.
- Source code is a proof of concept that the model works, though a corpus of longer articles is needed for the final application.


- The text needed to be cleaned before being fed into the neural network. Contractions are expanded, punctuations and numbers removed, and all the text is converted to lowercase. Then it is tokenized and padded for training or prediction.


- The model is a recurrent neural network. It mainly relies on LSTM cells and their ability to understand sequence data. It has two parts: an encoder and a decoder.
- The encoder has an embedding layer that converts the words into a vector, and LSTM layers to calculate the hidden state. The embedding layer helps the model make the distinction of different types of words _e.g.,_ noun, verb, adjective etc.
- The output of the encoder is sent as input for the decoder. The decoder consists of an LSTM layer and  a dense layer that predicts the summary of the paragraph.
- Model uses the fast RMSprop optimizer with "Sparse Categorical Crossentropy" loss function


Source Code
-----------

Driver code
`sum_abstractive.py`:
```python
from nodetails_model import *
import preprocess


# No Details: Essence of the text
if __name__ == "__main__":
    DATA_DIR = "../data/food_reviews"
    MODEL_DIR = "../models"

    DATA_SIZE = 100000
    BATCH_SIZE = 128

    data_file = f"{DATA_DIR}/Reviews.csv"
    MAX_LEN_TEXT = 80
    MAX_LEN_SUM = 10
    MODEL_NAME = f"nodetails--{MAX_LEN_TEXT}-{MAX_LEN_SUM}-{BATCH_SIZE}"

    (x_train, y_train, x_val, y_val), (x_tokenizer, y_tokenizer) = \
        preprocess.prepare_dataset(data_file,
                                   nrows=DATA_SIZE,
                                   max_len_text=MAX_LEN_TEXT,
                                   max_len_sum=MAX_LEN_SUM)

    # Deep learning part
    model, parameters = define_model(x_tokenizer, y_tokenizer,
                                     max_len_text=MAX_LEN_TEXT,
                                     max_len_sum=MAX_LEN_SUM)
    model.summary()

    model = train_model(model,
                        (x_train, y_train), (x_val, y_val),
                        batch_size=BATCH_SIZE)
    model_params = prep_for_inference(model, parameters)
    
    test_validation_set(x_val, y_val, model_params,
                        item_range=(0, 10))

    print("Done.")

# END OF sum_abstractive.py
```

`preprocess.py`:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import bs4
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(articletext, stopwords=stopwords_en, contractions=contractions_en):
    articletext = bs4.BeautifulSoup(articletext.lower(), "lxml").text
    articletext = re.sub(r"\([^)]*\)", "", articletext)  # Remove any parentheses and text between them
    articletext = re.sub("\"", "", articletext)  # Remove quotation marks
    articletext = " ".join([contractions[it] if it in contractions else it for it in
                            articletext.split(" ")])  # Remove contractions
    articletext = re.sub(r"'s\b", "", articletext)  # Remove trailing "'s"s

    articletext = re.sub(r"[^a-zA-z]", " ", articletext)
    tokens = [it for it in articletext.split() if it not in stopwords]

    long_words = [it for it in tokens if len(it) > 3]
    return (" ".join(long_words)).strip()


def clean_summary(summary, contractions):
    summary = re.sub("\"", "", summary)  # Remove quotation marks
    summary = " ".join([contractions[it] if it in contractions else it for it in
                        summary.split(" ")])  # Expand contractions
    summary = re.sub(r"'s\b", "", summary)  # Remove trailing "'s"s
    summary = re.sub(r"[^a-zA-z]", " ", summary)
    summary = summary.lower()

    tokens = summary.split()

    result = " ".join([it for it in tokens if len(it) > 1])
    return result


def prepare_dataset(datafile, max_len_text, max_len_sum, nrows=None):
    data = pd.read_csv(datafile, nrows=nrows)

    data.drop_duplicates(subset=["Text"], inplace=True)
    data.dropna(inplace=True)

    cleaned_text = []
    for text in data["Text"]:
        cleaned_text.append(clean_text(text))

    cleaned_summary = []
    for summary in data["Summary"]:
        cleaned_summary.append(clean_summary(summary))

    data["cleaned_text"] = cleaned_text
    data["cleaned_summary"] = cleaned_summary

    # Drop any rows with empty values
    data["cleaned_summary"].replace("", np.nan, inplace=True)
    data.dropna(inplace=True)

    # Mark start and end of each summary
    data["cleaned_summary"] = data["cleaned_summary"].apply(lambda x: f"__START__ {x} __END__")
 
    text_word_count = []
    for it in data["cleaned_text"]:
        text_word_count.append(len(it.split()))

    summary_word_count = []
    for it in data["cleaned_summary"]:
        summary_word_count.append(len(it.split()))

    length_df = pd.DataFrame({"text": text_word_count, "summary": summary_word_count})

    x_train, x_val, y_train, y_val = \
        train_test_split(data["cleaned_text"], data["cleaned_summary"],
                         test_size=.1, random_state=0, shuffle=True)

    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_train))

    x_train = x_tokenizer.texts_to_sequences(x_train)
    x_val = x_tokenizer.texts_to_sequences(x_val)
    x_train = pad_sequences(x_train, maxlen=max_len_text, padding="post")
    x_val = pad_sequences(x_val, maxlen=max_len_text, padding="post")

    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_train))

    y_train = y_tokenizer.texts_to_sequences(y_train)
    y_val = y_tokenizer.texts_to_sequences(y_val)
    y_train = pad_sequences(y_train, maxlen=max_len_sum, padding="post")
    y_val = pad_sequences(y_val, maxlen=max_len_sum, padding="post")

    return (x_train, y_train, x_val, y_val), (x_tokenizer, y_tokenizer)

# END OF preprocess.py
```

Deep learning model
`nodetails_model.py`:
```python
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


def define_model(x_tokenizer, y_tokenizer, max_len_text, max_len_sum, latent_dim=500):
    x_voc_size = len(x_tokenizer.word_index) + 1
    y_voc_size = len(y_tokenizer.word_index) + 1

    tf.keras.backend.clear_session()

    enc_input = layers.Input(shape=(max_len_text,))
    enc_embedding = layers.Embedding(x_voc_size, latent_dim, trainable=True, input_shape=(max_len_text,))(enc_input)

    enc_lstm1 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    enc_lstm1_out, _, _ = enc_lstm1(enc_embedding)

    enc_lstm2 = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    enc_lstm2_out, state_h2, state_c2 = enc_lstm2(enc_lstm1_out)

    enc_lstm3 = layers.LSTM(latent_dim, return_state=True, return_sequences=True)
    enc_output, state_h, state_c = enc_lstm3(enc_lstm2_out)

    dec_input = layers.Input(shape=(None,))
    dec_embedding = layers.Embedding(y_voc_size, latent_dim, trainable=True)
    dec_embedding_out = dec_embedding(dec_input)

    # Decoder LSTM is initialized with encoder states
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


def train_model(model, training_data, validation_data, batch_size=512):
    (x_train, y_train), (x_val, y_val) = training_data, validation_data

    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1)

    model.fit([x_train, y_train[:, :-1]],
              y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:],
              epochs=50, callbacks=[es], batch_size=batch_size,
              validation_data=([x_val, y_val[:, :-1]],
                               y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))
    
    return model


def prep_for_inference(model, model_params, latent_dim=500):
    (x_tokenizer, y_tokenizer,
     max_len_text, max_len_sum, enc_input, enc_output, state_h, state_c,
     dec_input, dec_output, dec_embedding, dec_lstm, dec_dense, attn) = model_params

    # Encoder inference
    encoder_model = Model(inputs=enc_input, outputs=[enc_output, state_h, state_c])

    # Decoder inference
    # Keep track of states from the previous time step
    decoder_state_input_h = layers.Input(shape=(latent_dim,))
    decoder_state_input_c = layers.Input(shape=(latent_dim,))
    decoder_hidden_state_input = layers.Input(shape=(max_len_text, latent_dim))

    # Get the embeddings of the decoder sequence
    dec_emb2 = dec_embedding(dec_input)

    # Set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = dec_lstm(dec_emb2,
                                                    initial_state=[decoder_state_input_h, decoder_state_input_c])

    # Attention inference
    attn_out_inf, attn_states_inf = attn([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = layers.Concatenate(axis=-1, name="concat")([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
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


def _decode_sequence(input_seq, model_params):
    (encoder_model, decoder_model,
     reverse_target_word_index, reverse_source_word_index, target_word_index,
     max_len_text, max_len_sum) = model_params

    # Encode the input as state vectors.
    enc_out, enc_h, enc_c = encoder_model.predict(input_seq)
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
        if sampled_token != "end":
            decoded_sentence += " " + sampled_token

            # Exit condition: either hit max length or find stop word.
        if sampled_token == "end" or len(decoded_sentence.split()) >= (max_len_sum - 1):
            _done = True

        # Update the target sequence and internal states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        enc_h, enc_c = h, c

    return decoded_sentence


def test_validation_set(x_val, y_val, model_params, item_range=None):
    (encoder_model, decoder_model,
     reverse_target_word_index, reverse_source_word_index, target_word_index,
     max_len_text, max_len_sum) = model_params

    def decode_seq(it):
        return _decode_sequence(it.reshape(1, max_len_text), model_params)

    if item_range:
        range_lo = max(0, min(item_range[0], len(x_val)))
        range_hi = min(item_range[1], len(x_val))

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


def save_nodetails_model(model_params, save_location):
    (encoder_model, decoder_model,
     reverse_target_word_index, reverse_source_word_index, target_word_index,
     max_len_text, max_len_sum) = model_params

    encoder_model.save(f"{save_location}/encoder")
    decoder_model.save(f"{save_location}/decoder")

    params = (reverse_target_word_index, reverse_source_word_index, target_word_index,
              max_len_text, max_len_sum)

    with open(f"{save_location}/parameters.pkl", "wb") as fp:
        pickle.dump(params, fp)


def load_nodetails_model(save_location):
    encoder_model = keras_load_model(f"{save_location}/encoder",
                                     custom_objects={"Attention": Attention},
                                     compile=False)
    decoder_model = keras_load_model(f"{save_location}/decoder",
                                     custom_objects={"Attention": Attention},
                                     compile=False)

    with open(f"{save_location}/parameters.pkl", "rb") as fp:
        (reverse_target_word_index, reverse_source_word_index, target_word_index,
         max_len_text, max_len_sum) = pickle.load(fp)

    return (encoder_model, decoder_model,
            reverse_target_word_index, reverse_source_word_index, target_word_index,
            max_len_text, max_len_sum)

# END OF nodetails_model.py
```

Attention layer implementation
`attention_layer.py`:
```python
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class Attention(Layer):
    """
    This class implements Bahdanau attention.
    There are three sets of weights introduced W_a, U_a, and V_a
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(Attention, self).build(input_shape)

    def call(self, inputs, verbose=False):
        """ inputs: [encoder_output_sequence, decoder_output_sequence] """

        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
                tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))]

# END OF attention_layer.py
```

End.