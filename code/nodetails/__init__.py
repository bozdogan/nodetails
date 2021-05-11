from collections import namedtuple


DatasetResult = namedtuple(
    "DatasetResult",
    ["x_train", "y_train", "x_val", "y_val", 
     "x_tokenizer", "y_tokenizer"])

ModelParameters = namedtuple(
    "ModelParameters",
    ["x_tokenizer", "y_tokenizer", "max_len_text", "max_len_sum", "latent_dim",
     "enc_input", "enc_output", "state_h", "state_c",
     "dec_input", "dec_output", "dec_embedding", "dec_lstm", "dec_dense", "attn"])

InferenceParameters = namedtuple(
    "InferenceParameters",
    ["encoder_model", "decoder_model",
     "y_index_word", "x_index_word", "y_word_index",
     "max_len_text", "max_len_sum"])

ExtractiveSummary = namedtuple(
    "ExtractiveSummary",
    ["summary", "reference", "sentences", "paragraphs"])

# END OF __init__.py