#!/usr/bin/env python3
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Uncomment and modify this line according to your hardware setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails_model import *
import preprocess


# NOTE(bora): No Details: Essence of the text
if __name__ == "__main__":
    DATA_DIR = "../data/food_reviews"
    MODEL_DIR = "../models"

    DATA_SIZE = 1000
    BATCH_SIZE = 128  # NOTE(bora): Most efficient when it is a power of 2

    data_file = f"{DATA_DIR}/Reviews.csv"
    MAX_LEN_TEXT = 80
    MAX_LEN_SUM = 10
    MODEL_NAME = f"nodetails--{MAX_LEN_TEXT}-{MAX_LEN_SUM}-{BATCH_SIZE}"

    (x_train, y_train, x_val, y_val), (x_tokenizer, y_tokenizer) = preprocess.prepare_dataset(
        data_file, nrows=DATA_SIZE,
        max_len_text=MAX_LEN_TEXT,
        max_len_sum=MAX_LEN_SUM,
        verbose=True,
        show_histogram=False)

    print("LEN X TRAIN", len(x_train))
    print("LEN X VAL", len(x_val))

    # NOTE(bora): Deep learning part
    model, parameters = define_model(x_tokenizer, y_tokenizer,
                                     max_len_text=MAX_LEN_TEXT,
                                     max_len_sum=MAX_LEN_SUM)
    model.summary()

    model_params = train_model(model, parameters,
                               (x_train, y_train), (x_val, y_val),
                               batch_size=BATCH_SIZE,
                               show_graph=True)
    # NOTE(bora): model_params_prime is not the same with model_params.
    # The functions successfully save max lengths and tokens and whatnot but
    # cannot save and load the actual models. It is something to do with the
    # backend of Attention layer I suppose.
    save_nodetails_model(model_params, f"{MODEL_DIR}/{MODEL_NAME}-{DATA_SIZE}.model", debug_output=True)
    model_params_prime = load_nodetails_model(f"{MODEL_DIR}/{MODEL_NAME}-{DATA_SIZE}.model", debug_output=True)

    test_validation_set(x_val, y_val, model_params_prime,
                        item_range=(0, 10),
                        debug_output=False)

    print("Done.")

# END OF sum_abstractive.py
