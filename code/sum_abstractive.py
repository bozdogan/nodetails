#!/usr/bin/env python3
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Uncomment this line to force CPU mode
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails_model import *
import preprocess


# NOTE(bora): No Details: Essence of the text
if __name__ == "__main__":
    DATA_DIR = "../data/food_reviews"

    DATA_SIZE = 100000
    BATCH_SIZE = 128  # NOTE(bora): Most efficient when it is a power of 2

    data_file = f"{DATA_DIR}/Reviews.csv"
    MAX_LEN_TEXT = 80
    MAX_LEN_SUM = 10

    (x_train, y_train, x_val, y_val), (x_tokenizer, y_tokenizer) = preprocess.prepare_dataset(
        data_file, nrows=DATA_SIZE,
        max_len_text=MAX_LEN_TEXT,
        max_len_sum=MAX_LEN_SUM,
        verbose=True,
        show_histogram=True)

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

    test_validation_set(x_val, y_val, model_params,
                        item_range=(0, 10),
                        debug_output=False)

    print("Done.")

# END OF sum_abstractive.py
