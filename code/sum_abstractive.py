#!/usr/bin/env python3
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Uncomment and modify this line according to your hardware setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails.abstractive import *
from nodetails.util import prepare_dataset


""" No Details: Essence of the text """
if __name__ == "__main__":
    DATA_DIR = "../data"
    MODEL_SAVE_DIR = "../data/_models"

    DATASET_NAME = "food_reviews"
    input_file = f"{DATA_DIR}/food_reviews/Reviews.csv"
    INPUT_SIZE = 100000  # NOTE(bora): `None` to import the entire dataset

    MAX_LEN_TEXT = 80
    MAX_LEN_SUM = 10
    MODEL_NAME = f"nodetails--{DATASET_NAME}--{MAX_LEN_TEXT}-{MAX_LEN_SUM}--{INPUT_SIZE}"

    (x_train, y_train, x_val, y_val), (x_tokenizer, y_tokenizer) = \
            prepare_dataset(input_file,
                            max_len_text=MAX_LEN_TEXT,
                            max_len_sum=MAX_LEN_SUM,
                            nrows=INPUT_SIZE,
                            verbose=True,
                            show_histogram=False)

    # NOTE(bora): Deep learning part

    if 0:
        model_params = define_model(x_tokenizer, y_tokenizer,
                                         max_len_text=MAX_LEN_TEXT,
                                         max_len_sum=MAX_LEN_SUM)

        model = train_model(model_params,
                            (x_train, y_train), (x_val, y_val),
                            batch_size=128,
                            show_graph=True)

        model.summary()

        infr_params = prep_model_for_inference(model_params)
        #save_nodetails_model(infr_params, f"{MODEL_SAVE_DIR}/{MODEL_NAME}.model", debug_output=True)
    else:
        infr_params = load(f"{MODEL_SAVE_DIR}/{MODEL_NAME}.model", debug_output=True)

    print("LEN X TRAIN", len(x_train))
    print("LEN X VAL", len(x_val))

    test_validation_set(x_val, y_val, infr_params,
                        item_range=(42, 52),
                        debug_output=False)

    print("Done.")

# END OF sum_abstractive.py
