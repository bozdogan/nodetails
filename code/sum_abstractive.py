#!/usr/bin/env python3
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Uncomment and modify this line according to your hardware setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails.model import *
from nodetails.util import prepare_dataset


# NOTE(bora): No Details: Essence of the text
if __name__ == "__main__":
    DATA_DIR = "../data"
    MODEL_SAVE_DIR = "../data/_models"

    DATASET_NAME = "food_reviews"
    input_file = f"{DATA_DIR}/food_reviews/Reviews.csv"
    INPUT_SIZE = None  # NOTE(bora): Import entire dataset if set to None

    MAX_LEN_TEXT = 80
    MAX_LEN_SUM = 10
    MODEL_NAME = f"nodetails--{DATASET_NAME}--{MAX_LEN_TEXT}-{MAX_LEN_SUM}--{INPUT_SIZE}"

    (x_train, y_train, x_val, y_val), (x_tokenizer, y_tokenizer) = \
            prepare_dataset(input_file, nrows=INPUT_SIZE,
                            max_len_text=MAX_LEN_TEXT,
                            max_len_sum=MAX_LEN_SUM,
                            verbose=True,
                            show_histogram=False)

    # NOTE(bora): Deep learning part
    if not "train from scratch":
        model, parameters = define_model(x_tokenizer, y_tokenizer,
                                         max_len_text=MAX_LEN_TEXT,
                                         max_len_sum=MAX_LEN_SUM)
        model.summary()

        model = train_model(model,
                            (x_train, y_train), (x_val, y_val),
                            batch_size=128,
                            show_graph=True)
        model_params = prep_for_inference(model, parameters)
        save_nodetails_model(model_params, f"{MODEL_SAVE_DIR}/{MODEL_NAME}.model", debug_output=True)
    else:
        model_params = load_nodetails_model(f"{MODEL_SAVE_DIR}/{MODEL_NAME}.model", debug_output=True)

    print("LEN X TRAIN", len(x_train))
    print("LEN X VAL", len(x_val))

    test_validation_set(x_val, y_val, model_params,
                        item_range=(42, 52),
                        debug_output=False)

    print("Done.")

# END OF sum_abstractive.py
