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

    DATA_SIZE = 100000
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

    # NOTE(bora): Deep learning part. Change below if statement to choose
    # between training or loading the model. Arayüz yazmaya üşendim bunun için :/
    if 2 + 2 == 5:
        model, parameters = define_model(x_tokenizer, y_tokenizer,
                                         max_len_text=MAX_LEN_TEXT,
                                         max_len_sum=MAX_LEN_SUM)
        model.summary()

        model = train_model(model,
                            (x_train, y_train), (x_val, y_val),
                            batch_size=BATCH_SIZE,
                            show_graph=True)
        model_params = prep_for_inference(model, parameters)
        save_nodetails_model(model_params, f"{MODEL_DIR}/{MODEL_NAME}-{DATA_SIZE}.model", debug_output=True)
    else:
        # NOTE(bora): It works fine now! Make sure you put the model to the correct path.
        model_params = load_nodetails_model(f"{MODEL_DIR}/{MODEL_NAME}-{DATA_SIZE}.model", debug_output=True)

    print("LEN X TRAIN", len(x_train))
    print("LEN X VAL", len(x_val))

    test_validation_set(x_val, y_val, model_params,
                        item_range=(42, 52),
                        debug_output=False)

    print("Done.")

# END OF sum_abstractive.py
