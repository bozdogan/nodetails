import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Uncomment this line to force CPU mode
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import nodetails_model as nodetails
import preprocess


# NOTE(bora): No Details: Essence of the text
if __name__ == "__main__":
    DATA_DIR = "../data/food_reviews"

    # TODO(bora): Memory hatası alıyorum sürekli, çözemedim bir türlü. Her şey çalışıyor ama tam sonucu görecekken
    # hafızam bitiyor her halde anlamadım..
    DATA_SIZE = 100000
    BATCH_SIZE = 128  # NOTE(bora): Most efficient when it is a power of 2

    if 1:
        data = pd.read_csv(f"{DATA_DIR}/Reviews.csv", nrows=DATA_SIZE)
        MAX_LEN_TEXT = 80
        MAX_LEN_SUM = 10

        (x_train, y_train, x_val, y_val), (x_tokenizer, y_tokenizer) = \
            preprocess.prepare_dataset(data, max_len_text=MAX_LEN_TEXT, max_len_sum=MAX_LEN_SUM, verbose=False)

        # NOTE(bora): Deep learning kısmı
        model, parameters = nodetails.define_model(x_tokenizer, y_tokenizer, max_len_text=MAX_LEN_TEXT, max_len_sum=MAX_LEN_SUM)
        model.summary()

        nodetails.train_model(model, parameters, (x_train, y_train), (x_val, y_val),
                              batch_size=BATCH_SIZE, verbose=True)

        print("Hadi be, çalıştı mı harbiden?!")

# END OF sum_abstractive.py
