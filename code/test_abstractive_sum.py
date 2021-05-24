import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


import nodetails
from nodetails import abs, prep, util

import tensorflow.keras.backend as K; K.clear_session()
nodetails.enable_vram_growth()
nodetails._DEBUG = True


if __name__ == "__main__":
    DATA_DIR = "../data"
    model_save_dir = f"{DATA_DIR}/_models"

    input_file = f"{DATA_DIR}/wikihow_articles/wikihowAll.csv"
    input_size = 10000  # NOTE(bora): `None` to import the entire dataset
    
    dataset_name = "wikihow"
    max_len_text = 80
    max_len_sum = 10
    model_name = f"nodetails--{dataset_name}--{max_len_text}-{max_len_sum}--{input_size}"

    dataset = util.cached_read_dataset_csv(input_file, nrows=input_size,
                                           renaming_map={"text": "text", "title": "sum"})
    training_data, lexicon = prep.prepare_training_set(
        dataset, x_len=max_len_text, y_len=max_len_sum, split=.1)

    # NOTE(bora): Deep learning part
    training_model, infr_model = abs.create_models(lexicon, latent_dim=500)

    training_model.model.summary()
    #infr_model.encoder.summary()
    #infr_model.decoder.summary()

    if 0:
        abs.train_model(training_model, training_data,
                        batch_size=128,
                        show_graph=True)

        # NOTE(bora): Save the model to disk
        abs.save_model(infr_model, f"{model_save_dir}/{model_name}.model")
    # NOTE(bora): Load the model from disk
    infr_model_reloaded = abs.load_model(f"{model_save_dir}/{model_name}.model")

    abs.test_validation_set(
        infr_model_reloaded,
        training_data.x_val, training_data.y_val,
        lexicon, item_range=(0, 10))

    print("Done.")

# END OF test_abstractive_sum.py
