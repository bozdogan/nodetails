import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import nodetails
from nodetails import abs, prep, util

nodetails.enable_vram_growth()


if __name__ == "__main__":
    DATA_DIR = "../data"
    model_save_dir = f"{DATA_DIR}/_models"
    
    input_file = f"{DATA_DIR}/wikihow_articles/wikihowAll.csv"
    input_size = 1000  # NOTE(bora): `None` to import the entire dataset
    
    dataset_name = "wikihow"
    x_len = 120
    y_len = 15
    model_name = f"nodetails--{dataset_name}--{x_len}-{y_len}--{input_size}"
    column_names = {"text": "text", "title": "sum"}

    nodetails.set_debug(True)

    dataset = util.cached_read_dataset_csv(input_file,
                                           nrows=input_size,
                                           renaming_map=column_names)
    training_data, lexicon = prep.prepare_training_set(dataset,
                                                       x_len, y_len,
                                                       split=.1)

    # NOTE(bora): Deep learning part
    abs_model = abs.create_models(lexicon, latent_dim=500)

    if nodetails.is_debug():
        abs_model.training.summary()
        abs_model.encoder.summary()
        abs_model.decoder.summary()

    if 1:
        abs.train_model(abs_model, training_data,
                        batch_size=128,
                        show_graph=nodetails.is_debug())
        abs.save_model(abs_model, f"{model_save_dir}/{model_name}.model")

    abs_model_reloaded = abs.load_model(f"{model_save_dir}/{model_name}.model")

    abs.test_validation_set(
        abs_model_reloaded,
        training_data.x_val, training_data.y_val,
        lexicon, item_range=(0, 10))

    print("Done.")

# END OF test_abstractive_sum.py
