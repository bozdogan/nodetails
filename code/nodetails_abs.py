if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Uncomment and modify this line according to your hardware setup

from nodetails import abstractive, sequence_model, util


if __name__ == "__main__":
    DATA_DIR = "../data"
    model_save_dir = f"{DATA_DIR}/_models"

    input_file = f"{DATA_DIR}/food_reviews/Reviews.csv"
    input_size = None  # NOTE(bora): `None` to import the entire dataset
    
    dataset_name = "food_reviews"
    max_len_text = 80
    max_len_sum = 10
    model_name = f"nodetails--{dataset_name}--{max_len_text}-{max_len_sum}--{input_size}"

    training_data = util.preprocess_dataset(input_file, nrows=input_size, verbose=True)
    (_, _, x_val, y_val, _, _) = util.prepare_for_training(training_data, max_len_text, max_len_sum, verbose=True)

    # NOTE(bora): Deep learning part
    if 1:
        absmodel = abstractive.create_model(
            training_data, max_len_text, max_len_sum,
            latent_dim=500, batch_size=128)

        abstractive.save(absmodel, model_save_dir, model_name)
    else:
        absmodel = abstractive.load(model_save_dir, model_name)

    sequence_model.test_validation_set(x_val, y_val, absmodel, item_range=(0, 10))

    print("Done.")

# END OF nodetails_abs.py
