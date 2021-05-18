if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Modify this line according to your hardware setup

from nodetails import abstractive, prep
from nodetails.nn import sequence_model

if __name__ == "__main__":
    DATA_DIR = "../data"
    model_save_dir = f"{DATA_DIR}/_models"
    input_file = f"{DATA_DIR}/food_reviews/Reviews.csv"
    input_size = 10000  # NOTE(bora): `None` to import the entire dataset
    
    dataset_name = "food_reviews"
    max_len_text = 80
    max_len_sum = 10

    data_str = prep.preprocess_dataset(input_file, nrows=input_size)
    data_seq = prep.split_train_test(data_str, max_len_text, max_len_sum)

    # NOTE(bora): Deep learning part
    if "train":
        import nodetails_train
        model_savepath = nodetails_train.train_and_save_model(
            data_seq, model_save_dir, max_len_text, max_len_sum, latent_dim=500,
            name_prefix=dataset_name, name_postfix=input_size)

    absmodel = abstractive.load(model_savepath)

    (_, _, x_val, y_val, _, _) = data_seq
    sequence_model.test_validation_set(x_val, y_val, absmodel, item_range=(0, 10))

    print("Done.")

# END OF test_nodetails_abstractive.py
