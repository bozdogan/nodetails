if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Uncomment and modify this line according to your hardware setup

from nodetails import abstractive, sequence_model
from nodetails.util import prepare_dataset


if __name__ == "__main__":
    DATA_DIR = "../data"
    model_save_dir = f"{DATA_DIR}/_models"

    input_file = f"{DATA_DIR}/food_reviews/Reviews.csv"
    input_size = 1000  # NOTE(bora): `None` to import the entire dataset
    
    dataset_name = "food_reviews"
    max_len_text = 80
    max_len_sum = 10
    model_name = f"nodetails--{dataset_name}--{max_len_text}-{max_len_sum}--{input_size}"

    (x_train, y_train, x_val, y_val,
     x_tokenizer, y_tokenizer) = prepare_dataset(input_file,
                                                 max_len_text=max_len_text,
                                                 max_len_sum=max_len_sum,
                                                 nrows=input_size,
                                                 verbose=True)

    # NOTE(bora): Deep learning part
    if 0:
        model_params = sequence_model.define_model(x_tokenizer, y_tokenizer,
                                    max_len_text=max_len_text,
                                    max_len_sum=max_len_sum)

        model = sequence_model.train_model(model_params,
                            (x_train, y_train), (x_val, y_val),
                            batch_size=128,
                            show_graph=True)

        model.summary()

        infr_params = sequence_model.prep_model_for_inference(model_params)
        abstractive.save(infr_params, model_save_dir, model_name)
    else:
        infr_params = abstractive.load(model_save_dir, model_name)

    print("LEN X TRAIN", len(x_train))
    print("LEN X VAL", len(x_val))

    sequence_model.test_validation_set(x_val, y_val, infr_params, item_range=(0, 10))

    print("Done.")

# END OF nodetails_abs.py
