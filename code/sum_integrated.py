if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Uncomment and modify this line according to your hardware setup

from nodetails import abstractive
from nodetails.integration import summary_from_wikipedia


""" No Details: Essence of the text """
if __name__ == "__main__":
    model_dir = "../data/_models"
    model_name = f"nodetails--food_reviews--80-10--None"
    # NOTE(bora): Assumed the model is already trained.
    infr_params = abstractive.load(f"{model_dir}/{model_name}.model", verbose=True)

    summary = summary_from_wikipedia("https://en.wikipedia.org/wiki/Amazon_Fresh", infr_params)

# END OF integrated_sum.py
