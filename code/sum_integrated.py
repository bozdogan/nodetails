if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # NOTE(bora): Uncomment and modify this line according to your hardware setup

from nodetails import abstractive
from nodetails.integration import IntegratedSummarizer


""" No Details: Essence of the text """
if __name__ == "__main__":
    model_dir = "../data/_models"
    model_name = f"nodetails--food_reviews--80-10--None"
    # NOTE(bora): Assumed the model is already trained.
    infr_params = abstractive.load(f"{model_dir}/{model_name}.model", debug_output=True)
    
    summarizer = IntegratedSummarizer(infr_params)
    summary = summarizer.from_wikipedia_article("https://en.wikipedia.org/wiki/Amazon_Fresh")

# END OF integrated_sum.py
