import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails import abstractive, extractive

if __name__ == "__main__":
    model_dir = "../data/_models"
    model_name = f"nodetails--food_reviews--80-10--None"
    absmodel = abstractive.load(model_dir, model_name, verbose=True)

    text = ""
    sum_orig = ""

    sum_pred = abstractive.make_inference(absmodel, text)
    print(sum_pred)