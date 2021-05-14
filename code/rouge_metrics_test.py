import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails import abstractive, extractive
from nodetails import evaluation

if __name__ == "__main__":
    model_dir = "../data/_models"
    model_name = f"nodetails--food_reviews--80-10--None"
    # absmodel = abstractive.load(model_dir, model_name, verbose=True)
    #
    # text = ""
    # sum_orig = ""
    #
    # sum_pred = abstractive.make_inference(absmodel, text)
    # print(sum_pred)

    hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"
    reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

    scores = evaluation.get_rogue_f1_score(hypothesis, reference)

    print(scores)