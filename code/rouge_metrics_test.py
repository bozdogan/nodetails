import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails import abstractive, ext
from nodetails import evaluation

if __name__ == "__main__":
    # model_dir = "../data/_models"
    # model_name = f"nodetails--food_reviews--80-10--None"
    # absmodel = abstractive.load(model_dir, model_name, verbose=True)

    # text = ("My main use for almond, soy, or rice milk is to use in coffee "
    #         "or tea.  The best so far is Silk soymilk original but this Silk "
    #         "almond milk is very nearly as good.  Actually as far as taste goes "
    #         "I'd say the almond milk might be best but for using in coffee the "
    #         "soy edges out the almond for creaminess. I totally enjoy them "
    #         "both and have been buying the almond milk from Amazon at a very "
    #         "fair price.  But now it's off the Subscribe and Save program "
    #         "so the cost will go up. I intend to continue buying either "
    #         "Silk almond or Silk soy milk because they are the best for me.")

    # sum_pred = abstractive.make_inference(absmodel, text)

    sum_orig = ("My main use for almond soy, or rice milk is to use in coffee "
                "or tea.  The best so far is Silk soymilk original but this Silk "
                "almond milk is very nearly as good."
                "Silk almond or Silk soy milk because they are the best for me.")
    sum_pred = ("My main use almond for soy, or rice milk is to use in coffee "
                "or tea.  The best so far is Silk soymilk original but this Silk "
                "almond milk is very nearly as good.")

    scores = evaluation.get_rogue_f1_score(sum_pred, sum_orig)
    print(scores)

# END OF rogue_metrics_text.py