"""
This module demonstrates basic usage of NoDetails API.
"""

import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails import abstractive, extractive


if __name__ == "__main__":
    #
    # Extractive
    # ----------

    # Extractive summarization doesn't require training. You can directly use
    # api functions without prior preparation steps.
    extractive_summary = extractive.get_summary_from_url("https://en.wikipedia.org/wiki/Citation_needed")
    print(extractive_summary.summary)

    # Extractive summary is some subset of sentences from the original article.
    # We keep their reference to the article so they can be located in the article.
    ext_summary, reference, sentences, paragraphs = extractive_summary
    print(reference)
    # This feature can be used as hyperlinks to the sentences in the source material.

    #
    # Abstractive
    # -----------

    # Abstractive summary, on the other hand, relies on capabilities of the neural
    # network we call NoDetails Abstractive Model. It is an Encoder/Decoder model
    # with Attention based on Recurrent Neural Networks. We need to load a trained
    # model before making an inference.
    #
    # See nodetails.abs_py for training code
    model_dir = "../data/_models"
    model_name = f"nodetails--food_reviews--80-10--None"
    infr_params = abstractive.load(model_dir, model_name, verbose=True)

    text = ("My main use for almond, soy, or rice milk is to use in coffee "
            "or tea.  The best so far is Silk soymilk original but this Silk "
            "almond milk is very nearly as good.  Actually as far as taste goes "
            "I'd say the almond milk might be best but for using in coffee the "
            "soy edges out the almond for creaminess. I totally enjoy them "
            "both and have been buying the almond milk from Amazon at a very "
            "fair price.  But now it's off the Subscribe and Save program "
            "so the cost will go up. I intend to continue buying either "
            "Silk almond or Silk soy milk because they are the best for me.")
    abstractive_summary = abstractive.make_inference(infr_params, text)
    print(abstractive_summary)

    #
    # Integrated
    # ----------

    # We intend to combine these two methods into a more flexible solution:
    extsum = extractive.get_summary_from_url("https://en.wikipedia.org/wiki/Citation_needed").summary
    integrated_example = abstractive.make_inference(infr_params, extsum)

    print(integrated_example)
    print(extsum)

# END OF api_usage.py
