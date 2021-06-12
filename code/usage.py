"""
This module demonstrates basic usage of NoDetails API.
"""
if __name__ == "__main__":
    # NOTE(bora): This prevent errors like this:
    # tensorflow.python.framework.errors_impl.InternalError:  Blas GEMM launch failed : a.shape=(80, 500), b.shape=(500, 500), m=80, n=500, k=500
    # 	 [[{{node model_2/attention/while/body/_1/model_2/attention/while/MatMul}}]] [Op:__inference_predict_function_4374]
    from nodetails import enable_vram_growth; enable_vram_growth()

    # NOTE(bora): This is to suppress tensorflow info messages
    import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import os.path as osp
import nodetails.ndabs
import nodetails.ndext


if __name__ == "__main__":
    #
    # Extractive
    # ----------

    # Extractive summarization doesn't require training. You can directly use
    # api functions without prior preparation steps.
    extractive_summary = nodetails.ndext.get_summary_from_url("https://en.wikipedia.org/wiki/Citation_needed")
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
    # For training the model, see test_abstractive_sum.py
    model_dir = "../data/_models"
    model_name = f"nodetails--food_reviews--80-10--None.model"
    infr_params = nodetails.ndabs.load_model(osp.join(model_dir, model_name))

    text = ("My main use for almond, soy, or rice milk is to use in coffee "
            "or tea.  The best so far is Silk soymilk original but this Silk "
            "almond milk is very nearly as good.  Actually as far as taste goes "
            "I'd say the almond milk might be best but for using in coffee the "
            "soy edges out the almond for creaminess. I totally enjoy them "
            "both and have been buying the almond milk from Amazon at a very "
            "fair price.  But now it's off the Subscribe and Save program "
            "so the cost will go up. I intend to continue buying either "
            "Silk almond or Silk soy milk because they are the best for me.")
    abstractive_summary = nodetails.ndabs.make_inference(infr_params, text)
    print(abstractive_summary)

    #
    # Integrated
    # ----------

    # We intend to combine these two methods into a more flexible solution:
    extsum = nodetails.ndext.get_summary_from_url("https://en.wikipedia.org/wiki/Citation_needed").summary
    integrated_example = nodetails.ndabs.make_inference(infr_params, extsum)

    print(extsum)
    print(integrated_example)

# END OF api_usage.py
