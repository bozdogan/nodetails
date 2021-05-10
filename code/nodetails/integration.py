from nodetails import abstractive, extractive
from nodetails.abstractive import InferenceParameters


def summary_from_wikipedia(article_url, abst_model: InferenceParameters):
    extsum = extractive.get_summary_from_url(article_url, 10, preset="wikipedia")
    abstsum = abstractive.make_inference(abst_model, extsum.summary, debug_output=True)

    return abstsum

# END OF integration.py
