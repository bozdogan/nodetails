from nodetails.metrics.rouge import Rouge

def get_rogue_f1_score(hypothesis: str, reference: str):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    f1_scores = {rogue_n: scores[0][rogue_n]["f"] for rogue_n in scores[0]}

    return f1_scores
