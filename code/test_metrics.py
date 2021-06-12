import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import nodetails as nd
from nodetails import ndabs, util, prep
from nodetails import metrics


def score_validation_set(abs_model, x_val, y_val, lexicon,
                         item_range=(0, 1), depug_output=False):
    x_tkn, y_tkn, x_len, y_len = lexicon

    def decode_validation_seq(it):
        result = ndabs.decode_sequence(it.reshape(1, x_len), abs_model)
        assert result, f"Empty result of type {type(result)} at item #{it}"
        return result

    rouge1 = []
    rouge2 = []
    rougeL = []
    for i in range(*item_range):
        sum_orig = (ndabs.seq2text(y_val[i], y_tkn).replace("<start>", "")
                    .replace("<end>", "")
                    .strip())
        sum_pred = decode_validation_seq(x_val[i])
        scores = metrics.get_rogue_f1_score(sum_pred, sum_orig)
        rouge1.append(scores["rouge-1"])
        rouge2.append(scores["rouge-2"])
        rougeL.append(scores["rouge-l"])
        if depug_output:
            print("\nReview #%s: " % i)
            print("rouge-1 : %.7f" % scores["rouge-1"])
            print("rouge-2 : %.7f" % scores["rouge-2"])
            print("rouge-l : %.7f" % scores["rouge-l"])

    def mean(L):
        return sum(L)/len(L)

    return mean(rouge1), mean(rouge2), mean(rougeL)


if __name__ == "__main__":
    nd.enable_vram_growth()
    nd.set_debug(True)

    dataset_name = "wikihow"
    x_len = 120
    y_len = 15
    column_names = {"text": "text", "title": "sum"}

    dataset = util.cached_read_dataset_csv("../data/wikihow_articles/wikihowAll.csv",
                                           nrows=1000,
                                           renaming_map=column_names)
    training_data, lexicon = prep.prepare_training_set(dataset,
                                                       x_len, y_len,
                                                       split=.1)

    abs_model = ndabs.load_model("../data/_models/nodetails--wikihow--120-15--1000.model")
    scores = score_validation_set(
        abs_model, training_data.x_val, training_data.y_val, lexicon,
        item_range=(0, 99))

    print(scores)
    print("Done")

# END OF test_metrics.py
