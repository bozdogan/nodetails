# NoDetails: Essence of the Text

It is a text summarization tool that utilizes Deep Learning.

See [Thesis Report](./Midterm%20Report.md) for more information.


## Simple Usage

Getting an extractive summarization:

```python
import nodetails.ndext

# Getting summary from a URL
extractive_url = nodetails.ndext.get_summary_from_url("https://en.wikipedia.org/wiki/Citation_needed")
    print(extractive_summary.summary)

# Getting summary from a string
article = "..."
extractive_str = nodetails.ndext.get_summary(article, length=7)  # You can specify length of the result in sentences.
```

Abstractive summarization requires a deep learning model to be loaded. 
```python
import os.path as osp
import nodetails.ndabs
import nodetails.ndext

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
```

A model can be trained with `train_model` utility as follows (150 is input length, 20 is output length):  
```
python trainmodel.py news_summary_more.csv 150 20 --save-dir {DATA_DIR}/_models --name news_summary --sum-col headlines --batch-size 64 --nrows 10
```

The models we trained are stored at [our Drive folder](https://drive.google.com/drive/folders/1i4Ax8-OiEyFLTS0vdKFiS-xmHMQdkXzR?usp=sharing).

More detailed usage is demonstrated in `usage.py`.