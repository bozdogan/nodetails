import os.path as osp
from tensorflow.python.keras import backend as K
from nodetails._types import BaseEngine, ExtractiveSummary, NDSummary
from nodetails import ndabs, ndext


class AbstractiveEngine(BaseEngine):
    """Engine for pure abstractive summarization
        
    Abstractive engine uses an abstractive model to summarize 
    short texts.

    ``model_dir`` and ``model_name`` parameters
    must be provided either on initialization or via modifying
    ``config`` field before calling ``load()`` method.
    """

    def __init__(self, name, **config):
        super().__init__(name, config)
        self.model = None
        self.ready = False

    def load(self):
        model_dir = self.config["model_dir"] or "."
        model_name = self.config["model_name"]
        if model_name:
            self.model = ndabs.load_model(osp.join(model_dir, model_name))
            self.ready = True

    def unload(self):
        del self.model
        K.clear_session()
        self.ready = False

    def summarize(self, text_body, **kwargs):
        if self.ready:
            return ndabs.make_inference(self.model, text_body)


class ExtractiveEngine(BaseEngine):
    """Engine for frequency analysis based extractive summarization
        
    Extractive engine that uses a word frequency analysis based
    approach. Desired summary length in sentences can be set
    via ``config["lenght"]`` variable.
    """

    def __init__(self, name, **config):
        super().__init__(name, config)
    
    def load(self):
        pass

    def unload(self):
        pass

    def summarize(self, text_body, **kwargs):
        length = kwargs.get("length", None) or self.config["length"]
        preset = self.config["preset"] or "wikipedia"
        extsum = ndext.get_summary(text_body, length=length, preset=preset)

        if kwargs.get("keep_references", False):
            return extsum
        else:
            return extsum.summary


class ExtractiveDLEngine(BaseEngine):
    """Engine for deep-learned extractive summarization
        
    Extractive engine uses a deep neural network to summarize 
    longer texts.

    ``model_dir`` and ``model_name`` parameters
    must be provided either on initialization or via modifying
    ``config`` field before calling ``load()`` method.

    NOT YET FULLY IMPLEMENTED
    """

    def __init__(self, name, **config):
        super().__init__(name, config)
        self.model = None
        self.ready = False

        raise NotImplementedError()

    def load(self):
        model_dir = self.config["model_dir"] or "."
        model_name = self.config["model_name"]
        if model_name:
            self.model = 0xDEADBEEF  #ndabs.load_model(osp.join(model_dir, model_name))
            self.ready = True

    def unload(self):
        del self.model
        K.clear_session()
        self.ready = False

    def summarize(self, text_body, **kwargs):
        if self.ready:
            return 0xDEADBEEF  #ndabs.make_inference(self.model, text_body)


class IntegratedEngine(BaseEngine):
    """An integrated engine for generating titles for longer texts
    
    Integrated engine uses abstractive model with frequency analysis based
    extractive methods.

    ``abs_model_dir`` and ``abs_model_name`` parameters for the
    abstractive model must be provided either on initialization
    or via modifying ``config`` field before calling ``load()`` method.

    Intermediate extractive summary length can be set via ``config["lenght"]``
    variable.
    """

    def __init__(self, name, **config):
        super().__init__(name, config)
        self.absengine = AbstractiveEngine(name + "--abs")
        self.extengine = ExtractiveEngine(name + "--ext")

    def load(self):
        self.absengine.load()
        self.extengine.load()

    def unload(self):
        self.absengine.unload()
        self.extengine.unload()

    def summarize(self, text_body, **kwargs):
        if self.ready:
            # if text_body is "too long":
            #     use two methods together
            # elif text_body is "about one or two paragraphs":
            #     just use abstractive methods
            # else: if text_body is "even shorter":
            #     return it as is

            extsum = self.extengine.summarize(text_body, keep_references=True)
            assert isinstance(extsum, ExtractiveSummary)

            abssum = self.absengine.summarize(extsum)
            
            return NDSummary(abssum, extsum.summary,
                             extsum.reference, extsum.sentences)


def defined_engines():
    excluded = ["BaseEngine"]
    available = [it for it in globals()
                 if it.endswith("Engine") and it not in excluded]

    return [globals()[it] for it in available]


if __name__ == '__main__':
    eng = AbstractiveEngine("isim", model_dir="../../data/_models",
                            model_name="nodetails--food_reviews--80-10--None.model")
    # eng = ExtractiveEngine("isim")
    text = ("My main use for almond, soy, or rice milk is to use in coffee "
            "or tea.  The best so far is Silk soymilk original but this Silk "
            "almond milk is very nearly as good.  Actually as far as taste goes "
            "I'd say the almond milk might be best but for using in coffee the "
            "soy edges out the almond for creaminess. I totally enjoy them "
            "both and have been buying the almond milk from Amazon at a very "
            "fair price.  But now it's off the Subscribe and Save program "
            "so the cost will go up. I intend to continue buying either "
            "Silk almond or Silk soy milk because they are the best for me.")

    print(eng.summarize(text))
    eng.unload()

    eng.load()
    print(eng.summarize(text))
    a = 17
