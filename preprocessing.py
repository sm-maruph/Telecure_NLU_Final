import re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("bn")

def bn_preprocess(text):
    text = normalizer.normalize(text)
    text = text.lower()
    text = re.sub(r"[^\u0980-\u09FF\s]", "", text)
    return text

def bn_tokenizer(text):
    # import indic_tokenize here as well
    from indicnlp.tokenize import indic_tokenize
    return indic_tokenize.trivial_tokenize(text)
