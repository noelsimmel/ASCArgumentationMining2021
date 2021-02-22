from abc import ABC, abstractmethod
from nltk import ne_chunk, pos_tag
from numpy import mean
from sklearn.base import BaseEstimator, TransformerMixin

class BaseCustomTransformer(ABC, BaseEstimator, TransformerMixin):
    ""

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        # Uncomment for pandas version:
        # data = data["text"].apply(self.make_feature)
        
        return self.make_feature(data)
    
    @abstractmethod
    def make_feature(self, data):
        ""
        
        return []

class AverageWordLengthExtractor(BaseCustomTransformer):
    ""

    def make_feature(self, data):
        ""
        # print(data)
        # Uncomment for pandas version:
        # return [np.mean([len(token) for token in data.split()])]
        
        # For untokenized text
        try:
            return [[mean([len(token) for token in doc.split()])] for doc in data]
        # For tokenized text
        except AttributeError:
            return [[mean([len(token) for token in doc])] for doc in data]
        
class NamedEntityExtractor(BaseCustomTransformer):
    ""
    
    def make_feature(self, data):
        ""
        
        # POS tagging
        # Exclude masks as they confuse the parser
        masks = {"MENTION", "HASHTAG", "BIBLEQUOTE", "QURANQUOTE"}
        data = [[t for t in doc if t not in masks] for doc in data]
        tagged = [pos_tag(doc) for doc in data]
        # NER using nltk
        chunked = [ne_chunk(doc) for doc in tagged]
        # Quick and dirty counting named entities:
        # Flatten list (= syntax tree) and count leftover tuples, 
        # since NEs are the innermost branches
        # Based on https://stackoverflow.com/a/952952
        flattened = [[item for sublist in tree for item in sublist] for tree in chunked]
        return [[sum(type(t) == tuple for t in doc)] for doc in flattened]
    
class AtheismPolarityExtractor(BaseCustomTransformer):
    ""
    
    def __init__(self):
        ""
        
        self.pro_atheism  = {"freethinker", "evidence", "atheist", "freethink", "evid"}
        self.anti_atheism = {"teamjesus", "holy", "lord", "holi", "amen"}
        
    def make_feature(self, data):
        ""
        
        return [[self._get_polarity(doc)] for doc in data]
    
    def _get_polarity(self, doc):
        ""
        
        s = 0
        for tok in doc:
            if tok in self.pro_atheism: s += 1
            elif tok in self.anti_atheism: s -= 1
        return s 