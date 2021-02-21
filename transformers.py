from abc import ABC, abstractmethod
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
        
        # Uncomment for pandas version:
        # return [np.mean([len(token) for token in data.split()])]
        
        # For untokenized text
        try:
            return [[mean([len(token) for token in doc.split()])] for doc in data]
        # For tokenized text
        except AttributeError:
            return [[mean([len(token) for token in doc])] for doc in data]