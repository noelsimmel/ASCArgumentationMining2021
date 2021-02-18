from numpy import mean
from sklearn.base import BaseEstimator, TransformerMixin

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    ""

    def fit(self, data, y=None):
        ""
        
        return self

    def transform(self, data, y=None):
        ""
        
        # data = data[1].apply(self.average_word_length)
        return self.average_word_length(data)
    
    def average_word_length(self, tweets):
        ""
        
        return [[mean([len(token) for token in t.split()])] for t in tweets]
        # return [np.mean([len(token) for token in tweets.split()])]