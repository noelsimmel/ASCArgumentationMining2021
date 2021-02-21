from numpy import mean
from sklearn.base import BaseEstimator, TransformerMixin

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    ""

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        # Uncomment for pandas version:
        # data = data["text"].apply(self.average_word_length)
        
        return self.average_word_length(data)
    
    def average_word_length(self, data):
        ""
        
        # Uncomment for pandas version:
        # return [np.mean([len(token) for token in data.split()])]
        
        # For untokenized text
        try:
            return [[mean([len(token) for token in doc.split()])] for doc in data]
        # For tokenized text
        except AttributeError:
            return [[mean([len(token) for token in doc])] for doc in data]