from collections import namedtuple
import logging
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
file_handler = logging.FileHandler('logfile.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class ASCClassifier:
    def __init__(self):
        ""

        model_attributes = ['clf', 'accuracy', 'f1']
        self.model = namedtuple('Model', model_attributes)

    def train(self, train_file, target, test_model=False):
        ""

        train_data = self._read_file(train_file, target)
        test_data = pd.DataFrame()
        # Split dataset if classifier should be tested
        if test_model:
            # seed=21 shows atheism=[0,-1,1,0,0] at .head()
            train_data, test_data = self._split_dataset(train_data, target, seed=21)
        logger.info(f"Training start")
        
        # Baseline classifier: Linear SVM 
        baseline_clf = self.train_baseline_clf(train_data, test_data)
        print(baseline_clf)
        
        # Extract features
        # features = self._extract_features(train_data)
        # data = pd.merge(train_data, features, on="id")
        # logger.info(f"Training finished: df shape {data.shape}, {features.shape[1]} features")

        # TODO: Build model

        # if test_model:
        #     self.evaluate(test_data, clf)
        
    def train_baseline_clf(self, train_data, test_data):
        ""
        
        # Class label must be at column index 2
        labels = train_data.columns[2]
        train_text = train_data.text
        y_train = train_data[labels]

        # Bag of words
        vectorizer = CountVectorizer(min_df=2)
        # Or choose TF-IDF:
        # vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_text)
        
        # Linear SVM
        # Balanced class weight gave better accuracy than unbalanced (+0.035)
        clf = svm.SVC(kernel="linear", class_weight="balanced")
        clf.fit(X_train, y_train)
        logger.info(f"Created baseline classifier {clf}")
        
        return self.evaluate(clf, test_data, vectorizer)

    def predict(self):
        ""

        pass

    def evaluate(self, clf, test_data, vectorizer=None):
        ""

        # If no test data supplied, return
        if test_data.empty:
            return self.model(clf, None, None)
        
        y_test = test_data[test_data.columns[2]]
        X_test = vectorizer.transform(test_data.text)
        pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, pred)
        f1 = metrics.f1_score(y_test, pred, average="micro")
        logger.info(f"Baseline classifier {clf}: Accuracy {accuracy:.3f}, micro f1 {f1:.3f}")
        return self.model(clf, accuracy, f1)

    def _read_file(self, fn, target):
        ""

        df = pd.read_table(fn)
        if target not in df.columns:
            raise ValueError(f"Target '{target}' is not a column in the data")
        logger.info(f"Read data from {fn}: {df.shape} dataframe")
        # Drop all columns except id, text, atheism stance
        return df[["id", "text", target]]

    def _split_dataset(self, df, target, seed=None):
        ""

        # Split in train and test sets (80:20)
        train, test = train_test_split(df, test_size=0.2, random_state=seed,
                                       stratify=df[target])
        return train, test
    
    def _extract_features(self, df):
        ""

        logger.info(f"Feature extraction start: {len(df)} instances")
        pass
    

if __name__ == '__main__':
    # pd.set_option('display.max_columns', None)
    f = 'data/corpus.txt'
    clf = ASCClassifier()
    clf.train(f, "atheism", test_model=True)