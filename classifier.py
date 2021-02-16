from collections import namedtuple
import logging
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score, train_test_split

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(funcName)s:%(message)s")
file_handler = logging.FileHandler("logfile.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class ASCClassifier:
    def __init__(self):
        ""

        model_attributes = ["clf", "accuracy", "micro_f1", "cv_mean", "cv_std"]
        self.model = namedtuple("Model", model_attributes)

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
        label = train_data.columns[2]
        y_train = y_all = train_data[label]
        
        # Vectorize text for classification (= convert to numbers)
        # Bag of words
        vectorizer = CountVectorizer()
        # Or choose TF-IDF:
        # vectorizer = TfidfVectorizer()
        X_train = X_all = vectorizer.fit_transform(train_data.text)
        
        # Linear SVM
        clf = svm.SVC(kernel="linear", class_weight="balanced")
        clf.fit(X_train, y_train)
        accuracy = f1 = None
        logger.info(f"Created {label} baseline classifier {clf}")
        
        # If testing model, split into X and y
        if not test_data.empty:
            y_test = test_data[label]
            X_test = vectorizer.transform(test_data.text)
            # Evaluate
            accuracy, f1 = self.evaluate(clf, X_test, y_test)
            # Join train and test data for cross validation
            all_data = pd.concat([train_data, test_data])
            X_all = vectorizer.fit_transform(all_data.text)
            y_all = all_data[label]
        
        # 10-fold cross validation on all available data
        cv_scores = cross_val_score(clf, X_all, y_all, cv=10, scoring="f1_micro")
        logger.info(f"Cross-validation micro f1 mean {cv_scores.mean():.3f}, std {cv_scores.std():.3f}")
        
        # Return namedtuple of classifier and metrics
        return self.model(clf, accuracy, f1, cv_scores.mean(), cv_scores.std())

    def predict(self):
        ""

        pass

    def evaluate(self, clf, X_test, y_test):
        ""

        pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, pred)
        f1 = metrics.f1_score(y_test, pred, average="micro")
        logger.info(f"Accuracy on test set {accuracy:.3f}, micro f1 {f1:.3f}")
        return accuracy, f1

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
    # clf.train(f, "atheism", test_model=False)
    
    # Run automatically for different targets
    # targets = ["atheism", "supernatural", "christianity", "islam"]
    targets = ["atheism"]
    testing = False
    for t in targets:
        print(t.upper())
        clf.train(f, t, test_model=testing)
        print()