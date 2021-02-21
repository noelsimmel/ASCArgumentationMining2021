from collections import namedtuple
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from transformers import AverageWordLengthExtractor
from preprocessor import Preprocessor

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(funcName)s:%(message)s")
file_handler = logging.FileHandler("logfile.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def dummy(arg):
    ""
    # https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
    
    return arg

class ASCClassifier:
    def __init__(self):
        ""

        model_attributes = ["clf", "accuracy", "micro_f1", "cv_mean", "cv_std"]
        self.model = namedtuple("Model", model_attributes)
        
    def train(self, train_file, target, test_model=False):
        ""

        # Import data
        train_data = self._read_file(train_file, target)
        # FIXME smallest possible dataset for splitting => 8:3, classes [1, 0, -1]
        # train_data = train_data[:11]
        X_test = y_test = None
        
        # Split dataset if classifier should be tested
        if test_model:
            # seed=21 gives atheism classes [0,-1,1,0,0]
            train_data, test_data = self._split_dataset(train_data, target, seed=21)
            X_test = list(test_data["text"])
            y_test = list(test_data[target])
        X_train = list(train_data["text"])
        y_train = list(train_data[target])
        
        logger.info(f"Training start")
        countvec = CountVectorizer(tokenizer=dummy, preprocessor=dummy, lowercase=False)
        
        # Baseline classifier: Linear SVM with bag of words features
        baseline_clf = svm.SVC(kernel="linear", class_weight="balanced")
        baseline_transformers = [("bow", countvec)]
        baseline_ppl = self.build_pipeline(X_train, y_train, baseline_transformers, baseline_clf)
        # 10-fold cross validation on all available data
        cv_scores = self.evaluate_cv(baseline_ppl, X_train, y_train)
        
        # If testing model, evaluate
        accuracy = f1 = None
        if X_test and y_test:
            accuracy, f1 = self.evaluate_metrics(baseline_ppl, X_test, y_test)
        baseline_model = self.model(baseline_ppl, accuracy, f1, cv_scores.mean(), cv_scores.std())
        print(baseline_model)
        
        
        # # another classifier goes here ...
        # transformers = [("bow", countvec),
        #                 ("average", AverageWordLengthExtractor())]
        # clf = svm.SVC(kernel="linear", class_weight="balanced")
        # ppl = self.build_pipeline(X_train, y_train, transformers, clf)
        # # 10-fold cross validation on all available data
        # cv_scores = self.evaluate_cv(ppl, X_train, y_train)
        # print(cv_scores.mean(), cv_scores.std())
        
    def build_model(self, X_train, y_train, X_test, y_test, vectorizer, clf):
        ""
        
        y_train = y_all = train_data[target]
        
        data = train_data.text
        # data = pd.Series(["dies ist text eins", "das hier ist text text zwei"])
        X_train = X_all = vectorizer.fit_transform(data)
        
        clf.fit(X_train, y_train)
        accuracy = f1 = None
        logger.info(f"Created {label} classifier {clf}")
        
        # doc idx (axis 0) of most frequent token: idx_0 = int(X_train.argmax()/X_train.shape[1])
        # token idx (axis 1) of most frequent token: idx_1 = int(X_train.argmax(axis=1)[idx_0])
        # freq of most freq token: X_train.max()
        
        # list with tuples (token, freq), ordered alphabetically:
        # https://stackoverflow.com/a/16078639/2491761
        # tc = zip(vectorizer.get_feature_names(), np.asarray(X_train.sum(axis=0)).ravel())
        # # as df (idx is feature idx as in vectorizer.vocabulary_ !)
        # df = pd.DataFrame(tc, columns=["token", "freq"])
        # print(X_train.toarray())
        # print(int(X_train.argmax(axis=1)[1]))
        # print(vectorizer.vocabulary_)
        # print(vectorizer.get_feature_names())
        # return
        
        # If testing model, split into X and y
        if not test_data.empty:
            y_test = test_data[target]
            X_test = vectorizer.transform(test_data.text)
            # Evaluate
            accuracy, f1 = self.evaluate_metrics(clf, X_test, y_test)
            # Join train and test data for cross validation
            all_data = pd.concat([train_data, test_data])
            X_all = vectorizer.fit_transform(all_data.text)
            y_all = all_data[label]
        
        # 10-fold cross validation on all available data
        cv_scores = self.evaluate_cv(clf, X_all, y_all)
        
        # Return namedtuple of classifier and metrics
        return self.model(clf, accuracy, f1, cv_scores.mean(), cv_scores.std())
    
    def build_pipeline(self, X_train, y_train, transformers, clf):
        ""
        
        pipeline = Pipeline([("preprocessing", Preprocessor()),
                             ("features", FeatureUnion(transformers)), 
                             ("clf", clf)])
        logger.info(f"Created pipeline: {pipeline.get_params()}")
        
        pipeline.fit(X_train, y_train)
        logger.info("Fitted pipeline to training data")
        
        return pipeline
    
    def evaluate_metrics(self, pipeline, X_test, y_test):
        ""

        pred = pipeline.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, pred)
        f1 = metrics.f1_score(y_test, pred, average="micro")
        logger.info(f"Accuracy on test set {accuracy:.3f}, micro f1 {f1:.3f}")
        return accuracy, f1
    
    def evaluate_cv(self, pipeline, X, y):
        ""
        
        # FIXME cv=10
        cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring="f1_micro")
        logger.info(f"Cross-validation micro f1 mean {cv_scores.mean():.3f}, std {cv_scores.std():.3f}")
        return cv_scores
    
    def predict(self):
        ""

        pass
    
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