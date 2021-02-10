import pandas as pd

def main(f):
    df = pd.read_csv(f, sep="\t")
    # Replace columns for easier access
    df.columns = ["id", "text", "atheism", "secularism", "religious_freedom", "freethinking", 
                  "no_evidence", "supernatural", "christianity", "afterlife", "usa", "islam",
                  "conservatism", "same_sex_marriage"]
    # Convert stances to ints
    df.fillna(0, inplace=True)
    df.replace(["none", "favor", "against"], [0, 1, -1], inplace=True)
    print(df.head())

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    # ! Use preprocess.py to preprocess the ASC
    corpus = "data/corpus_preprocessed.txt"
    main(corpus)