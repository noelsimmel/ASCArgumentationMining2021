# ascpreprocess.py
# Preprocesses the Atheism Stance Corpus for use with Pandas

import pandas as pd

class ASCPreprocess():
    """Preprocesses the Atheism Stace Corpus to fix whitespace errors.
    The id and text columns are separated by spaces only, instead of tabs,
    resulting in erroneous import in Pandas."""

    def __init__(self, input_fn, output_fn):
        """Constructor. Calls the individual preprocessing functions.
        
        Args:
            input_fn (str): Path to the original ASC
            output_fn (str): Desired output path
        """

        self.fix_whitespace(input_fn, output_fn)
        self.preprocess_for_classification(output_fn)

    def fix_whitespace(self, input_fn, output_fn):
        """Fixes the whitespace error and simplifies the stance columns.
        Also removes superfluous whitespace from text.
        
        Args:
            input_fn (str): Path to the original ASC
            output_fn (str): Desired output path
        """

        with open(input_fn, mode="r") as source:
            with open(output_fn, mode="w+") as target:
                # Separate "id  text" in header
                header = ["id", "text"] + source.readline().split("\t")[1:]
                header_str = "\t"
                header_str = header_str.join(header)
                target.write(header_str + "\n")
                for l in source:
                    # Separate line into id/text + categories (argument targets)
                    l = l.strip().split("\t")
                    id_text = l[0].split()
                    categories = l[1:]
                    # Only keep stance since target is named in header
                    for i in range(len(categories)):
                        cat = categories[i].split(":")
                        categories[i] = cat[1]
                    # Remove unnecessary whitespace from text
                    text = " "
                    text = text.join(id_text[1:])
                    # Join everything together and write to file
                    l = [id_text[0], text] + categories
                    l_str = "\t"
                    l_str = l_str.join(l)
                    target.write(l_str + "\n")

    def preprocess_for_classification(self, fn):
        """Simplifies the data further by renaming columns and representing
        stances as ints.
        
        Args:
            fn (str): Input/output path (the input file is overwritten)
        """

        df = pd.read_csv(fn, sep="\t")
        # Rename columns for easier access
        df.columns = ["id", "text", "atheism", "secularism", "religious_freedom", "freethinking",
                    "no_evidence", "supernatural", "christianity", "afterlife", "usa", "islam",
                    "conservatism", "same_sex_marriage"]
        # Represent stances as ints 0, 1, -1
        df.fillna(0, inplace=True)
        df.replace(["none", "favor", "against"], [0, 1, -1], inplace=True)
        # Overwrite file with new dataframe
        df.to_csv(fn, sep='\t', index=False)


if __name__ == "__main__":
    ASCPreprocess("data/atheism stance corpus.txt", "data/corpus.txt")