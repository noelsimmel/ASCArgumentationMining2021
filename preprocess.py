def main(f):
    """
    Preprocesses the Atheism Stace Corpus to fix whitespace errors. 
    The id and text columns are separated by spaces only, instead of tabs, 
    resulting in erroneous import in Pandas.

    - f (str): Path to Atheism Stance Corpus 

    Creates a new txt file in a data folder.
    """
    with open(f, mode="r") as source:
        with open("data/corpus_preprocessed.txt", mode="w+") as target:
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


if __name__ == "__main__":
    main("data/atheismstancecorpus.txt")