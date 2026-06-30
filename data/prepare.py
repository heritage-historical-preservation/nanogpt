"""Preprocess + tokenize the corpus into train/val token arrays.

Reads cleaned text from data/processed/, builds the char-level vocab, encodes
the whole corpus, and writes train.bin / val.bin (+ meta.pkl with the vocab).
"""


def main():
    raise NotImplementedError("build vocab, encode corpus, write train/val arrays")


if __name__ == "__main__":
    main()
