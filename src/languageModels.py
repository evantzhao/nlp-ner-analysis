from collections import Counter
from typing import List, Dict


class Constants:
    START = "<s>"
    PER = "person"
    LOC = "location"
    ORG = "organization"
    MISC = "miscellaneous"
    NUMBER_OF_UNIQUE_LABELS = 4


class Unigram:
    def find_unigram_counts(corpus: List[str]) -> Dict[str, int]:
        unigram = Counter()
        for token in corpus:
            unigram[token] += 1
        return unigram


class Preprocess:
    def add_start_token():
        pass


class Bigram:
    def find_bigram_counts(
        corpus: List[str]
    ) -> Dict[str, Dict[str, int]]:
        bigram_counts = {}
        start = corpus[0]
        for token in corpus[1:]:
            if start not in bigram_counts:
                bigram_counts[start] = Counter()
            if token not in bigram_counts[start]:
                bigram_counts[start][token] += 1
            start = token
        return bigram_counts

    def word_count_given_prior(bigram, prior, word):
        return bigram[prior][word]

    def probability_of_word_given_prior(unigram, bigram, prior, word):
        return float(bigram[prior][word]) / unigram[prior]


def main():
    corpus = [Constants.START, "the", "quick", "brown", "fox", "jumped", "over", "a", "the", "poop"]
    bc = Bigram.find_bigram_counts(corpus)
    for key in bc:
        print(f"{key}: {bc[key]}")
    ug = Unigram.find_unigram_counts(corpus)
    print(ug)
    print(Bigram.probability_of_word_given_prior(ug, bc, "the", "quick"))


if __name__ == '__main__':
    main()
