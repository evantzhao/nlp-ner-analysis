from collections import Counter
from typing import List, Dict
from constants import Constants


class TransitionMatrix:

    def __init__(self, corpus: List[str]) -> None:
        self.unigram = self.construct_unigram_counts(corpus)
        self.bigram = self.construct_bigram_counts(corpus)

    def construct_unigram_counts(self, corpus: List[str]) -> Dict[str, int]:
        unigram = Counter()
        for token in corpus:
            unigram[token] += 1
        return unigram

    def construct_bigram_counts(
        self,
        corpus: List[str]
    ) -> Dict[str, Dict[str, int]]:
        bigram_counts = {}
        start = corpus[0]
        for token in corpus[1:]:
            # Don't double count this
            if token != Constants.START:
                if start not in bigram_counts:
                    bigram_counts[start] = Counter()
                if token not in bigram_counts[start]:
                    bigram_counts[start][token] += 1
            start = token
        return bigram_counts

    def word_count_given_prior(self, prior: str, word: str) -> int:
        return self.bigram[prior][word]

    def probability_of_word_given_prior(self, prior: str, word: str) -> float:
        if prior not in self.bigram:
            return 0.0
        return float(self.bigram[prior][word]) / self.unigram[prior]


def main():
    pass
    # corpus = [Constants.START, "the", "quick", "brown", "fox", "jumped", "over", "a", "the", "poop"]
    # bc = Bigram.find_bigram_counts(corpus)
    # for key in bc:
    #     print(f"{key}: {bc[key]}")
    # ug = Unigram.find_unigram_counts(corpus)
    # print(ug)
    # print(Bigram.probability_of_word_given_prior(ug, bc, "the", "quick"))


if __name__ == '__main__':
    main()
