from collections import Counter
from typing import List, Dict
from constants import Constants


class InterpolationConstants:
    # TODO evanzhao separate training set and find better lambda params
    # with empirical evidence
    BIGRAM = 0.99
    UNIGRAM = 0.01


class TransitionMatrix:

    def __init__(self, corpus: List[str]) -> None:
        self.unigram, self.unique_token_count = self.construct_unigram_counts(
            corpus=corpus
        )
        self.bigram = self.construct_bigram_counts(corpus)

    def construct_unigram_counts(self, corpus: List[str]) -> Dict[str, int]:
        unigram = Counter()
        unique_token_count = 0
        for token in corpus:
            unigram[token] += 1
            if token != Constants.START and token != Constants.END:
                unique_token_count += 1
        return unigram, unique_token_count

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

    def get_word_count(self, word: str) -> int:
        return self.unigram[word]

    def probability_of_word(self, word: str) -> float:
        return self.get_word_count(word) / float(self.unique_token_count)

    def word_count_given_prior(self, prior: str, word: str) -> int:
        return self.bigram[prior][word]

    def probability_of_word_given_prior(self, prior: str, word: str) -> float:
        if prior not in self.bigram:
            return 0.0
        return float(self.bigram[prior][word]) / self.unigram[prior]

    def interpolated_prob_of_word_given_prior(
        self,
        prior: str,
        word: str
    ) -> float:
        bigram_term = self.probability_of_word_given_prior(prior, word)
        unigram_term = self.probability_of_word(word)
        scaled_bigram = InterpolationConstants.BIGRAM * bigram_term
        scaled_unigram = InterpolationConstants.UNIGRAM * unigram_term
        return scaled_bigram + scaled_unigram
