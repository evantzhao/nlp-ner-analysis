from collections import Counter
from typing import List, Dict
from constants import Constants


class TransitionMatrix:

    def __init__(
        self,
        tag_stream: List[str],
    ):
        self.total_tag_count = self.compute_total_tag_count(tag_stream)
        self.unigram_counts = self.compute_unigram_counts(tag_stream)
        self.bigram_counts = self.compute_bigram_counts(tag_stream)
        self.trigram_counts = self.compute_trigram_counts(tag_stream)

    def compute_total_tag_count(
        self,
        tag_stream: List[str],
    ) -> int:
        total_tag_count = 0
        for tag in tag_stream:
            if tag != Constants.START and tag != Constants.END:
                total_tag_count += 1
        return total_tag_count

    def compute_unigram_counts(
        self,
        tag_stream: List[str],
    ) -> Dict[str, int]:
        unigram_counts = Counter()
        for token in tag_stream:
            unigram_counts[token] += 1
        return unigram_counts

    def compute_bigram_counts(
        self,
        tag_stream: List[str],
    ) -> Dict[str, Dict[str, int]]:
        bigram_counts = {}
        prior_i_1 = tag_stream[0]
        for tag in tag_stream[1:]:
            if tag != Constants.START:
                if prior_i_1 not in bigram_counts:
                    bigram_counts[prior_i_1] = Counter()
                bigram_counts[prior_i_1][tag] += 1
            prior_i_1 = tag
        return bigram_counts

    def compute_trigram_counts(
        self,
        tag_stream: List[str],
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        trigram_counts = {}
        prior_i_2 = tag_stream[0]
        prior_i_1 = tag_stream[1]
        for tag in tag_stream[2:]:
            if tag != Constants.START and prior_i_1 != Constants.START:
                if prior_i_2 not in trigram_counts:
                    trigram_counts[prior_i_2] = {}
                if prior_i_1 not in trigram_counts[prior_i_2]:
                    trigram_counts[prior_i_2][prior_i_1] = Counter()
                trigram_counts[prior_i_2][prior_i_1][tag] += 1
            prior_i_2 = prior_i_1
            prior_i_1 = tag
        return trigram_counts

    def get_unigram_probability(
        self,
        tag: str,
    ) -> float:
        return self.unigram_counts[tag] / float(self.total_tag_count)

    def get_bigram_probability(
        self,
        prior_i_1: str,
        tag: str,
    ) -> float:
        if prior_i_1 not in self.bigram_counts:
            return 0.0
        return self.bigram_counts[prior_i_1][tag] / float(self.unigram_counts[prior_i_1])

    def get_trigram_probability(
        self,
        prior_i_2: str,
        prior_i_1: str,
        tag: str,
    ) -> float:
        if prior_i_2 not in self.trigram_counts or prior_i_1 not in self.trigram_counts[prior_i_2]:
            return 0.0
        return self.self.trigram_counts[prior_i_2][prior_i_1][tag] / float(self.bigram_counts[prior_i_2][prior_i_1])

    def get_bigram_interpolated_probability(
        self,
        prior_i_1: str,
        tag: str,
    ) -> float:
        UNIGRAM_LAMBDA = 0.05
        BIGRAM_LAMBDA = 0.95
        weighted_unigram_probability = UNIGRAM_LAMBDA * self.get_unigram_probability(tag)
        weighted_bigram_probability = BIGRAM_LAMBDA * self.get_bigram_probability(prior_i_1, tag)
        return weighted_unigram_probability + weighted_bigram_probability

    def get_trigram_interpolated_probability(
        self,
        prior_i_2: str,
        prior_i_1: str,
        tag: str,
    ) -> float:
        UNIGRAM_LAMBDA = 0.01
        BIGRAM_LAMBDA = 0.09
        TRIGRAM_LAMBDA = 0.90
        weighted_unigram_probability = UNIGRAM_LAMBDA * self.get_unigram_probability(tag)
        weighted_bigram_probability = BIGRAM_LAMBDA * self.get_bigram_probability(prior_i_1, tag)
        weighted_trigram_probability = TRIGRAM_LAMBDA * self.get_trigram_probability(prior_i_2, prior_i_1, tag)
        return weighted_trigram_probability + weighted_unigram_probability + weighted_bigram_probability
