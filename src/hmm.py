import time
from typing import List
from typing import Set
from typing import Tuple

from transition import TransitionMatrix
from emission import EmissionMatrix
from constants import Constants
from unknown import Unknown


class HiddenMarkovModel:

    def __init__(
        self,
        token_stream: List[str],
        tag_stream: List[str],
        closed_vocabulary: Set[str]
    ):
        self.emission = EmissionMatrix(token_stream, tag_stream)
        self.transition = TransitionMatrix(tag_stream)
        self.closed_vocabulary = closed_vocabulary

    def classify_test_stream(
        self,
        test_stream: Tuple[List[str], List[str], List[str]],
    ) -> List[str]:
        predictions = []
        for token_stream, _, _ in test_stream:
            tokens = Unknown.convert_word_to_psuedo_word(
                token_stream,
                Unknown.compute_test_word_replacement_set(
                    self.closed_vocabulary,
                    token_stream,
                ),
            )
            memo, backtracking = self.viterbi(tokens)
            predicted_tags = self.backtrack(memo, backtracking)
            predictions.extend(predicted_tags)
        return predictions

    def viterbi(
        self,
        tokens: List[str],
    ) -> List[str]:

        memo = [
            [0] * len(Constants.ALL_TAGS) for i in range(len(tokens) - 1)
        ]
        backtracking = [
            [-1] * len(Constants.ALL_TAGS) for i in range(len(tokens) - 1)
        ]

        memo[0] = [1]
        for token_index in range(1, len(memo)):
            for tag_index in range(len(memo[token_index])):
                maximum_probability, backtrack_tag = 0, -1
                # deal with base case
                if token_index == 1:
                    past_prob = memo[token_index - 1][0]
                    transition_prob = self.transition.get_bigram_interpolated_probability(
                        Constants.START,
                        Constants.TAG_TO_STRING[tag_index]
                    )
                    emission_prob = self.emission.e(
                        word=tokens[token_index],
                        state=Constants.TAG_TO_STRING[tag_index]
                    )
                    # TODO evanzhao Use log probabilities
                    sequence_prob = past_prob * transition_prob * emission_prob
                    memo[token_index][tag_index] = sequence_prob
                    backtracking[token_index][tag_index] = Constants.START
                    continue
                for tag in Constants.ALL_TAGS:
                    past_prob = memo[token_index - 1][tag]
                    transition_prob = self.transition.get_bigram_interpolated_probability(
                        Constants.TAG_TO_STRING[tag],
                        Constants.TAG_TO_STRING[tag_index]
                    )
                    emission_prob = self.emission.e(
                        word=tokens[token_index],
                        state=Constants.TAG_TO_STRING[tag_index]
                    )
                    sequence_prob = float(past_prob) * transition_prob * emission_prob
                    if sequence_prob > maximum_probability:
                        maximum_probability = sequence_prob
                        backtrack_tag = tag
                memo[token_index][tag_index] = maximum_probability
                backtracking[token_index][tag_index] = backtrack_tag
        return memo, backtracking

    def backtrack(
        self,
        memo: List[List[float]],
        backtracking: List[List[int]]
    ) -> List[str]:
        seed_level = memo[-1]
        seed_index = seed_level.index(max(seed_level))

        result = []
        i = len(backtracking) - 1
        while i >= 1:
            result.append(seed_index)
            seed_index = backtracking[i][seed_index]
            i -= 1
        return [Constants.TAG_TO_STRING[tag] for tag in reversed(result)]
