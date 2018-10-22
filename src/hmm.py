import time
from typing import List
from transition import TransitionMatrix
from emission import EmissionMatrix
from constants import Constants


class HiddenMarkovModel:

    def viterbi(
        transition: TransitionMatrix,
        emission: EmissionMatrix,
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
                    transition_prob = transition.get_bigram_interpolated_probability(
                        Constants.START,
                        Constants.TAG_TO_STRING[tag_index]
                    )
                    emission_prob = emission.e(
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
                    transition_prob = transition.get_bigram_interpolated_probability(
                        Constants.TAG_TO_STRING[tag],
                        Constants.TAG_TO_STRING[tag_index]
                    )
                    emission_prob = emission.e(
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
        memoization_matrix: List[List[int]],
        backtracking_matrix: List[List[int]]
    ) -> List[str]:
        seed_level = memoization_matrix[-1]
        seed_index = seed_level.index(max(seed_level))

        result = []
        i = len(backtracking_matrix) - 1
        while i >= 1:
            result.append(seed_index)
            seed_index = backtracking_matrix[i][seed_index]
            i -= 1
        return [Constants.TAG_TO_STRING[tag] for tag in reversed(result)]
