import time
from typing import List
from transition import TransitionMatrix
from emission import EmissionMatrix
from constants import Constants


class HiddenMarkovModel:
    """ Aggregate class containing methods to run the Viterbi algorithm on a
    sequence of words. Requires a transition probability model (bigram/trigram)
    and emission probabilities.
    """
    def allowable_tags_at_sequence_index(index):
        if index == 1:
            return {Constants.START}
        else:
            return Constants.ALL_TAGS

    def viterbi(
        transition: TransitionMatrix,
        emission: EmissionMatrix,
        sentence: List[str],
    ) -> List[str]:
        """ Runs the viterbi algorithm
        """
        memo = [
            [0] * len(Constants.ALL_TAGS) for i in range(len(sentence) - 1)
        ]
        backtracking = [
            [-1] * len(Constants.ALL_TAGS) for i in range(len(sentence) - 1)
        ]

        memo[0] = [1]
        for sentence_index in range(len(memo)):
            if sentence_index == 0:
                continue
            for tag_index in range(len(memo[sentence_index])):
                maximum_probability, backtrack_tag = 0, -1
                possible_tags = HiddenMarkovModel.allowable_tags_at_sequence_index(sentence_index)
                # deal with base case
                if len(possible_tags) == 1:
                    past_prob = memo[sentence_index - 1][0]
                    transition_prob = transition.interpolated_prob_of_word_given_prior(
                        Constants.START,
                        Constants.TAG_TO_STRING[tag_index]
                    )
                    emission_prob = emission.e(
                        word=sentence[sentence_index],
                        state=Constants.TAG_TO_STRING[tag_index]
                    )
                    # TODO evanzhao Use log probabilities
                    sequence_prob = past_prob * transition_prob * emission_prob
                    memo[sentence_index][tag_index] = sequence_prob
                    backtracking[sentence_index][tag_index] = Constants.START
                    continue
                for tag in possible_tags:
                    past_prob = memo[sentence_index - 1][tag]
                    transition_prob = transition.interpolated_prob_of_word_given_prior(
                        prior=Constants.TAG_TO_STRING[tag],
                        word=Constants.TAG_TO_STRING[tag_index]
                    )
                    emission_prob = emission.e(
                        word=sentence[sentence_index],
                        state=Constants.TAG_TO_STRING[tag_index]
                    )
                    sequence_prob = float(past_prob) * transition_prob * emission_prob
                    if sequence_prob > maximum_probability:
                        maximum_probability = sequence_prob
                        backtrack_tag = tag
                memo[sentence_index][tag_index] = maximum_probability
                backtracking[sentence_index][tag_index] = backtrack_tag
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
