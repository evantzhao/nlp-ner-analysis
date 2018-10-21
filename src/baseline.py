from collections import Counter
from typing import Dict
from typing import List
from typing import Set

from constants import Constants
from emission import EmissionMatrix
from unknown import Unknown


class MostFrequentClassBaseline:
    """ Assign token to the class it occurred in most in the training set
    """

    def __init__(
        self,
        training_token_stream: List[str],
        training_tag_stream: List[str],
    ):
        self.training_token_stream = training_token_stream
        self.training_tag_stream = training_tag_stream
        self.counts = self.count()

    def count(self) -> Dict[str, Dict[str, int]]:
        counts = {}
        for token, tag in zip(self.training_token_stream, self.training_tag_stream):
            if token == Constants.START or token == Constants.END:
                continue
            if token not in counts:
                counts[token] = Counter()
            counts[token][tag] += 1
        return counts

    def most_frequent_class(
        self,
        test_stream,
        recognized_tokens: Set[str],
    ) -> List[str]:
        prediction = []
        for token_stream, pos_stream, _ in test_stream:
            pseudo_token_stream = Unknown.convert_word_to_psuedo_word(
                token_stream,
                Unknown.compute_test_word_replacement_set(recognized_tokens, token_stream),
            )
            for token in pseudo_token_stream:
                if token == Constants.START or token == Constants.END:
                    continue
                most_frequent_tag = ""
                most_frequent_tag_count = 0
                if token not in self.counts:
                    print(token)
                for tag in self.counts[token]:
                    if self.counts[token][tag] > most_frequent_tag_count:
                        most_frequent_tag_count = self.counts[token][tag]
                        most_frequent_tag = tag
                prediction.append(tag)
        return prediction
