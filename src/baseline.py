from collections import Counter
from typing import Dict
from typing import List
from typing import Tuple

from constants import Constants


class MostFrequentClassBaseline:

    def __init__(
        self,
        training_token_stream: List[str],
        training_tag_stream: List[str],
    ):
        self.training_token_stream = training_token_stream
        self.training_tag_stream = training_tag_stream
        self.most_frequent_tag_per_token = self.compute_most_frequent_tag_per_token()

    def compute_tag_counts_per_token(self) -> Dict[str, Dict[str, int]]:
        tag_counts_per_token = {}
        for token, tag in zip(self.training_token_stream, self.training_tag_stream):
            if token == Constants.START or token == Constants.END:
                continue
            if token not in tag_counts_per_token:
                tag_counts_per_token[token] = Counter()
            tag_counts_per_token[token][tag] += 1
        return tag_counts_per_token

    def compute_most_frequent_tag_per_token(self) -> Dict[str, str]:
        most_frequent_tag_per_token = {}
        tag_counts_per_token = self.compute_tag_counts_per_token()
        for token in tag_counts_per_token:
            most_frequent_tag = ""
            most_frequent_tag_count = 0
            for tag in tag_counts_per_token[token]:
                if tag_counts_per_token[token][tag] > most_frequent_tag_count:
                    most_frequent_tag_count = tag_counts_per_token[token][tag]
                    most_frequent_tag = tag
            most_frequent_tag_per_token[token] = most_frequent_tag
        return most_frequent_tag_per_token

    def classify_test_stream(
        self,
        test_stream: Tuple[List[str], List[str], List[str]],
    ) -> List[str]:
        predictions = []
        for token_stream, _, _ in test_stream:
            for token in token_stream:
                if token == Constants.START or token == Constants.END:
                    continue
                predictions.append(self.most_frequent_tag_per_token[token])
        return predictions
