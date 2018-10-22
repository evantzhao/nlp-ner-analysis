from collections import Counter
from typing import Dict
from typing import List

from constants import Constants


class EmissionMatrix:

    def __init__(
        self,
        token_stream: List[str],
        tag_stream: List[str]
    ) -> None:
        self.emissions_matrix = self.construct_emissions(
            token_stream,
            tag_stream,
        )
        self.tag_count = self.construct_emissions_tag_count()

    def construct_emissions(
        self,
        token_stream: List[str],
        tag_stream: List[str]
    ) -> Dict[str, Dict[str, int]]:
        assert(len(token_stream) == len(tag_stream))

        emissions_matrix = {}
        for token, tag in zip(token_stream, tag_stream):
            if token == Constants.START or token == Constants.END:
                continue
            if tag not in emissions_matrix:
                emissions_matrix[tag] = Counter()
            emissions_matrix[tag][token] += 1

        return emissions_matrix

    def get_state_with_token_count(
        self,
        state: str,
        token: str,
    ) -> int:
        return self.emissions_matrix[state][token]

    def construct_emissions_tag_count(self) -> Dict[str, int]:
        tag_count = {}
        for tag in self.emissions_matrix:
            count = sum(self.emissions_matrix[tag].values())
            tag_count[tag] = count
        return tag_count

    def get_state_occurences(
        self,
        state: str,
    ) -> int:
        if state not in self.tag_count:
            return 1
        return self.tag_count[state]

    def e(
        self,
        word: str,
        state: str,
    ) -> float:
        state_token_count = self.get_state_with_token_count(state, word)
        state_occurs = self.get_state_occurences(state)
        return float(state_token_count) / state_occurs
