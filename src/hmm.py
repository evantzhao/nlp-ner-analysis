import time
from typing import List, Dict
from languageModels import Constants


class HiddenMarkovModel:
    """ Aggregate class containing methods to run the Viterbi algorithm on a
    sequence of words. Requires a transition probability model (bigram/trigram)
    and emission probabilities.
    """
    def viterbi(
        transition: Dict[str, Dict[str, int]],
        emission: Dict[str, int],
        corpus: List[str],
    ) -> List[str]:
        """ Runs the viterbi algorithm
        """
        memo = [
            [""*len(corpus)] for i in range(Constants.NUMBER_OF_UNIQUE_LABELS)
        ]
        backtracking = [
            [""*len(corpus)] for i in range(Constants.NUMBER_OF_UNIQUE_LABELS)
        ]

        prior = corpus[0]
        for token in corpus[1:]:
            pass
        pass

    def backtrack(self):
        pass


def log(s):
    """ Log a message with time consumed dialog
    """
    def generate_spaces(s):
        return " " * (50 - len(s))

    elapsed = time.time() - start
    minutes = int(elapsed/60)
    seconds = int(elapsed) % 60
    print(f"{s}{generate_spaces(s)}Elapsed time: {minutes} min {seconds} sec")


def main():
    hmm = HiddenMarkovModel([], [])


if __name__ == '__main__':
    global start
    start = time.time()
    main()
