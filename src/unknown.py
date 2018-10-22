import re
from collections import Counter
from typing import List, Dict, Set, Tuple
from constants import Constants


class Regex:

    twoDigitNum = re.compile('^[0-9]{2}$'), ".twoDigitNum."
    fourDigitNum = re.compile('^[0-9]{4}$'), ".fourDigitNum."
    containsDigitAndDash = re.compile('^[0-9]+(\-)[0-9]+$'), ".containsDigitAndDash."
    containsDigitAndSlash = re.compile('^([0-9]+(\/))+[0-9]+$'), ".containsDigitAndSlash."
    containsDigitAndComma = re.compile('^[0-9]+,[0-9.]+$'), ".containsDigitAndComma."
    containsDigitAndPeriod = re.compile('^[0-9]*\.[0-9]+$'), ".containsDigitAndPeriod."
    otherNum = re.compile('^[0-9]+$'), ".otherNum."
    allCaps = re.compile('^[A-Z]+$'), ".allCaps."
    capPeriod = re.compile('^[A-Z]\.$'), ".capPeriod."
    firstWord = re.compile('^$'), ".firstWord."
    initCap = re.compile('^[A-Z][a-z]+$'), ".initCap."
    lowercase = re.compile('^[a-z]+$'), ".lowercase."
    OTHER = ".other."

    ITERABLE_REGEX = [
        twoDigitNum,
        fourDigitNum,
        containsDigitAndDash,
        containsDigitAndSlash,
        containsDigitAndComma,
        containsDigitAndPeriod,
        otherNum,
        allCaps,
        capPeriod,
        initCap,
        lowercase
    ]


class Unknown:

    LOW_FREQUENCY_THRESHOLD = 3

    def get_token_counts(
        token_stream: List[str]
    ) -> Dict[str, int]:
        token_counts = Counter()
        for token in token_stream:
            token_counts[token] += 1
        return token_counts

    def replace_low_frequency_tokens(
        token_stream: List[str],
    ) -> Tuple[List[str], Set[str]]:
        """ For all words that appear less than [frequency_threshold] times,
        we remove them from the corpus and replace them with a psuedo word.
        The same thing is done to the test set at classification time --
        if an unknown word appears, we will simply convert it to this psuedo
        word and continue classifying as normal.
        """
        token_counts = Unknown.get_token_counts(token_stream)

        low_frequency_tokens = set()
        for token in token_counts:
            if token_counts[token] < Unknown.LOW_FREQUENCY_THRESHOLD:
                low_frequency_tokens.add(token)

        closed_vocab_corpus = Unknown.convert_word_to_psuedo_word(
            token_stream,
            low_frequency_tokens
        )

        for token in low_frequency_tokens:
            del token_counts[token]

        return closed_vocab_corpus, set(token_counts)

    def replace_unknown_tokens(
        test_stream: Tuple[List[str], List[str], List[str]],
        closed_vocabulary: Set[str],
    ) -> List[Tuple[List[str], List[str], List[str]]]:
        replaced_test_stream = []
        for token_stream, pos_stream, index_stream in test_stream:
            tokens = Unknown.convert_word_to_psuedo_word(
                token_stream,
                Unknown.compute_test_word_replacement_set(
                    closed_vocabulary,
                    token_stream,
                ),
            )
            replaced_test_stream.append((tokens, pos_stream, index_stream))
        return replaced_test_stream

    def convert_word_to_psuedo_word(
        og_stream: List[str],
        low_freq_words: Set[str]
    ) -> List[str]:
        stream = list(og_stream)
        # Convert stream into a closed vocabulary
        for index in range(0, len(stream)):
            if stream[index] == Constants.START or stream[index] == Constants.END:
                continue
            word = stream[index]
            if word in low_freq_words:
                has_match = False
                for i in range(9):
                    if Regex.ITERABLE_REGEX[i][0].match(word):
                        stream[index] = Regex.ITERABLE_REGEX[i][1]
                        has_match = True
                        break
                if has_match:
                    continue
                if stream[index - 1] == Constants.START:
                    stream[index] = Regex.firstWord[1]
                    continue
                for i in range(9, len(Regex.ITERABLE_REGEX)):
                    if Regex.ITERABLE_REGEX[i][0].match(word):
                        stream[index] = Regex.ITERABLE_REGEX[i][1]
                        has_match = True
                        break
                if not has_match:
                    stream[index] = Regex.OTHER
        return stream

    def compute_test_word_replacement_set(
        recognized_words: Set[str],
        sentence: List[str]
    ):
        replace_set = set()
        for i in range(len(sentence)):
            word = sentence[i]
            if word == Constants.START or word == Constants.END:
                continue
            if word not in recognized_words:
                replace_set.add(word)
        return replace_set
