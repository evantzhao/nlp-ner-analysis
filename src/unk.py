import re
from collections import Counter
from typing import List, Dict, Set
from constants import Constants


class Regex:
    twoDigitNum = re.compile('^[0-9]{2}$'), ".twoDigitNum."
    fourDigitNum = re.compile('^[0-9]{4}$'), ".fourDigitNum."
    # containsDigitAndAlpha = '^$'
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

    ITERABLE_REGEX = [twoDigitNum, fourDigitNum, containsDigitAndDash, containsDigitAndSlash, containsDigitAndComma, containsDigitAndPeriod, otherNum, allCaps, capPeriod, initCap, lowercase]


class Unknown:
    def construct_word_counts(stream: List[str]) -> Dict[str, int]:
        counts = Counter()
        for word in stream:
            counts[word] += 1
        return counts

    def remove_low_freqency_words(
        frequency_threshold: int,
        unigram: Dict[str, int],
        stream: List[str],
    ):
        """ For all words that appear less than [frequency_threshold] times,
        we remove them from the corpus and replace them with a psuedo word.
        The same thing is done to the test set at classification time --
        if an unknown word appears, we will simply convert it to this psuedo
        word and continue classifying as normal.
        """
        low_freq_set = set()
        for word in unigram:
            if unigram[word] < frequency_threshold:
                low_freq_set.add(word)
        closed_vocab_corpus = Unknown.convert_word_to_psuedo_word(
            stream,
            low_freq_set
        )
        return closed_vocab_corpus, low_freq_set

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
