from utilities import Utils
from emission import EmissionMatrix
from transition import TransitionMatrix
from hmm import HiddenMarkovModel
from constants import Constants
import time
from unk import Unknown

TRAINING_FILE = "../train.txt"
TEST_FILE = "../test.txt"


def main():
    token_stream, pos_stream, tag_stream = Utils.read_training_file(
        TRAINING_FILE,
        insert_start_and_end=True
    )
    log("Read training file")
    test_token_stream = Utils.read_test_file(TEST_FILE)
    log("Read test file")

    word_counts = Unknown.construct_word_counts(token_stream)
    token_stream, low_freq_set = Unknown.remove_low_freqency_words(
        Constants.LOW_FREQUENCY_WORD_THRESHOLD,
        word_counts,
        token_stream
    )
    recognized_words = set(word_counts.keys()) - low_freq_set
    log("Filtered out low frequency words from training")

    em = EmissionMatrix(token_stream, tag_stream)
    tm = TransitionMatrix(tag_stream)
    log("Constructed HMM matrices")

    res = []
    # Test token stream, test part of speech, test index
    for tt_stream, test_pos, tidx in test_token_stream:
        log(f"Classifying sentence {tidx[1]}")
        psuedo_tokens = Unknown.convert_word_to_psuedo_word(
            tt_stream,
            Unknown.compute_test_word_replacement_set(recognized_words, tt_stream)
        )
        memo, bt = HiddenMarkovModel.viterbi(tm, em, psuedo_tokens)
        predicted_tags = HiddenMarkovModel.backtrack(memo, bt)
        res.extend(predicted_tags)
    log("Classification completed")

    output = Utils.compile_output_data(res)
    log("Compilation of data complete")

    Utils.write_results_to_file(output, "../output/hmm_output.txt")
    log("Classification complete")


def log(s):
    """ Log a message with time consumed dialog
    """
    def generate_spaces(s):
        return " " * (50 - len(s))

    elapsed = time.time() - start
    minutes = int(elapsed/60)
    seconds = int(elapsed) % 60
    print(f"{s}{generate_spaces(s)}Elapsed time: {minutes} min {seconds} sec")


if __name__ == '__main__':
    global start
    start = time.time()
    main()
