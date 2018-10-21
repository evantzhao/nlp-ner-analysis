import time

from baseline import MostFrequentClassBaseline
from constants import Constants
from emission import EmissionMatrix
from hmm import HiddenMarkovModel
from transition import TransitionMatrix
from unknown import Unknown
from utilities import Utils


TRAINING_FILE_PATH = "../train.txt"
TEST_FILE_PATH = "../test.txt"


def log(msg):
        """ Log a message to stdout with time elapsed"""
        def fill_spaces(msg):
            return " " * (50 - len(msg))

        elapsed = time.time() - start
        minutes = int(elapsed / 60)
        seconds = int(elapsed) % 60
        print(f"{msg}{fill_spaces(msg)}Elapsed time: {minutes} min {seconds} sec")


def main():
    """Performs Named Entity Recognition (NER) using
    a Most Frequent Class Baseline, a Hidden Markove Model (HMM),
    and a Maximum Entropy Markov Model (MEMM).
    """
    log("Starting named entity recognition task")

    log("Reading training file")
    (token_stream, pos_stream, tag_stream), (v_token_stream, v_pos_stream, v_tag_stream) = Utils.create_training_validation_split(TRAINING_FILE_PATH)

    log("Filtering low frequency tokens from training set")
    token_counts = Unknown.get_token_counts(token_stream)
    token_stream, low_freq_set = Unknown.replace_low_frequency_tokens(
        token_counts,
        token_stream,
    )
    for token in low_freq_set:
        del token_counts[token]

    log("Reading test file")
    test_token_stream = Utils.read_test_file(TEST_FILE_PATH)

    log("Training most frequent class baseline")
    baseline = MostFrequentClassBaseline(token_stream, tag_stream)
    log("Predicting tags using baseline")
    baseline_predictions = baseline.most_frequent_class(test_token_stream, token_counts.keys())
    baseline_output = Utils.compile_output_data(baseline_predictions)
    log("Writing predictions with baseline to file")
    Utils.write_results_to_file(baseline_output, "../output/baseline_output.txt")

    # TODO: This should be moved into HMM
    em = EmissionMatrix(token_stream, tag_stream)
    tm = TransitionMatrix(tag_stream)
    log("Constructed HMM matrices")

    res = []
    # Test token stream, test part of speech, test index
    for tt_stream, test_pos, tidx in test_token_stream:
        # log(f"Classifying sentence {tidx[1]}")
        psuedo_tokens = Unknown.convert_word_to_psuedo_word(
            tt_stream,
            Unknown.compute_test_word_replacement_set(token_counts.keys(), tt_stream)
        )
        memo, bt = HiddenMarkovModel.viterbi(tm, em, psuedo_tokens)
        predicted_tags = HiddenMarkovModel.backtrack(memo, bt)
        res.extend(predicted_tags)
    log("Classification completed")

    output = Utils.compile_output_data(res)
    log("Compilation of data complete")

    Utils.write_results_to_file(output, "../output/hmm_output.txt")
    log("Classification complete")


if __name__ == '__main__':
    global start
    start = time.time()
    main()
