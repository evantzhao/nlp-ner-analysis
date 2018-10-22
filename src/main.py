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
    def fill_spaces(msg):
        return " " * (60 - len(msg))

    elapsed = time.time() - start
    minutes = int(elapsed / 60)
    seconds = int(elapsed) % 60
    print(f"{msg}{fill_spaces(msg)}Elapsed time: {minutes} min {seconds} sec")


def main():
    log("Starting named entity recognition task")

    log("Reading training file")
    token_stream, pos_stream, tag_stream = Utils.read_training_file(TRAINING_FILE_PATH)
    # (token_stream, pos_stream, tag_stream), (v_token_stream, v_pos_stream, v_tag_stream) = Utils.create_training_validation_split(TRAINING_FILE_PATH)
    log("Replacing low frequency tokens from training set")
    token_stream, closed_vocabulary = Unknown.replace_low_frequency_tokens(token_stream)

    log("Reading test file")
    test_stream = Utils.read_test_file(TEST_FILE_PATH)
    log("Replacing unknown tokens from test set")
    test_stream = Unknown.replace_unknown_tokens(test_stream, closed_vocabulary)

    log("Training most frequent class baseline")
    baseline = MostFrequentClassBaseline(token_stream, tag_stream)
    log("Predicting tags using baseline")
    baseline_predictions = baseline.classify_test_stream(test_stream)
    log("Writing predictions with baseline to file")
    Utils.write_results_to_file(baseline_predictions, "../output/baseline_output.txt")

    log("Training Hidden Markov Model")
    hmm = HiddenMarkovModel(token_stream, tag_stream)
    log("Predicting tags using HMM")
    hmm_predictions = hmm.classify_test_stream(test_stream)
    log("Writing predictions with HMM to file")
    Utils.write_results_to_file(hmm_predictions, "../output/hmm_output.txt")


if __name__ == '__main__':
    global start
    start = time.time()
    main()
