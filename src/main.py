from utilities import Utils
from emission import EmissionMatrix
from transition import TransitionMatrix
from hmm import HiddenMarkovModel
from constants import Constants

TRAINING_FILE = "../train.txt"
TRAINING_FILE = "../woah.txt"
TEST_FILE = "../test.txt"

token_stream, pos_stream, tag_stream = Utils.read_training_file(
    TRAINING_FILE,
    insert_start_and_end=True
)
em = EmissionMatrix(token_stream, tag_stream)
tm = TransitionMatrix(tag_stream)
sentence = ['<START(*)>', 'Michael', 'Chang', 'is', 'playing', 'in', 'his', '10th', 'U.S.', 'Open', 'and', 'enjoying', 'his', 'highest', 'seeding', 'ever', ',', 'but', 'the', '24-year-old', 'American', 'had', 'to', 'overcome', 'a', 'case', 'of', 'the', 'jitters', 'Monday', 'before', 'winning', 'his', 'first-round', 'match', 'on', 'opening', 'day', '.', '</END(STOP)>']
memo, bt = HiddenMarkovModel.viterbi(tm, em, sentence)
predicted_tags = HiddenMarkovModel.backtrack(memo, bt)
expected_tags = ['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
passed = True
if len(predicted_tags) == len(expected_tags):
    for t1, t2 in zip(predicted_tags, expected_tags):
        if t1 != t2:
            passed = False
print(f"Test status?: {'Passed' if passed else 'Failed'}")
# print(bt, memo)
