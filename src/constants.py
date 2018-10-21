class Constants:

    START = "<START(*)>"
    END = "</END(STOP)>"
    BPER = 0
    BLOC = 1
    BORG = 2
    BMISC = 3
    IPER = 4
    ILOC = 5
    IORG = 6
    IMISC = 7
    OTHER = 8
    ALL_TAGS = {BPER, BLOC, BORG, BMISC, IPER, ILOC, IORG, IMISC, OTHER}
    TAG_TO_STRING = {
        BPER: "B-PER",
        BLOC: "B-LOC",
        BORG: "B-ORG",
        BMISC: "B-MISC",
        IPER: "I-PER",
        ILOC: "I-LOC",
        IORG: "I-ORG",
        IMISC: "I-MISC",
        OTHER: "O",
    }
