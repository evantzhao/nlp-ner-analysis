from constants import Constants
from typing import List


class Utils:
    def read_training_file(
        filepath: str,
        insert_start_and_end: bool = False,
    ) -> List[str]:
        streams = [[], [], []]
        index = 0

        with open(filepath, 'r') as f:
            for line in f:
                line_arr = line.strip().split()
                if insert_start_and_end:
                    line_arr.insert(0, Constants.START)
                    line_arr.append(Constants.END)
                streams[index].extend(line_arr)
                index = (index + 1) % len(streams)
        return streams


def main():
    Utils.read_training_file("../train.txt")


if __name__ == '__main__':
    main()
