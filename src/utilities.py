from constants import Constants
from typing import Dict
from typing import List
from typing import Tuple


class Utils:

    # Reads in training file, concatenating all three data types into one ds
    def read_training_file(
        filepath: str,
        insert_start_and_end: bool = True,
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

    def create_training_validation_split(
        training_file_path: str,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        training_streams = [[], [], []]
        validation_streams = [[], [], []]
        with open(training_file_path, 'r') as f:
            lines = f.readlines()
            offset = 0
            entry_num = 0
            while offset < len(lines):
                tokens = Utils.add_start_end_tokens(lines[offset])
                pos = Utils.add_start_end_tokens(lines[offset + 1])
                tags = Utils.add_start_end_tokens(lines[offset + 2])
                if entry_num % 10 != 9:
                    training_streams[0].extend(tokens)
                    training_streams[1].extend(pos)
                    training_streams[2].extend(tags)
                else:
                    validation_streams[0].extend(tokens)
                    validation_streams[1].extend(pos)
                    validation_streams[2].extend(tags)
                offset += 3
                entry_num += 1
        return training_streams, validation_streams


    def add_start_end_tokens(
        line: str,
    ) -> List[str]:
        line_arr = line.strip().split()
        line_arr.insert(0, Constants.START)
        line_arr.append(Constants.END)
        return line_arr

    def read_test_file(
        filepath: str,
    ):
        sentences = []
        index = 0

        with open(filepath, 'r') as f:
            streams = [[], [], []]
            for line in f:
                line_arr = line.strip().split()
                if index == 2:
                    line_arr.insert(0, -1)
                    line_arr.append(-1)
                else:
                    line_arr.insert(0, Constants.START)
                    line_arr.append(Constants.END)
                streams[index].extend(line_arr)
                if index == 2:
                    sentences.append((streams[0], streams[1], streams[2]))
                    streams = [[], [], []]
                index = (index + 1) % len(streams)
        return sentences

    # TODO ryan can u refactor this :(
    def compile_output_data(
        tags: List[str]
    ) -> Dict[str, List[str]]:
        results = {
            "PER": [],
            "LOC": [],
            "ORG": [],
            "MISC": [],
        }
        mode = Constants.OTHER
        start = -1
        for idx, predicted_tag in enumerate(tags):
            if "I-" in predicted_tag:
                continue
            # Either something has terminated, or nothing is happening
            if predicted_tag == "O":
                # Something has terminated
                if mode == Constants.BMISC:
                    results["MISC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BPER:
                    results["PER"].append(f"{start}-{idx-1}")
                elif mode == Constants.BLOC:
                    results["LOC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BORG:
                    results["ORG"].append(f"{start}-{idx-1}")
                mode = Constants.OTHER
            # Transition here from another B or from O or from I
            elif predicted_tag == "B-MISC":
                if mode == Constants.BMISC:
                    results["MISC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BPER:
                    results["PER"].append(f"{start}-{idx-1}")
                elif mode == Constants.BLOC:
                    results["LOC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BORG:
                    results["ORG"].append(f"{start}-{idx-1}")
                mode = Constants.BMISC
                start = idx
            elif predicted_tag == "B-PER":
                if mode == Constants.BMISC:
                    results["MISC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BPER:
                    results["PER"].append(f"{start}-{idx-1}")
                elif mode == Constants.BLOC:
                    results["LOC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BORG:
                    results["ORG"].append(f"{start}-{idx-1}")
                mode = Constants.BPER
                start = idx
            elif predicted_tag == "B-LOC":
                if mode == Constants.BMISC:
                    results["MISC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BPER:
                    results["PER"].append(f"{start}-{idx-1}")
                elif mode == Constants.BLOC:
                    results["LOC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BORG:
                    results["ORG"].append(f"{start}-{idx-1}")
                mode = Constants.BLOC
                start = idx
            elif predicted_tag == "B-ORG":
                if mode == Constants.BMISC:
                    results["MISC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BPER:
                    results["PER"].append(f"{start}-{idx-1}")
                elif mode == Constants.BLOC:
                    results["LOC"].append(f"{start}-{idx-1}")
                elif mode == Constants.BORG:
                    results["ORG"].append(f"{start}-{idx-1}")
                mode = Constants.BORG
                start = idx
        return results

    def write_results_to_file(
        results: Dict[str, List[str]],
        output_file: str = "../output/output.txt"
    ) -> None:
        with open(output_file, 'w') as f:
            f.write("Type,Prediction\n")
            f.write("PER,")
            f.write(" ".join(results["PER"]))
            f.write("\n")
            f.write("LOC,")
            f.write(" ".join(results["LOC"]))
            f.write("\n")
            f.write("ORG,")
            f.write(" ".join(results["ORG"]))
            f.write("\n")
            f.write("MISC,")
            f.write(" ".join(results["MISC"]))
            f.write("\n")
