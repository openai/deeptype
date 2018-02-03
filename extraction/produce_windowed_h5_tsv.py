import argparse

from os import remove
from wikidata_linker_utils.bash import execute_bash
import h5py


def produce_window_dataset(path, window_size, out):
    num_columns = 0
    with open(path, "rt") as fin:
        line_locations = []
        for idx, line in enumerate(fin):
            if "\t" in line:
                line_locations.append(idx)
                if num_columns == 0:
                    num_columns = len(line.split("\t"))
            if line == "\n":
                line_locations.append(-1)
    groups = []
    current_group = []

    max_buffer_size = 250000
    read_size = 100000
    seen_classes = {}

    for line_location in line_locations:
        if line_location == -1:
            if len(current_group) > 0:
                groups.append(current_group)
                current_group = []
        else:
            if len(current_group) == 0:
                current_group.append(line_location)
            elif abs(current_group[-1] - line_location) <= window_size:
                current_group.append(line_location)
            else:
                groups.append(current_group)
                current_group = [line_location]
    if len(current_group) > 0:
        groups.append(current_group)

    num_examples = len(groups)
    EMPTY = ""

    with h5py.File(out, "w") as handle:
        datasets = []
        for col in range(num_columns):
            datasets.append(
                handle.create_dataset(
                    str(col),
                    (num_examples,),
                    dtype=h5py.special_dtype(vlen=str),
                    chunks=(1500,)
                    # compression="gzip",
                    # compression_opts=9
                )
            )
        k = 0
        with open(path, "rt") as fin:
            current_location = 0
            current_lines = fin.readlines(read_size)
            current_end = current_location + len(current_lines)
            for group in groups:
                start = max(0, group[0] - window_size)
                end = group[-1] + window_size
                if end > current_end:
                    # read more lines into buffer:
                    current_lines = current_lines + fin.readlines(read_size)
                    # advance buffer max location
                    current_end = current_location + len(current_lines)
                    if len(current_lines) > max_buffer_size:
                        # compute how much to remove from buffer
                        to_chop = len(current_lines) - max_buffer_size
                        # move start location
                        current_location += to_chop
                        # remove extra buffer lines
                        current_lines = current_lines[to_chop:]
                # ensure that we do not cross white space boundaries
                start_delay = 0
                for idx, line in enumerate(current_lines[start - current_location:group[0] - current_location]):
                    if line == "\n":
                        start_delay = idx
                start += start_delay
                early_end = window_size
                for idx, line in enumerate(current_lines[group[-1] - current_location:end - current_location]):
                    if line == "\n":
                        early_end = idx
                        break
                end = group[-1] + early_end
                cols = [[] for i in range(num_columns)]
                for line in current_lines[start - current_location:end - current_location]:
                    vals = line.rstrip().split("\t")
                    for col_index in range(num_columns):
                        if len(vals) > col_index:
                            cols[col_index].append(vals[col_index])
                        else:
                            cols[col_index].append(EMPTY)
                for col_index, dataset in zip(cols, datasets):
                    dataset[k] = "\n".join(col_index)
                k += 1


def file_slice(path, start, end, destination, append):
    file_operator = ">>" if append else ">"
    delta = end - start
    command = "head -n %d %s | tail -n %d %s %s" % (
        end,
        path,
        delta,
        file_operator,
        destination
    )
    execute_bash(command)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("out_train")
    parser.add_argument("out_validation")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--total_size", type=int, required=True)
    parser.add_argument("--validation_start", type=int, required=True)
    parser.add_argument("--validation_size", type=int, default=500000)
    return parser.parse_args(args=args)


def main():
    args = parse_args()
    if args.total_size < args.validation_size:
        raise ValueError("cannot have total_size (%d) < validation_size "
                         "(%d)" % (args.total_size, args.validation_size))
    if args.validation_start > args.total_size:
        raise ValueError("cannot have validation_start (%d) begin after "
                         "total_size (%d)" % (args.validation_start, args.total_size))
    if args.validation_start + args.validation_size > args.total_size:
        raise ValueError("cannot have validation_start + validation_size (%d)"
                         " be larger than total_size (%d)" % (
            args.validation_start + args.validation_size, args.total_size
        ))
    train_temp = args.out_train + ".train_temp"
    try:
        file_slice(
            args.path,
            0,
            args.validation_start,
            train_temp,
            append=False
        )
        file_slice(
            args.path,
            args.validation_start + args.validation_size,
            args.total_size,
            train_temp,
            append=True
        )
        print("created temp file %s" % (train_temp))
        produce_window_dataset(
            train_temp, args.window_size, args.out_train
        )
        print("created windowed dataset for train")
    finally:
        print("removing temp file %s" % (train_temp))
        remove(train_temp)


    try:
        validation_temp = args.out_validation + ".validation_temp"
        file_slice(
            args.path,
            args.validation_start,
            args.validation_start + args.validation_size,
            validation_temp,
            append=False
        )
        print("created temp file %s" % (validation_temp))
        produce_window_dataset(validation_temp, args.window_size, args.out_validation)
        print("created windowed dataset for validation")
    finally:
        print("removing temp file %s" % (validation_temp))
        remove(validation_temp)


if __name__ == "__main__":
    main()
