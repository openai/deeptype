import numpy as np
import subprocess
import h5py
import ciseau
from os.path import exists, splitext, join
from wikidata_linker_utils.wikidata_ids import load_wikidata_ids

def count_examples(lines, comment, ignore_value, column_indices):
    example_length = 0
    has_labels = False
    found = 0
    for line in lines:
        if len(line) == 0 or (comment is not None and line.startswith(comment)):
            if example_length > 0 and has_labels:
                found += 1
            example_length = 0
            has_labels = False
        else:
            example_length += 1
            if not has_labels:
                cols = line.split("\t")
                if len(cols) > 1:
                    if ignore_value is not None:
                        for col_index in column_indices:
                            if cols[col_index] != ignore_value:
                                has_labels = True
                                break

                    else:
                        has_labels = True
    if example_length > 0 and has_labels:
        found += 1
    return found


def retokenize_example(x, y):
    tokens = ciseau.tokenize(" ".join(w for w in x),
                             normalize_ascii=False)
    out_y = []
    regular_cursor = 0
    tokens_length_total = 0
    regular_length_total = len(x[regular_cursor]) + 1 if len(x) > 0 else 0
    if regular_cursor + 1 == len(x):
        regular_length_total -= 1
    for i in range(len(tokens)):
        tokens_length_total = tokens_length_total + len(tokens[i])
        while regular_length_total < tokens_length_total:
            regular_cursor += 1
            regular_length_total = regular_length_total + len(x[regular_cursor]) + 1
            if regular_cursor + 1 == len(x):
                regular_length_total -= 1
        out_y.append(y[regular_cursor])
    assert(regular_cursor + 1 == len(x)), "error with %r" % (x,)
    return ([tok.rstrip() for tok in tokens], out_y)


def convert_lines_to_examples(lines, comment, ignore_value,
                              column_indices, x_column, empty_column,
                              retokenize=False):
    examples = []
    x = []
    y = []
    for line in lines:
        if len(line) == 0 or (comment is not None and line.startswith(comment)):
            if len(x) > 0:
                if not all(row == empty_column for row in y):
                    examples.append((x, y))
                x = []
                y = []
        else:
            cols = line.split("\t")
            x.append(cols[x_column])
            if len(cols) == 1:
                y.append(empty_column)
            else:
                if ignore_value is not None:
                    y.append(
                        tuple(
                            cols[col_index] if col_index is not None and cols[col_index] != ignore_value else None
                            for col_index in column_indices
                        )
                    )
                else:
                    y.append(
                        tuple(
                            cols[col_index] if col_index is not None else None
                            for col_index in column_indices
                        )
                    )
    if len(x) > 0 and not all(row == empty_column for row in y):
        examples.append((x, y))
    if retokenize:
        examples = [retokenize_example(x, y) for x, y in examples]
    return examples


def load_tsv(path, x_column, y_columns, objective_names, comment, ignore_value,
             retokenize):
    """"
    Deprecated method for loading a tsv file as a training/test set for a model.

    Arguments:
    ----------
        path: str, location of tsv file
        x_column: int
        y_columns: list<dict>, objectives in this file along with their column.
            (e.g. `y_columns=[{"objective": "POS", "column": 2}, ...])`)
        objective_names: name of all desired columns
        comment: line beginning indicating it's okay to skip
        ignore_value: label value that should be treated as missing
        retokenize: run tokenizer again.
    Returns
    -------
        list<tuple> : examples loaded into memory

    Note: can use a lot of memory since entire file is loaded.
    """
    objective2column = {col['objective']: col['column'] for col in y_columns}
    column_indices = [objective2column.get(name, None) for name in objective_names]
    empty_column = tuple(None for _ in objective_names)

    if all(col_index is None for col_index in column_indices):
        return []

    with open(path, "rt") as fin:
        lines = fin.read().splitlines()

    return convert_lines_to_examples(lines,
                                     ignore_value=ignore_value,
                                     empty_column=empty_column,
                                     x_column=x_column,
                                     column_indices=column_indices,
                                     comment=comment,
                                     retokenize=retokenize)


class RandomizableDataset(object):
    def set_rng(self, rng):
        self.rng = rng

    def set_randomize(self, randomize):
        self.randomize = randomize

    def set_ignore_y(self, ignore):
        self.ignore_y = ignore

class TSVDataset(RandomizableDataset):
    _fhandle = None
    _fhandle_position = 0
    _examples = None
    _example_indices = None
    _example_index = 0
    _eof = False
    ignore_y = False
    def __init__(self, path, x_column, y_columns, objective_names, comment, ignore_value,
                 retokenize=False, chunksize=50000000, randomize=False, rng=None):
        """"
        Arguments:
        ----------
            path: str, location of tsv file
            x_column: int
            y_columns: list<dict>, objectives in this file along with their column.
                (e.g. `y_columns=[{"objective": "POS", "column": 2}, ...])`)
            objective_names: name of all desired columns
            comment: line beginning indicating it's okay to skip
            ignore_value: label value that should be treated as missing
            chunksize: how many bytes to read from the file at a time.
            rng: numpy RandomState
            retokenize: run tokenizer on x again.
        """
        self.path = path
        self.randomize = randomize
        self.x_column = x_column
        self.y_columns = y_columns
        self.objective_names = objective_names
        self.comment = comment
        self.ignore_value = ignore_value
        self.retokenize = retokenize
        self.chunksize = chunksize
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng
        # column picking setup:
        objective2column = {col['objective']: col['column'] for col in y_columns}
        self.column_indices = [objective2column.get(name, None) for name in objective_names]
        self.empty_column = tuple(None for _ in objective_names)
        if all(col_index is None for col_index in self.column_indices):
            self.length = 0
        else:
            self._compute_length()

    def _signature(self):
        try:
            file_sha1sum = subprocess.check_output(
                ["sha1sum", self.path], universal_newlines=True
            ).split(" ")[0]
        except FileNotFoundError:
            file_sha1sum = subprocess.check_output(
                ["shasum", self.path], universal_newlines=True
            ).split(" ")[0]
        sorted_cols = list(
            map(
                str,
                sorted(
                    [col for col in self.column_indices if col is not None]
                )
            )
        )
        return "-".join([file_sha1sum] + sorted_cols)

    def _compute_length(self):
        length_file = (
            splitext(self.path)[0] +
            "-length-" +
            self._signature() + ".txt"
        )
        if exists(length_file):
            with open(length_file, "rt") as fin:
                total = int(fin.read())
        else:
            total = 0
            while True:
                total += self._count_examples()
                if self._eof:
                    break
            with open(length_file, "wt") as fout:
                fout.write(str(total) + "\n")
        self.length = total

    def __len__(self):
        return self.length

    def close(self):
        if self._fhandle is not None:
            self._fhandle.close()
            self._fhandle = None
        self._fhandle_position = 0
        self._eof = False
        self._examples = None
        self._example_indices = None

    def __del__(self):
        self.close()

    def _read_file_until_newline(self):
        if self._fhandle is None:
            self._fhandle = open(self.path, "rb")
        if self._eof:
            self._fhandle_position = 0
            self._fhandle.seek(0)
            self._eof = False

        read_chunk = None
        while True:
            new_read_chunk = self._fhandle.read(self.chunksize)
            if read_chunk is None:
                read_chunk = new_read_chunk
            else:
                read_chunk += new_read_chunk
            if len(new_read_chunk) < self.chunksize:
                del new_read_chunk
                self._fhandle_position += len(read_chunk)
                self._eof = True
                break
            else:
                del new_read_chunk
                newline_pos = read_chunk.rfind(b"\n\n")
                if newline_pos != -1:
                    # move to last line end position (so that we don't get
                    # half an example.)
                    self._fhandle.seek(self._fhandle_position + newline_pos + 2)
                    self._fhandle_position += newline_pos + 2
                    read_chunk = read_chunk[:newline_pos]
                    break
        return read_chunk

    def _count_examples(self):
        read_chunk = self._read_file_until_newline()
        return count_examples(
            read_chunk.decode("utf-8").splitlines(),
            ignore_value=self.ignore_value,
            column_indices=self.column_indices,
            comment=self.comment
        )

    def _load_examples(self):
        read_chunk = self._read_file_until_newline()
        if self._examples is not None:
            del self._examples
        self._examples = convert_lines_to_examples(
            read_chunk.decode("utf-8").splitlines(),
            ignore_value=self.ignore_value,
            empty_column=self.empty_column,
            x_column=self.x_column,
            column_indices=self.column_indices,
            comment=self.comment,
            retokenize=self.retokenize
        )
        self._example_indices = np.arange(len(self._examples))
        if self.randomize:
            # access loaded data randomly:
            self.rng.shuffle(self._example_indices)
        self._example_index = 0

    def __getitem__(self, index):
        """Retrieve the next example (index is ignored)"""
        if index >= self.length:
            raise StopIteration()
        if self._example_indices is None or self._example_index == len(self._example_indices):
            self._load_examples()
        while len(self._examples) == 0:
            self._load_examples()
            if len(self._examples) > 0:
                break
            if self._eof:
                raise StopIteration()
        ex = self._examples[self._example_indices[self._example_index]]
        self._example_index += 1
        return ex

    def set_randomize(self, randomize):
        if randomize != self.randomize:
            self.randomize = randomize

    def close(self):
        if self._fhandle is not None:
            self._fhandle.close()
            self._fhandle = None


class OracleClassification(object):
    def __init__(self, classes, classification, path):
        self.classes = classes
        self.classification = classification
        self.path = path
        self.contains_other = self.classes[-1] == "other"

    def classify(self, index):
        return self.classification[index]

def load_oracle_classification(path):
    with open(join(path, "classes.txt"), "rt", encoding="UTF-8") as fin:
        classes = fin.read().splitlines()
    classification = np.load(join(path, "classification.npy"))
    return OracleClassification(classes, classification, path)



class ClassificationHandler(object):
    def __init__(self, wikidata_path, classification_path):
        self.classification_path = classification_path
        _, self.name2index = load_wikidata_ids(wikidata_path, verbose=False)
        self.classifiers = {}

    def get_classifier(self, name):
        if name not in self.classifiers:
            self.classifiers[name] = load_oracle_classification(
                join(self.classification_path, name)
            )
        return self.classifiers[name]


class H5Dataset(RandomizableDataset):
    handle_open = False
    ignore_y = False
    _max_generated_example = 0
    _min_generated_example = 0
    def __init__(self, path, x_column, y_columns, objective_names,
                 classifications, ignore_value, randomize=False, rng=None):
        self.x_column = str(x_column)
        self.y_columns = y_columns
        self.ignore_value = ignore_value
        self.objective_names = objective_names
        self.randomize = randomize
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng
        self._classifications = classifications
        self.handle = h5py.File(path, "r")
        self.path = path
        self.handle_open = True
        self.length = len(self.handle[self.x_column])
        self.chunksize = self.handle[self.x_column].chunks[0]
        self._example_indices = None
        objective2column = {
            col['objective']: (
                str(col['column']),
                self._classifications.get_classifier(col['classification'])
            ) for col in y_columns
        }
        if self.ignore_value is not None:
            for _, classifier in objective2column.values():
                if self.ignore_value in classifier.classes:
                    classifier.classes[classifier.classes.index(self.ignore_value)] = None

        self.column2col_indices = {}
        for col_idx, name in enumerate(self.objective_names):
            if name not in objective2column:
                continue
            column, classifier = objective2column[name]
            if column not in self.column2col_indices:
                self.column2col_indices[column] = [(classifier, col_idx)]
            else:
                self.column2col_indices[column].append((classifier, col_idx))

    def close(self):
        if self.handle_open:
            self.handle.close()
            self.handle_open = False

    def __del__(self):
        self.close()

    def __len__(self):
        return self.length

    def _build_examples(self, index):
        x = [x_chunk.split("\n") for x_chunk in self.handle[self.x_column][index:index + self.chunksize]]
        y = [[[None for k in range(len(self.objective_names))] for j in range(len(x[i]))] for i in range(len(x))]
        if not self.ignore_y:
            for handle_column, col_content in self.column2col_indices.items():
                col_ids = [[self._classifications.name2index[name] if name != "" else None
                            for name in y_chunk.split("\n")]
                           for y_chunk in self.handle[handle_column][index:index + self.chunksize]]
                for i in range(len(col_ids)):
                    for j, idx in enumerate(col_ids[i]):
                        if idx is not None:
                            for classifier, k in col_content:
                                y[i][j][k] = classifier.classify(idx)

        return x, y

    def set_randomize(self, randomize):
        if self.randomize != randomize:
            self.randomize = randomize
            if self._max_generated_example != self._min_generated_example:
                self.xorder = np.arange(self._min_generated_example, self._max_generated_example)
                self.rng.shuffle(self.xorder)


    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration()
        if self.randomize:
            if self._example_indices is None or index == 0:
                self._example_indices = np.arange(0, len(self), self.chunksize)
                self.rng.shuffle(self._example_indices)
            # transformed index:
            index = (self._example_indices[index // self.chunksize] + (index % self.chunksize)) % len(self)

        if index < self._min_generated_example or index >= self._max_generated_example:
            self.x, self.y = self._build_examples(index)
            # store bounds of generated data:
            self._min_generated_example = index
            self._max_generated_example = index + len(self.x)

            if self.randomize:
                self.xorder = np.arange(self._min_generated_example, self._max_generated_example)
                self.rng.shuffle(self.xorder)
        if self.randomize:
            index = self.xorder[index - self._min_generated_example]
        return self.x[index - self._min_generated_example], self.y[index - self._min_generated_example]

class CombinedDataset(object):
    _which_dataset = None
    _dataset_counters = None
    def set_rng(self, rng):
        self.rng = rng
        for dataset in self.datasets:
            dataset.rng = rng

    def set_randomize(self, randomize):
        self.randomize = randomize
        for dataset in self.datasets:
            dataset.set_randomize(randomize)

    def set_ignore_y(self, ignore):
        for dataset in self.datasets:
            dataset.set_ignore_y(ignore)

    def close(self):
        for dataset in self.datasets:
            dataset.close()

    def _build_which_dataset(self):
        self._which_dataset = np.empty(self.length, dtype=np.int16)
        self._dataset_counters = np.zeros(len(self.datasets), dtype=np.int64)
        offset = 0
        for index, dataset in enumerate(self.datasets):
            # ensure each dataset is seen as much as its content
            # says:
            self._which_dataset[offset:offset + len(dataset)] = index
            offset += len(dataset)

    def __getitem__(self, index):
        if index == 0:
            if self.randomize:
                # visit datasets in random orders:
                self.rng.shuffle(self._which_dataset)
            self._dataset_counters[:] = 0
        which = self._which_dataset[index]
        idx = self._dataset_counters[which]
        self._dataset_counters[which] += 1
        return self.datasets[which][idx]

    def __init__(self, datasets, rng=None, randomize=False):
        self.datasets = datasets
        if rng is None:
            rng = np.random.RandomState(0)
        self.set_rng(rng)
        self.set_randomize(randomize)
        self.length = sum(len(dataset) for dataset in datasets)
        self._build_which_dataset()

    def __len__(self):
        return self.length
