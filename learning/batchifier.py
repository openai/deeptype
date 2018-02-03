import numpy as np
import string
from dataset import TSVDataset, H5Dataset, CombinedDataset
from generator import prefetch_generator

def word_dropout(inputs, rng, keep_prob):
    inputs_ndim = inputs.ndim
    mask_shape = [len(inputs)] + [1] * (inputs_ndim - 1)
    return (
        inputs *
        (
            rng.random_sample(size=mask_shape) <
            keep_prob
        )
    ).astype(inputs.dtype)


def extract_feat(feat):
    if feat["type"] == "word":
        return lambda x: x
    elif feat["type"] == "suffix":
        length = feat["length"]
        return lambda x: x[-length:]
    elif feat["type"] == "prefix":
        length = feat["length"]
        return lambda x: x[:length]
    elif feat["type"] == "digit":
        return lambda x: x.isdigit()
    elif feat["type"] == "punctuation_count":
        return lambda x: sum(c in string.punctuation for c in x)
    elif feat["type"] == "uppercase":
        return lambda x: len(x) > 0 and x[0].isupper()
    elif feat["type"] == "character-conv":
        max_size = feat["max_word_length"]
        def extract(x):
            x_bytes = x.encode("utf-8")
            if len(x_bytes) > max_size:
                return np.concatenate(
                    [
                        [255],
                        list(x_bytes[:max_size]),
                        [256]
                    ]
                )
            else:
                return np.concatenate(
                    [
                        [255],
                        list(x_bytes),
                        [256],
                        -np.ones(max_size - len(x_bytes), dtype=np.int32),
                    ]
                )
        return extract
    else:
        raise ValueError("unknown feature %r." % (feat,))


def extract_word_keep_prob(feat):
    return feat.get("word_keep_prob", 0.85)


def extract_case_keep_prob(feat):
    return feat.get("case_keep_prob", 0.95)


def extract_s_keep_prob(feat):
    return feat.get("s_keep_prob", 0.95)


def apply_case_s_keep_prob(feat, rng, keep_case, keep_s):
    if len(feat) == 0:
        return feat
    if keep_case < 1 and feat[0].isupper() and rng.random_sample() >= keep_case:
        feat = feat.lower()
    if keep_s < 1 and feat.endswith("s") and rng.random_sample() >= keep_s:
        feat = feat[:-1]
    return feat


def requires_character_convolution(feat):
    return feat["type"] in {"character-conv"}


def requires_vocab(feat):
    return feat["type"] in {"word", "suffix", "prefix"}


def feature_npdtype(feat):
    if requires_vocab(feat):
        return np.int32
    elif feat["type"] in {"digit", "punctuation_count", "uppercase"}:
        return np.float32
    elif requires_character_convolution(feat):
        return np.int32
    else:
        raise ValueError("unknown feature %r." % (feat,))


def get_vocabs(dataset, max_vocabs, extra_words=None):
    index2words = [[] for i in range(len(max_vocabs))]
    occurrences = [{} for i in range(len(max_vocabs))]
    for els in dataset:
        for el, index2word, occurrence in zip(els, index2words, occurrences):
            if el not in occurrence:
                index2word.append(el)
                occurrence[el] = 1
            else:
                occurrence[el] += 1
    index2words = [
        sorted(index2word, key=lambda x: occurrence[x], reverse=True)
        for index2word, occurrence in zip(index2words, occurrences)
    ]
    index2words = [
        index2word[:max_vocab] if max_vocab > 0 else index2word
        for index2word, max_vocab in zip(index2words, max_vocabs)
    ]
    if extra_words is not None:
        index2words = [
            extra_words + index2word for index2word in index2words
        ]
    return index2words


def get_feature_vocabs(features, dataset, extra_words=None):
    out, feats_needing_vocab, feats_with_vocabs, vocabs = [], [], [], []
    if hasattr(dataset, "set_ignore_y"):
        dataset.set_ignore_y(True)
    try:
        for feat in features:
            if requires_vocab(feat):
                if feat.get("path") is not None:
                    with open(feat["path"], "rt") as fin:
                        index2word = fin.read().splitlines()
                    if feat.get("max_vocab", -1) > 0:
                        index2word = index2word[:feat["max_vocab"]]
                    if extra_words is not None:
                        index2word = extra_words + index2word
                    feats_with_vocabs.append(index2word)
                else:
                    feats_needing_vocab.append(feat)
        if len(feats_needing_vocab) > 0:
            extractors = tuple(
                [extract_feat(feat) for feat in feats_needing_vocab]
            )
            vocabs = get_vocabs(
                ((extractor(w) for extractor in extractors)
                 for x, _ in dataset for w in x),
                max_vocabs=[feat.get("max_vocab", -1) for feat in feats_needing_vocab],
                extra_words=extra_words
            )
        vocab_feature_idx = 0
        preexisting_vocab_feature_idx = 0
        for feat in features:
            if requires_vocab(feat):
                if feat.get("path") is not None:
                    out.append(feats_with_vocabs[preexisting_vocab_feature_idx])
                    preexisting_vocab_feature_idx += 1
                else:
                    out.append(vocabs[vocab_feature_idx])
                    vocab_feature_idx+=1
            else:
                out.append(None)
    finally:
        if hasattr(dataset, "set_ignore_y"):
            dataset.set_ignore_y(False)
    return out


def pad_arrays_into_array(arrays, padding):
    out_ndim = arrays[0].ndim + 1
    out_shape = [0] * out_ndim
    out_shape[0] = len(arrays)
    for arr in arrays:
        for dim_idx in range(arr.ndim):
            out_shape[1 + dim_idx] = max(out_shape[1 + dim_idx], arr.shape[dim_idx])
    out = np.empty(out_shape, dtype=arrays[0].dtype)
    out.fill(padding)
    for arr_idx, array in enumerate(arrays):
        arr_slice = [arr_idx]
        for dim_idx in range(arr.ndim):
            arr_slice.append(slice(0, array.shape[dim_idx]))
        arr_slice = tuple(arr_slice)
        out[arr_slice] = array
    return out


def build_objective_mask(label_sequence, objective_idx, objective_type):
    if objective_type == 'crf':
        if len(label_sequence) == 0 or label_sequence[0][objective_idx] is None:
            return np.array(False, dtype=np.bool)
        else:
            return np.array(True, dtype=np.bool)
    elif objective_type == 'softmax':
        return np.array(
            [w[objective_idx] is not None for w in label_sequence], dtype=np.bool
        )
    else:
        raise ValueError(
            "unknown objective type %r." % (objective_type,)
        )


def allocate_shrunk_batches(max_length, batch_size, lengths):
    typical_indices = max_length * batch_size
    i = 0
    ranges = []
    while i < len(lengths):
        j = i + 1
        current_batch_size = 1
        longest_ex = lengths[j - 1]
        while j < len(lengths) and j - i < batch_size:
            # can grow?
            new_batch_size = current_batch_size + 1
            new_j = j + 1
            if max(longest_ex, lengths[new_j - 1]) * new_batch_size < typical_indices:
                j = new_j
                longest_ex = max(longest_ex, lengths[new_j - 1])
                current_batch_size = new_batch_size
            else:
                break
        ranges.append((i, j))
        i = j
    return ranges


def convert_label_to_index(label, label2index):
    if label is None:
        return 0
    if isinstance(label, str):
        return label2index[label]
    return label


class Batchifier(object):
    def __init__(self, rng, feature_word2index, objective_types, label2index,
                 fused, sequence_lengths, labels, labels_mask,
                 input_placeholders, features, dataset, batch_size, train,
                 autoresize=True, max_length=100):
        assert(batch_size > 0), (
            "batch size must be strictly positive (got %r)." % (batch_size,)
        )
        # dictionaries, strings defined by model:
        self.objective_types = objective_types
        self.label2index = label2index
        self.feature_word2index = feature_word2index
        self.rng = rng
        self.fused = fused

        # tf placeholders:
        self.sequence_lengths = sequence_lengths
        self.labels = labels
        self.labels_mask = labels_mask
        self.input_placeholders = input_placeholders

        self.dataset = dataset
        self.batch_size = batch_size
        self.train = train

        self.dataset_is_lazy = isinstance(dataset, (TSVDataset, H5Dataset, CombinedDataset))
        self.autoresize = autoresize
        self.max_length = max_length

        indices = np.arange(len(dataset))

        if train:
            if self.dataset_is_lazy:
                dataset.set_rng(rng)
                dataset.set_randomize(True)
            elif isinstance(dataset, list):
                rng.shuffle(indices)
        self.batch_indices = []
        if self.autoresize and not self.dataset_is_lazy:
            ranges = allocate_shrunk_batches(
                max_length=self.max_length,
                batch_size=self.batch_size,
                lengths=[len(dataset[indices[i]][0]) for i in range(len(indices))]
            )
            for i, j in ranges:
                self.batch_indices.append(indices[i:j])
        else:
            for i in range(0, len(indices), self.batch_size):
                self.batch_indices.append(indices[i:i + self.batch_size])
        self.extractors = [
            (extract_feat(feat), requires_vocab(feat), feature_npdtype(feat),
             extract_word_keep_prob(feat), extract_case_keep_prob(feat), extract_s_keep_prob(feat))
            for feat in features
        ]

    def generate_batch(self, examples):
        X = [[] for i in range(len(self.extractors))]
        Y = []
        Y_mask = []
        for ex, label in examples:
            for idx, (extractor, uses_vocab, dtype, word_keep_prob, case_keep_prob, s_keep_prob) in enumerate(self.extractors):
                if self.train and (case_keep_prob < 1 or s_keep_prob < 1):
                    ex = [apply_case_s_keep_prob(w, self.rng, case_keep_prob, s_keep_prob) for w in ex]
                if uses_vocab:
                    word_feats = np.array(
                        [self.feature_word2index[idx].get(extractor(w), 0) for w in ex],
                        dtype=dtype
                    )
                else:
                    word_feats = np.array([extractor(w) for w in ex], dtype=dtype)
                if self.train and word_keep_prob < 1:
                    word_feats = word_dropout(
                        word_feats, self.rng, word_keep_prob
                    )
                X[idx].append(word_feats)
            Y.append(
                tuple(
                    np.array([convert_label_to_index(w[objective_idx], label2index)
                              for w in label], dtype=np.int32)
                    for objective_idx, label2index in enumerate(self.label2index)
                )
            )

            Y_mask.append(
                tuple(
                    build_objective_mask(label, objective_idx, objective_type)
                    for objective_idx, objective_type in enumerate(self.objective_types)
                )
            )
        sequence_lengths = np.array([len(x) for x in X[0]], dtype=np.int32)
        X = [pad_arrays_into_array(x, -1) for x in X]
        Y = [
            pad_arrays_into_array([row[objective_idx] for row in Y], 0)
            for objective_idx in range(len(self.objective_types))
        ]
        Y_mask = [
            pad_arrays_into_array([row[objective_idx] for row in Y_mask], 0.0)
            for objective_idx in range(len(self.objective_types))
        ]
        feed_dict = {
            self.sequence_lengths: sequence_lengths
        }
        if self.fused:
            feed_dict[self.labels[0]] = np.stack([y.T for y in Y], axis=-1)
            feed_dict[self.labels_mask[0]] = np.stack([y.T for y in Y_mask], axis=-1)
        else:
            for y, placeholder in zip(Y, self.labels):
                feed_dict[placeholder] = y.T
            for y, placeholder in zip(Y_mask, self.labels_mask):
                feed_dict[placeholder] = y.T
        for idx, x in enumerate(X):
            feed_dict[self.input_placeholders[idx]] = x.swapaxes(0, 1)
        return feed_dict

    def as_list(self):
        return list(self.iter_batches())

    def iter_batches(self, pbar=None):
        gen = range(len(self.batch_indices))
        if pbar is not None:
            pbar.max_value = len(self.batch_indices)
            pbar.value = 0
            gen = pbar(gen)
        if self.autoresize and self.dataset_is_lazy:
            for idx in gen:
                examples = [self.dataset[ex] for ex in self.batch_indices[idx]]
                ranges = allocate_shrunk_batches(
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                    lengths=[len(ex[0]) for ex in examples]
                )
                for i, j in ranges:
                    yield self.generate_batch(examples[i:j])
        else:
            for idx in gen:
                yield self.generate_batch(
                    [self.dataset[ex] for ex in self.batch_indices[idx]]
                )


def allocate_shrunk_batches(max_length, batch_size, lengths):
    typical_indices = max_length * batch_size
    i = 0
    ranges = []
    while i < len(lengths):
        j = i + 1
        current_batch_size = 1
        longest_ex = lengths[j - 1]
        while j < len(lengths) and j - i < batch_size:
            # can grow?
            new_batch_size = current_batch_size + 1
            new_j = j + 1
            if max(longest_ex, lengths[new_j - 1]) * new_batch_size < typical_indices:
                j = new_j
                longest_ex = max(longest_ex, lengths[new_j - 1])
                current_batch_size = new_batch_size
            else:
                break
        ranges.append((i, j))
        i = j
    return ranges



def batch_worker(rng,
                 features,
                 feature_word2index,
                 objective_types,
                 label2index,
                 fused,
                 sequence_lengths,
                 labels,
                 labels_mask,
                 input_placeholders,
                 autoresize,
                 train,
                 batch_size,
                 max_length,
                 dataset,
                 pbar,
                 batch_queue,
                 death_event):
    batchifier = Batchifier(
         rng=rng,
         features=features,
         feature_word2index=feature_word2index,
         objective_types=objective_types,
         label2index=label2index,
         fused=fused,
         sequence_lengths=sequence_lengths,
         labels=labels,
         labels_mask=labels_mask,
         input_placeholders=input_placeholders,
         autoresize=autoresize,
         train=train,
         batch_size=batch_size,
         max_length=max_length,
         dataset=dataset
    )
    for batch in batchifier.iter_batches(pbar=pbar):
        if death_event.is_set():
            break
        batch_queue.put(batch)
    if not death_event.is_set():
        batch_queue.put(None)


def range_size(start, size):
    return [i for i in range(start, start + size)]


class ProcessHolder(object):
    def __init__(self, process, death_event, batch_queue):
        self.process = process
        self.batch_queue = batch_queue
        self.death_event = death_event

    def close(self):
        self.death_event.set()
        try:
            self.batch_queue.close()
            while True:
                self.batch_queue.get_nowait()
        except Exception as e:
            pass
        self.process.terminate()
        self.process.join()

    def __del__(self):
        self.close()


def iter_batches_single_threaded(model,
                                 dataset,
                                 batch_size,
                                 train,
                                 autoresize=True,
                                 max_length=100,
                                 pbar=None):
    tensorflow_placeholders = [model.sequence_lengths] + model.labels + model.labels_mask + model.input_placeholders
    labels_start = 1
    labels_mask_start = labels_start + len(model.labels)
    placeholder_start = labels_mask_start + len(model.labels_mask)
    batchifier = Batchifier(
         rng=model.rng,
         features=model.features,
         feature_word2index=model.feature_word2index,
         objective_types=[obj["type"] for obj in model.objectives],
         label2index=model.label2index,
         fused=model.fused,
         sequence_lengths=0,
         labels=range_size(labels_start, len(model.labels)),
         labels_mask=range_size(labels_mask_start, len(model.labels_mask)),
         input_placeholders=range_size(placeholder_start, len(model.input_placeholders)),
         autoresize=autoresize,
         train=train,
         batch_size=batch_size,
         max_length=max_length,
         dataset=dataset
    )
    for batch in prefetch_generator(batchifier.iter_batches(pbar=pbar), to_fetch=100):
        feed_dict = {}
        for idx, key in enumerate(tensorflow_placeholders):
            feed_dict[key] = batch[idx]
        yield feed_dict


def iter_batches(model,
                 dataset,
                 batch_size,
                 train,
                 autoresize=True,
                 max_length=100,
                 pbar=None):
    import multiprocessing
    batch_queue = multiprocessing.Queue(maxsize=10)
    tensorflow_placeholders = [model.sequence_lengths] + model.labels + model.labels_mask + model.input_placeholders
    labels_start = 1
    labels_mask_start = labels_start + len(model.labels)
    placeholder_start = labels_mask_start + len(model.labels_mask)
    death_event = multiprocessing.Event()
    batch_process = ProcessHolder(multiprocessing.Process(
        target=batch_worker,
        daemon=True,
        args=(
            model.rng,
            model.features,
            model.feature_word2index,
            [obj["type"] for obj in model.objectives],
            model.label2index,
            model.fused,
            0,
            range_size(labels_start, len(model.labels)),
            range_size(labels_mask_start, len(model.labels_mask)),
            range_size(placeholder_start, len(model.input_placeholders)),
            autoresize,
            train,
            batch_size,
            max_length,
            dataset,
            pbar,
            batch_queue,
            death_event
        )
    ), death_event, batch_queue)
    batch_process.process.name = "iter_batches"
    batch_process.process.start()
    while True:
        batch = batch_queue.get()
        if batch is None:
            break
        else:
            feed_dict = {}
            for idx, key in enumerate(tensorflow_placeholders):
                feed_dict[key] = batch[idx]
            yield feed_dict
        del batch
