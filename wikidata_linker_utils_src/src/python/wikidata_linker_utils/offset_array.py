from os.path import exists
import numpy as np
from .successor_mask import (
    convert_to_offset_array, make_dense, make_sparse
)


def count_non_zero(dense):
    return len(np.nonzero(dense[1:] - dense[:-1])[0]) + int(dense[0] != 0)


def should_compress(dense):
    nonzeros = count_non_zero(dense)
    return (2 * nonzeros + 1) < 0.5 * len(dense)


class OffsetArray(object):
    def __init__(self, values, offsets):
        self.values = values
        self.offsets = offsets

    def __getitem__(self, idx):
        end = self.offsets[idx]
        start = 0 if idx == 0 else self.offsets[idx - 1]
        return self.values[start:end]

    def is_empty(self, idx):
        end = self.offsets[idx]
        start = 0 if idx == 0 else self.offsets[idx - 1]
        return start == end

    def size(self):
        return self.offsets.shape[0]

    def edges(self):
        num_edges = np.zeros(len(self.offsets), dtype=np.int32)
        num_edges[0] = self.offsets[0]
        num_edges[1:] = self.offsets[1:] - self.offsets[:-1]
        return num_edges

    @classmethod
    def load(cls, path, compress=True):
        values = np.load(path + "_values.npy")
        if exists(path + "_offsets.sparse.npy"):
            offsets_compressed = np.load(path + "_offsets.sparse.npy")
            offsets = make_dense(offsets_compressed, cumsum=True)
        else:
            # legacy mode, load dense versions:
            offsets = np.load(path + "_offsets.npy")
            if compress:
                if should_compress(offsets):
                    offsets_compressed = make_sparse(offsets)
                    np.save(path + "_offsets.sparse.npy", offsets_compressed)
            # optionally delete the old version here
        return OffsetArray(
            values,
            offsets
        )


def convert_dict_to_offset_array(dictionary, num_values):
    offsets = np.zeros(num_values, dtype=np.int32)
    total_num_values = sum(len(v) for _, v in dictionary.items())
    values = np.zeros(total_num_values, dtype=np.int32)
    position = 0
    for key, value in sorted(dictionary.items(), key=lambda x: x[0]):
        values[position:position + len(value)] = value
        position += len(value)
        offsets[key] = len(value)
    np.cumsum(offsets, out=offsets)
    return values, offsets


def save_record_with_offset(path, index2indices, total_size=None):
    if isinstance(index2indices, dict):
        if total_size is None:
            raise ValueError("cannot leave total_size None "
                             "when using a dict.")
        values, offsets = convert_dict_to_offset_array(index2indices, total_size)
    else:
        values, offsets = convert_to_offset_array(index2indices)
    np.save(path + "_values.npy", values)
    if should_compress(offsets):
        compressed_offsets = make_sparse(offsets)
        np.save(path + "_offsets.sparse.npy", compressed_offsets)
    else:
        np.save(path + "_offsets.npy", offsets)


def load_sparse(path):
    compressed = np.load(path)
    dense = make_dense(compressed, cumsum=False)
    non_zero_indices = compressed[1::2]
    mask = np.zeros(len(dense), dtype=np.bool)
    mask[non_zero_indices] = True
    return dense, mask


class SparseAttribute(object):
    def __init__(self, dense, mask):
        self.dense = dense
        self.mask = mask

    def __lt__(self, value):
        return np.logical_and(self.dense < value, self.mask)

    def __le__(self, value):
        return np.logical_and(self.dense <= value, self.mask)

    def __gt__(self, value):
        return np.logical_and(self.dense > value, self.mask)

    def __ge__(self, value):
        return np.logical_and(self.dense >= value, self.mask)

    def __eq__(self, value):
        return np.logical_and(self.dense == value, self.mask)

    @classmethod
    def load(cls, path):
        dense, mask = load_sparse(path + "_values.sparse.npy")
        return SparseAttribute(dense, mask)
