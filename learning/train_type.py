import json
import pickle
import math
import sys
import argparse
import warnings

from os import makedirs
from os.path import basename, join, exists, dirname, splitext, realpath

from wikidata_linker_utils.progressbar import get_progress_bar
from dataset import TSVDataset, CombinedDataset, H5Dataset, ClassificationHandler
from batchifier import (iter_batches_single_threaded,
                        requires_vocab,
                        requires_character_convolution,
                        get_feature_vocabs)
import tensorflow as tf
import numpy as np

try:
    RNNCell = tf.nn.rnn_cell.RNNCell
    TFLSTMCell = tf.nn.rnn_cell.LSTMCell
    MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
    from tensorflow.contrib.cudnn_rnn import CudnnLSTM
except AttributeError:
    RNNCell = tf.contrib.rnn.RNNCell
    TFLSTMCell = tf.contrib.rnn.LSTMCell
    MultiRNNCell = tf.contrib.rnn.MultiRNNCell
    LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple
    from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import CudnnLSTM

from tensorflow.python.client import device_lib


class LazyAdamOptimizer(tf.train.AdamOptimizer):
    """Variant of the Adam optimizer that handles sparse updates more efficiently.

    The original Adam algorithm maintains two moving-average accumulators for
    each trainable variable; the accumulators are updated at every step.
    This class provides lazier handling of gradient updates for sparse variables.
    It only updates moving-average accumulators for sparse variable indices that
    appear in the current batch, rather than updating the accumulators for all
    indices. Compared with the original Adam optimizer, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original Adam algorithm, and
    may lead to different empirical results.
    """

    def _apply_sparse(self, grad, var):
        beta1_power = tf.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = tf.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m := beta1 * m + (1 - beta1) * g_t
        # We use a slightly different version of the moving-average update formula
        # that does a better job of handling concurrent lockless updates:
        # m -= (1 - beta1) * (m - g_t)
        m = self.get_slot(var, "m")
        m_t_delta = tf.gather(m, grad.indices) - grad.values
        m_t = tf.scatter_sub(m, grad.indices,
                                    (1 - beta1_t) * m_t_delta,
                                    use_locking=self._use_locking)

        # v := beta2 * v + (1 - beta2) * (g_t * g_t)
        # We reformulate the update as:
        # v -= (1 - beta2) * (v - g_t * g_t)
        v = self.get_slot(var, "v")
        v_t_delta = tf.gather(v, grad.indices) - tf.square(grad.values)
        v_t = tf.scatter_sub(v, grad.indices,
                                    (1 - beta2_t) * v_t_delta,
                                    use_locking=self._use_locking)

        # variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))
        m_t_slice = tf.gather(m_t, grad.indices)
        v_t_slice = tf.gather(v_t, grad.indices)
        denominator_slice = tf.sqrt(v_t_slice) + epsilon_t
        var_update = tf.scatter_sub(var, grad.indices,
                                    lr * m_t_slice / denominator_slice,
                                    use_locking=self._use_locking)
        return tf.group(var_update, m_t, v_t)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def split(values, axis, num_splits, name=None):
    return tf.split(values, num_splits, axis=axis, name=name)

def reverse(values, axis):
    return tf.reverse(values, [axis])


def sparse_softmax_cross_entropy_with_logits(logits, labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)


def concat(values, axis, name=None):
    if len(values) == 1:
        return values[0]
    return tf.concat(values, axis, name=name)


def concat_tensor_array(values, name=None):
    return values.stack(name=name)


def batch_gather_3d(values, indices):
    return tf.gather(tf.reshape(values, [-1, tf.shape(values)[2]]),
                     tf.range(0, tf.shape(values)[0]) * tf.shape(values)[1] +
                     indices)


def batch_gather_2d(values, indices):
    return tf.gather(tf.reshape(values, [-1]),
                     tf.range(0, tf.shape(values)[0]) * tf.shape(values)[1] +
                     indices)


def viterbi_decode(score, transition_params, sequence_lengths, back_prop=False,
                   parallel_iterations=1):
    """Decode the highest scoring sequence of tags inside of TensorFlow!!!
    This can be used anytime.
    Args:
        score: A [batch, seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
        sequence_lengths: A [batch] int32 vector of the length of each score
            sequence.
    Returns:
        viterbi: A [batch, seq_len] list of integers containing the highest
            scoring tag indices.
        viterbi_score: A vector of float containing the score for the Viterbi
            sequence.
    """
    sequence_lengths = tf.convert_to_tensor(
        sequence_lengths, name="sequence_lengths")
    score = tf.convert_to_tensor(score, name="score")
    transition_params = tf.convert_to_tensor(
        transition_params, name="transition_params")

    if sequence_lengths.dtype != tf.int32:
        sequence_lengths = tf.cast(sequence_lengths, tf.int32)

    def condition(t, *args):
        """Stop when full score sequence has been read in."""
        return tf.less(t, tf.shape(score)[1])

    def body(t, trellis, backpointers, trellis_val):
        """Perform forward viterbi pass."""
        v = tf.expand_dims(trellis_val, 2) + tf.expand_dims(transition_params, 0)
        new_trellis_val = score[:, t, :] + tf.reduce_max(v, axis=1)
        new_trellis = trellis.write(t, new_trellis_val)
        new_backpointers = backpointers.write(
            t, tf.cast(tf.argmax(v, axis=1), tf.int32))
        return t + 1, new_trellis, new_backpointers, new_trellis_val

    trellis_arr = tf.TensorArray(score.dtype, size=0,
        dynamic_size=True, clear_after_read=False, infer_shape=False)
    first_trellis_val = score[:, 0, :]
    trellis_arr = trellis_arr.write(0, first_trellis_val)

    backpointers_arr = tf.TensorArray(tf.int32, size=0,
        dynamic_size=True, clear_after_read=False, infer_shape=False)
    backpointers_arr = backpointers_arr.write(0,
        tf.zeros_like(score[:, 0, :], dtype=tf.int32))

    _, trellis_out, backpointers_out, _ = tf.while_loop(
        condition, body,
        (tf.constant(1, name="t", dtype=tf.int32), trellis_arr, backpointers_arr, first_trellis_val),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop)

    trellis_out = concat_tensor_array(trellis_out)
    backpointers_out = concat_tensor_array(backpointers_out)
    # make batch-major:
    trellis_out = tf.transpose(trellis_out, [1, 0, 2])
    backpointers_out = tf.transpose(backpointers_out, [1, 0, 2])

    def condition(t, *args):
        return tf.less(t, tf.shape(score)[1])

    def body(t, viterbi, last_decision):
        backpointers_timestep = batch_gather_3d(
            backpointers_out, tf.maximum(sequence_lengths - t, 0))
        new_last_decision = batch_gather_2d(
            backpointers_timestep, last_decision)
        new_viterbi = viterbi.write(t, new_last_decision)
        return t + 1, new_viterbi, new_last_decision

    last_timestep = batch_gather_3d(trellis_out, sequence_lengths - 1)
    # get scores for last timestep of each batch element inside
    # trellis:
    scores = tf.reduce_max(last_timestep, axis=1)
    # get choice index for last timestep:
    last_decision = tf.cast(tf.argmax(last_timestep, axis=1), tf.int32)

    # decode backwards using backpointers:
    viterbi = tf.TensorArray(tf.int32, size=0,
        dynamic_size=True, clear_after_read=False, infer_shape=False)
    viterbi = viterbi.write(0, last_decision)
    _, viterbi_out, _ = tf.while_loop(
        condition, body,
        (tf.constant(1, name="t", dtype=tf.int32), viterbi, last_decision),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop)
    viterbi_out = concat_tensor_array(viterbi_out)
    # make batch-major:
    viterbi_out = tf.transpose(viterbi_out, [1, 0])
    viterbi_out_fwd = tf.reverse_sequence(
        viterbi_out, sequence_lengths, seq_dim=1)
    return viterbi_out_fwd, scores


def sum_list(elements):
    total = elements[0]
    for el in elements[1:]:
        total += el
    return total


def explicitly_set_fields():
    received = set()
    for argument in sys.argv:
        if argument.startswith("--"):
            received.add(argument[2:])
            if argument[2:].startswith("no"):
                received.add(argument[4:])
    return received


def save_session(session, saver, path, verbose=False):
    """
    Call save on tf.train.Saver on a specific path to store all the variables
    of the current tensorflow session to a file for later restoring.

    Arguments:
        session : tf.Session
        path : str, place to save session
    """
    makedirs(path, exist_ok=True)
    if not path.endswith("/"):
        path = path + "/"

    path = join(path, "model.ckpt")
    if verbose:
        print("Saving session under %r" % (path,), flush=True)
    saver.save(session, path)
    print("Saved", flush=True)

### constants for saving & loading

# model config:
OBJECTIVE_NAMES = "OBJECTIVE_NAMES"
OBJECTIVE_TYPES = "OBJECTIVE_TYPES"

# inputs:
INPUT_PLACEHOLDERS = "INPUT_PLACEHOLDERS"
LABEL_PLACEHOLDERS = "LABEL_PLACEHOLDERS"
LABEL_MASK_PLACEHOLDERS = "LABEL_MASK_PLACEHOLDERS"
TRAIN_OP = "TRAIN_OP"
SEQUENCE_LENGTHS = "SEQUENCE_LENGTHS"
IS_TRAINING = "IS_TRAINING"

# outputs:
DECODED = "DECODED"
DECODED_SCORES = "DECODED_SCORES"
UNARY_SCORES = "UNARY_SCORES"

# per objective metrics:
TOKEN_CORRECT = "TOKEN_CORRECT"
TOKEN_CORRECT_TOTAL = "TOKEN_CORRECT_TOTAL"
SENTENCE_CORRECT = "SENTENCE_CORRECT"
SENTENCE_CORRECT_TOTAL = "SENTENCE_CORRECT_TOTAL"

# aggregate metrics over all objectives
NLL = "NLL"
NLL_TOTAL = "NLL_TOTAL"
TOKEN_CORRECT_ALL = "TOKEN_CORRECT_ALL"
TOKEN_CORRECT_ALL_TOTAL = "TOKEN_CORRECT_ALL_TOTAL"
SENTENCE_CORRECT_ALL = "SENTENCE_CORRECT_ALL"
SENTENCE_CORRECT_ALL_TOTAL = "SENTENCE_CORRECT_ALL_TOTAL"
CONFUSION_MATRIX = "CONFUSION_MATRIX"
GLOBAL_STEP = "global_step"
SUMMARIES_ASSIGNS = "SUMMARIES_ASSIGNS"
SUMMARIES_PLACEHOLDERS = "SUMMARIES_PLACEHOLDERS"
SUMMARIES_NAMES = "SUMMARIES_NAMES"
TRAIN_SUMMARIES = "TRAIN_SUMMARIES"

TRUE_POSITIVES = "TRUE_POSITIVES"
FALSE_POSITIVES = "FALSE_POSITIVES"
FALSE_NEGATIVES = "FALSE_NEGATIVES"

def maybe_dropout(inputs, keep_prob, is_training):
    return tf.cond(is_training,
        lambda : tf.nn.dropout(inputs, keep_prob),
        lambda : inputs
    ) if keep_prob < 1 else inputs


def compute_sentence_correct(correct, sequence_mask):
    any_label = tf.reduce_max(tf.cast(sequence_mask, tf.int32), 1)
    sentence_correct_total = tf.reduce_sum(any_label)
    # is 1 when all is correct, 0 otherwise
    sentence_correct = tf.reduce_sum(tf.reduce_prod(
        tf.cast(
            tf.logical_or(correct, tf.logical_not(sequence_mask)),
            tf.int32
        ),
        1
    ) * any_label)
    return sentence_correct, sentence_correct_total


def lstm_activation(inputs, input_h, input_c, W, b, activation):
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    cell_inputs = concat([inputs, input_h], axis=1)

    lstm_matrix = tf.nn.xw_plus_b(cell_inputs, W, b)
    preactiv = split(lstm_matrix, axis=1, num_splits=4)
    # from CUDNN docs:
    # Values 0 and 4 reference the input gate.
    # Values 1 and 5 reference the forget gate.
    # Values 2 and 6 reference the new memory gate.
    # Values 3 and 7 reference the output gate
    i, f, j, o = (
        preactiv[CUDNN_MAPPING["i"]],
        preactiv[CUDNN_MAPPING["f"]],
        preactiv[CUDNN_MAPPING["j"]],
        preactiv[CUDNN_MAPPING["o"]]
    )

    c = (tf.nn.sigmoid(f) * input_c +
         tf.nn.sigmoid(i) * activation(j))

    m = tf.nn.sigmoid(o) * activation(c)
    return (c, m)


class Logger(object):
    def __init__(self, session, writer):
        self.session = session
        self.writer = writer
        self._placeholders = {}
        summaries = tf.get_collection(SUMMARIES_ASSIGNS)
        summaries_pholders = tf.get_collection(SUMMARIES_PLACEHOLDERS)
        summaries_names = [name.decode("utf-8")
                           for name in tf.get_collection(SUMMARIES_NAMES)]

        for summary, pholder, name in zip(summaries, summaries_pholders, summaries_names):
            self._placeholders[name] = (pholder, summary)


    def log(self, name, value, step):
        if name not in self._placeholders:
            pholder = tf.placeholder(tf.float32, [], name=name)
            summary = tf.summary.scalar(name, pholder)
            tf.add_to_collection(SUMMARIES_ASSIGNS, summary)
            tf.add_to_collection(SUMMARIES_NAMES, name)
            tf.add_to_collection(SUMMARIES_PLACEHOLDERS, pholder)
            self._placeholders[name] = (pholder, summary)
        pholder, summary = self._placeholders[name]
        res = self.session.run(summary, {pholder:value})
        self.writer.add_summary(res, step)


class ParametrizedLSTMCell(RNNCell):
    def __init__(self, weights, biases, hidden_size):
        self._weights = weights
        self._biases = biases
        self.hidden_size = hidden_size

    @property
    def state_size(self):
        return (self.hidden_size, self.hidden_size)

    @property
    def output_size(self):
        return self.hidden_size

    def __call__(self, inputs, state, scope=None):
        input_h, input_c = state
        c, m = lstm_activation(inputs,
                               input_h=input_h,
                               input_c=input_c,
                               b=self._biases,
                               W=self._weights,
                               activation=tf.nn.tanh)
        return m, (m, c)


class LSTMCell(TFLSTMCell):
    def __init__(self,
                 num_units,
                 keep_prob=1.0,
                 is_training=False):
        self._is_training = is_training
        self._keep_prob = keep_prob
        TFLSTMCell.__init__(
            self,
            num_units=num_units,
            state_is_tuple=True
        )

    def __call__(self, inputs, state, scope=None):
        (c_prev, m_prev) = state

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with tf.variable_scope(scope or type(self).__name__,
                               initializer=self._initializer):  # "LSTMCell"
            concat_w = _get_concat_variable(
                    "W", [input_size.value + self._num_units, 4 * self._num_units],
                    dtype, 1)

            b = tf.get_variable(
                    "B", shape=[4 * self._num_units],
                    initializer=tf.zeros_initializer(), dtype=dtype)

        c, m = lstm_activation(inputs,
                               input_c=c_prev,
                               input_h=m_prev,
                               W=concat_w,
                               b=b,
                               activation=self._activation,
                               keep_prob=self._keep_prob,
                               is_training=self._is_training,
                               forget_bias=self._forget_bias)
        return m, LSTMStateTuple(c, m)



def cudnn_lstm_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 8 * hidden_size
    weights = 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
    return biases + weights


def direction_to_num_directions(direction):
    if direction == "unidirectional":
        return 1
    elif direction == "bidirectional":
        return 2
    else:
        raise ValueError("Unknown direction: %r." % (direction,))


def estimate_cudnn_parameter_size(num_layers,
                                  input_size,
                                  hidden_size,
                                  input_mode,
                                  direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. Assumes the hidden states
    of bidirectional LSTMs are concatenated before being
    sent to the next layer up.
    """
    num_directions = direction_to_num_directions(direction)
    params = 0
    isize = input_size
    for layer in range(num_layers):
        for direction in range(num_directions):
            params += cudnn_lstm_parameter_size(
                isize, hidden_size
            )
        isize = hidden_size * num_directions
    return params

# cudnn conversion to dynamic RNN:
CUDNN_LAYER_WEIGHT_ORDER = [
    "x", "x", "x", "x", "h", "h", "h", "h"
]
CUDNN_LAYER_BIAS_ORDER = [
    "bx", "bx", "bx", "bx", "bh", "bh", "bh", "bh"
]
CUDNN_TRANSPOSED = True
CUDNN_MAPPING = {"i": 0, "f": 1, "j": 2, "o": 3}


def consume_biases_direction(params, old_offset, hidden_size, isize):
    offset = old_offset
    layer_biases_x = []
    layer_biases_h = []

    for piece in CUDNN_LAYER_BIAS_ORDER:
        if piece == "bx":
            layer_biases_x.append(
                params[offset:offset + hidden_size]
            )
            offset += hidden_size
        elif piece == "bh":
            layer_biases_h.append(
                params[offset:offset + hidden_size]
            )
            offset += hidden_size
        else:
            raise ValueError("Unknown cudnn piece %r." % (piece,))
    b = concat(layer_biases_x, axis=0) + concat(layer_biases_h, axis=0)
    return b, offset


def consume_weights_direction(params, old_offset, hidden_size, isize):
    offset = old_offset
    layer_weights_x = []
    layer_weights_h = []
    for piece in CUDNN_LAYER_WEIGHT_ORDER:
        if piece == "x":
            layer_weights_x.append(
                tf.reshape(
                    params[offset:offset + hidden_size * isize],
                    [hidden_size, isize] if CUDNN_TRANSPOSED else [isize, hidden_size]
                )
            )
            offset += hidden_size * isize
        elif piece == "h":
            layer_weights_h.append(
                tf.reshape(
                    params[offset:offset + hidden_size * hidden_size],
                    [hidden_size, hidden_size]
                )
            )
            offset += hidden_size * hidden_size
        else:
            raise ValueError("Unknown cudnn piece %r." % (piece,))
    if CUDNN_TRANSPOSED:
        W_T = concat([concat(layer_weights_x, axis=0), concat(layer_weights_h, axis=0)], axis=1)
        W = tf.transpose(W_T)
    else:
        W = concat([concat(layer_weights_x, axis=1), concat(layer_weights_h, axis=1)], axis=0)
    return W, offset


def decompose_layer_params(params, num_layers,
                           hidden_size, cell_input_size,
                           input_mode, direction, create_fn):
    """
    This operation converts the opaque cudnn params into a set of
    usable weight matrices.
    Args:
        params : Tensor, opaque cudnn params tensor
        num_layers : int, number of stacked LSTMs.
        hidden_size : int, number of neurons in each LSTM.
        cell_input_size : int, input size for the LSTMs.
        input_mode: whether a pre-projection was used or not. Currently only
            'linear_input' is supported (e.g. CuDNN does its own projection
            internally)
        direction : str, 'unidirectional' or 'bidirectional'.
        create_fn: callback for weight creation. Receives parameter slice (op),
                   layer (int), direction (0 = fwd, 1 = bwd),
                   parameter_index (0 = W, 1 = b).
    Returns:
        weights : list of lists of Tensors in the format:
            first list is indexed layers,
            inner list is indexed by direction (fwd, bwd),
            tensors in the inner list are (Weights, biases)
    """
    if input_mode != "linear_input":
        raise ValueError("Only input_mode == linear_input supported for now.")
    num_directions = direction_to_num_directions(direction)
    offset = 0
    all_weights = [[[] for j in range(num_directions)]
                   for i in range(num_layers)]
    isize = cell_input_size
    with tf.variable_scope("DecomposeCudnnParams"):
        for layer in range(num_layers):
            with tf.variable_scope("Layer{}".format(layer)):
                for direction in range(num_directions):
                    with tf.variable_scope("fwd" if direction == 0 else "bwd"):
                        with tf.variable_scope("weights"):
                            W, offset = consume_weights_direction(
                                params,
                                old_offset=offset,
                                hidden_size=hidden_size,
                                isize=isize)
                            all_weights[layer][direction].append(
                                create_fn(W, layer, direction, 0))
            isize = hidden_size * num_directions
        isize = cell_input_size
        for layer in range(num_layers):
            with tf.variable_scope("Layer{}".format(layer)):
                for direction in range(num_directions):
                    with tf.variable_scope("fwd" if direction == 0 else "bwd"):
                        with tf.variable_scope("biases"):
                            b, offset = consume_biases_direction(
                                params,
                                old_offset=offset,
                                hidden_size=hidden_size,
                                isize=isize)
                            all_weights[layer][direction].append(
                                create_fn(b, layer, direction, 1))
            isize = hidden_size * num_directions
    return all_weights


def create_decomposed_variable(param, lidx, didx, pidx):
    with tf.device("cpu"):
        return tf.get_variable("w" if pidx == 0 else "b",
                               shape=param.get_shape().as_list(),
                               dtype=param.dtype,
                               trainable=False,
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                            "excluded_variables"])


def cpu_cudnn_params(params, num_layers, hidden_size, cell_input_size, input_mode,
                     direction):
    """
    This operation converts the opaque cudnn params into a set of
    usable weight matrices, and caches the conversion.
    Args:
        params : Tensor, opaque cudnn params tensor
        num_layers : int, number of stacked LSTMs.
        hidden_size : int, number of neurons in each LSTM.
        cell_input_size : int, input size for the LSTMs.
        input_mode: whether a pre-projection was used or not. Currently only
            'linear_input' is supported (e.g. CuDNN does its own projection
            internally)
        direction : str, 'unidirectional' or 'bidirectional'.
        skip_creation : bool, whether to build variables.
    Returns:
        weights : list of lists of Tensors in the format:
            first list is indexed layers,
            inner list is indexed by direction (fwd, bwd),
            tensors in the inner list are (Weights, biases)
    """
    # create a boolean status variable that checks whether the
    # weights have been converted to cpu format:
    with tf.device("cpu"):
        cpu_conversion_status = tf.get_variable(
            name="CudnnConversionStatus", dtype=tf.float32,
            initializer=tf.zeros_initializer(), shape=[],
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    # create a fresh copy of the weights (not trainable)
    reshaped = decompose_layer_params(
        params,
        num_layers=num_layers,
        hidden_size=hidden_size,
        cell_input_size=cell_input_size,
        input_mode=input_mode,
        direction=direction,
        create_fn=create_decomposed_variable)

    def cpu_convert():
        all_assigns = decompose_layer_params(
            params,
            num_layers=num_layers,
            hidden_size=hidden_size,
            cell_input_size=cell_input_size,
            input_mode=input_mode,
            direction=direction,
            create_fn=lambda p, lidx, didx, pidx: tf.assign(reshaped[lidx][didx][pidx], p))
        all_assigns = [assign for layer_assign in all_assigns
                       for dir_assign in layer_assign
                       for assign in dir_assign]
        all_assigns.append(tf.assign(cpu_conversion_status, tf.constant(1.0, dtype=tf.float32)))
        all_assigns.append(tf.Print(cpu_conversion_status, [0],
            message="Converted cudnn weights to CPU format. "))
        with tf.control_dependencies(all_assigns):
            ret = tf.identity(cpu_conversion_status)
            return ret
    # cache the reshaping/concatenating
    ensure_conversion = tf.cond(tf.greater(cpu_conversion_status, 0),
                                lambda: cpu_conversion_status,
                                cpu_convert)
    # if weights are already reshaped, go ahead:
    with tf.control_dependencies([ensure_conversion]):
        # wrap with identity to ensure there is a dependency between assignment
        # and using the weights:
        all_params = [[[tf.identity(p) for p in dir_param]
                       for dir_param in layer_param]
                      for layer_param in reshaped]
        return all_params


class CpuCudnnLSTM(object):
    def __init__(self, num_layers, hidden_size,
                 cell_input_size, input_mode, direction):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cell_input_size = cell_input_size
        self.input_mode = input_mode
        self.direction = direction

    def __call__(self,
                 inputs,
                 input_h,
                 input_c,
                 params,
                 is_training=True):
        layer_params = cpu_cudnn_params(params,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            cell_input_size=self.cell_input_size,
            input_mode=self.input_mode,
            direction=self.direction)
        REVERSED = 1
        layer_inputs = inputs
        cell_idx = 0
        for layer_param in layer_params:
            hidden_fwd_bwd = []
            final_output_c = []
            final_output_h = []
            for direction, (W, b) in enumerate(layer_param):
                if direction == REVERSED:
                    layer_inputs = reverse(layer_inputs, axis=0)
                hiddens, (output_h, output_c) = tf.nn.dynamic_rnn(
                    cell=ParametrizedLSTMCell(W, b, self.hidden_size),
                    inputs=layer_inputs,
                    dtype=inputs.dtype,
                    time_major=True,
                    initial_state=(input_h[cell_idx], input_c[cell_idx]))
                if direction == REVERSED:
                    hiddens = reverse(hiddens, axis=0)
                hidden_fwd_bwd.append(hiddens)
                final_output_c.append(tf.expand_dims(output_c, 0))
                final_output_h.append(tf.expand_dims(output_h, 0))
                cell_idx += 1
            if len(hidden_fwd_bwd) > 1:
                layer_inputs = concat(hidden_fwd_bwd, axis=2)
                final_output_c = concat(final_output_c, axis=0)
                final_output_h = concat(final_output_h, axis=0)
            else:
                layer_inputs = hidden_fwd_bwd[0]
                final_output_c = final_output_c[0]
                final_output_h = final_output_h[0]
        return layer_inputs, final_output_h, final_output_c


def highway(x, activation_fn=tf.nn.relu, scope=None):
    size = x.get_shape()[-1].value
    with tf.variable_scope(scope or "HighwayLayer"):
        activ = tf.contrib.layers.fully_connected(
            x, size * 2, activation_fn=None, scope="FC"
        )
        transform = tf.sigmoid(activ[..., :size], name="transform_gate")
        hidden = activation_fn(activ[..., size:])
        carry = 1.0 - transform
        return tf.add(hidden * transform, x * carry, "y")


def conv2d(inputs, output_dim, k_h, k_w,
           stddev=0.02, scope=None,
           weight_noise=0.0, is_training=True):
    with tf.variable_scope(scope or "Conv2D"):
         w = tf.get_variable('w', [k_h, k_w, inputs.get_shape()[-1], output_dim],
                   initializer=tf.truncated_normal_initializer(stddev=stddev))
         if weight_noise > 0 and not isinstance(is_training, bool):
            w = add_weight_noise(w, is_training=is_training, stddev=weight_noise)
         return tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="VALID")



def character_convolution(inputs, feature):
    inputs_2d = tf.reshape(inputs,
        [tf.shape(inputs)[0] * tf.shape(inputs)[1], tf.shape(inputs)[2]]
    )
    inputs_3d = embedding_lookup(
        inputs_2d,
        dim=feature["dimension"],
        # 255 different bytes (uint8)
        # & start and end symbol:
        size=257,
        dtype=tf.float32,
        mask_negative=True)
    inputs_4d = tf.expand_dims(inputs_3d, 1)
    feature_pools = []
    for idx, conv_filter in enumerate(feature["filters"]):
        width, channels = conv_filter["width"], conv_filter["channels"]
        # [batch * time x 1 x word_length x embed_dim x feature_map_dim]
        conv = tf.squeeze(conv2d(inputs_4d, channels, 1, width, scope="CharacterConvolution%d" % (idx,)), [1])
        # remove word dimension
        pool = tf.reduce_max(conv, 1)
        feature_pools.append(pool)
    activations = concat(feature_pools, axis=1)
    channels_out = sum(conv_filter["channels"] for conv_filter in feature["filters"])
    activations = tf.reshape(
        tf.tanh(activations),
        [tf.shape(inputs)[0], tf.shape(inputs)[1], channels_out],
        name="CharacterConvolutionPooled")
    for idx in range(feature["highway_layers"]):
        activations = highway(activations, scope="HighwayLayer%d" % (idx,),
            activation_fn=tf.tanh)
    return activations


def feature_dtype(feat):
    if requires_vocab(feat):
        return tf.int32
    elif feat["type"] in {"digit", "punctuation_count", "uppercase"}:
        return tf.float32
    elif requires_character_convolution(feat):
        return tf.int32
    else:
        raise ValueError("unknown feature %r." % (feat,))


def feature_shape(feature):
    if requires_vocab(feature) or feature["type"] in {'digit', 'punctuation_count', 'uppercase'}:
        return [None, None]
    elif requires_character_convolution(feature):
        return [None, None, None]
    else:
        raise ValueError("unknown feature %r." % (feature,))


def build_inputs(features, objectives, fused, class_weights,
                 class_weights_clipval):
    input_placeholders = []
    labels = []
    labels_mask = []
    labels_class_weights = []
    max_output_vocab = max(len(obj["vocab"]) for obj in objectives)

    with tf.variable_scope("Inputs"):
        is_training = tf.placeholder(tf.bool, [], name="is_training")
        tf.add_to_collection(IS_TRAINING, is_training)
        for idx, feat in enumerate(features):
            input_placeholder = tf.placeholder(
                feature_dtype(feat), feature_shape(feat),
                name="input_placeholders_%d" % (idx,)
            )
            input_placeholders.append(input_placeholder)
            tf.add_to_collection(INPUT_PLACEHOLDERS, input_placeholder)

        if fused:
            label_placeholder = tf.placeholder(
                tf.int32, [None, None, len(objectives)]
            )
            labels_mask_placeholder = tf.placeholder(
                tf.bool, [None, None,  len(objectives)], name="labels_mask"
            )

            labels.append(label_placeholder)
            labels_mask.append(labels_mask_placeholder)
            tf.add_to_collection(LABEL_PLACEHOLDERS, label_placeholder)
            tf.add_to_collection(LABEL_MASK_PLACEHOLDERS, labels_mask_placeholder)

            if class_weights:
                with tf.variable_scope("FusedClassWeights"):
                    init_class_weights = tf.get_variable(
                        name="class_weights",
                        shape=[len(objectives) * max_output_vocab],
                        initializer=tf.constant_initializer(1),
                        dtype=tf.int64,
                        trainable=False)
                    init_class_count = tf.get_variable(
                        name="class_weights_denominator",
                        shape=[len(objectives)],
                        initializer=tf.constant_initializer(1),
                        dtype=tf.int64,
                        trainable=False)

                    def update_class_weights():
                        mask_as_ints = tf.cast(tf.reshape(labels_mask_placeholder, [-1, len(objectives)]), tf.int64)
                        updated_cls_weights = tf.scatter_add(
                            init_class_weights,
                            tf.reshape(label_placeholder + tf.reshape(tf.range(len(objectives)) * max_output_vocab, [1, 1, len(objectives)]), [-1]),
                            tf.reshape(mask_as_ints, [-1])
                        )
                        updated_class_count = tf.assign_add(init_class_count, tf.reduce_sum(mask_as_ints, 0))

                        # class weight: weight_i = total / class_i
                        weights = tf.clip_by_value(tf.expand_dims(updated_class_count, 1) /
                                                   tf.reshape(updated_cls_weights, [len(objectives), max_output_vocab]),
                                                   1e-6, class_weights_clipval)
                        return tf.cast(weights, tf.float32)

                    def return_class_weights():
                        # class weight: weight_i = total / class_i
                        return tf.cast(
                            tf.clip_by_value(tf.expand_dims(init_class_count, 1) /
                                             tf.reshape(init_class_weights, [len(objectives), max_output_vocab]),
                                             1e-6, class_weights_clipval), tf.float32)

                    labels_class_weights.append(
                        tf.cond(is_training,
                                update_class_weights,
                                return_class_weights))
            else:
                labels_class_weights.append(None)
        else:
            for objective in objectives:
                with tf.variable_scope(objective["name"]):
                    label_placeholder = tf.placeholder(
                        tf.int32, [None, None], name="labels"
                    )
                    labels.append(label_placeholder)
                    if objective["type"] == "crf":
                        labels_mask_placeholder = tf.placeholder(
                            tf.bool, [None], name="labels_mask"
                        )
                        labels_class_weights.append(None)
                    elif objective["type"] == "softmax":
                        labels_mask_placeholder = tf.placeholder(
                            tf.bool, [None, None], name="labels_mask"
                        )
                        if class_weights:
                            init_class_weights = tf.get_variable(
                                name="class_weights",
                                shape=len(objective["vocab"]),
                                initializer=tf.constant_initializer(1),
                                dtype=tf.int64,
                                trainable=False)
                            init_class_count = tf.get_variable(
                                name="class_weights_denominator",
                                shape=[],
                                initializer=tf.constant_initializer(1),
                                dtype=tf.int64,
                                trainable=False)

                            def update_class_weights():
                                mask_as_ints = tf.cast(tf.reshape(labels_mask_placeholder, [-1]), tf.int64)
                                updated_cls_weights = tf.scatter_add(
                                    init_class_weights,
                                    tf.reshape(label_placeholder, [-1]),
                                    mask_as_ints
                                )
                                updated_class_count = tf.assign_add(init_class_count, tf.reduce_sum(mask_as_ints))

                                # class weight: weight_i = total / class_i
                                weights = tf.clip_by_value(updated_class_count / updated_cls_weights,
                                                           1e-6, class_weights_clipval)
                                return tf.cast(weights, tf.float32)

                            def return_class_weights():
                                # class weight: weight_i = total / class_i
                                return tf.cast(
                                    tf.clip_by_value(init_class_count / init_class_weights,
                                                     1e-6, class_weights_clipval), tf.float32)

                            labels_class_weights.append(
                                tf.cond(is_training, update_class_weights, return_class_weights)
                            )
                        else:
                            labels_class_weights.append(None)
                    else:
                        raise ValueError(
                            "unknown objective type %r." % (
                                objective["type"]
                            )
                        )
                    labels_mask.append(labels_mask_placeholder)
                    tf.add_to_collection(LABEL_PLACEHOLDERS, label_placeholder)
                    tf.add_to_collection(LABEL_MASK_PLACEHOLDERS, labels_mask_placeholder)
        sequence_lengths = tf.placeholder(tf.int32, [None],
                                          name="sequence_lengths")
        tf.add_to_collection(SEQUENCE_LENGTHS, sequence_lengths)
    return (input_placeholders,
            labels,
            labels_mask,
            labels_class_weights,
            sequence_lengths,
            is_training)


def add_weight_noise(x, is_training, stddev):
    return tf.cond(is_training,
                   lambda: x + tf.random_normal(
                       shape=tf.shape(x), stddev=stddev),
                   lambda: x)


def build_recurrent(inputs, cudnn, faux_cudnn, hidden_sizes, is_training,
                    keep_prob, weight_noise):
    dtype = tf.float32
    if cudnn:
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must be a list of length > 1.")
        hidden_size = hidden_sizes[0]
        if any(hidden_size != hsize for hsize in hidden_sizes):
            raise ValueError("cudnn RNN requires all hidden units "
                             "to be the same size (got %r)" % (
                hidden_sizes,
            ))
        num_layers = len(hidden_sizes)
        cell_input_size = inputs.get_shape()[-1].value

        est_size = estimate_cudnn_parameter_size(
            num_layers=num_layers,
            hidden_size=hidden_size,
            input_size=cell_input_size,
            input_mode="linear_input",
            direction="bidirectional"
        )
        # autoswitch to GPUs based on availability of alternatives:
        cudnn_params = tf.get_variable("RNNParams",
                                       shape=[est_size],
                                       dtype=tf.float32,
                                       initializer=tf.contrib.layers.variance_scaling_initializer())
        if weight_noise > 0:
            cudnn_params = add_weight_noise(cudnn_params,
                stddev=weight_noise, is_training=is_training)
        if faux_cudnn:
            cudnn_cell = CpuCudnnLSTM(num_layers,
                                      hidden_size,
                                      cell_input_size,
                                      input_mode="linear_input",
                                      direction="bidirectional")
        else:
            cpu_cudnn_params(cudnn_params,
                num_layers=num_layers,
                hidden_size=hidden_size,
                cell_input_size=cell_input_size,
                input_mode="linear_input",
                direction="bidirectional")
            cudnn_cell = CudnnLSTM(num_layers,
                                   hidden_size,
                                   cell_input_size,
                                   input_mode="linear_input",
                                   direction="bidirectional")
        init_state = tf.fill(
            (2 * num_layers, tf.shape(inputs)[1], hidden_size),
            tf.constant(np.float32(0.0)))
        hiddens, output_h, output_c = cudnn_cell(
            inputs,
            input_h=init_state,
            input_c=init_state,
            params=cudnn_params,
            is_training=True)
        hiddens = maybe_dropout(
            hiddens,
            keep_prob,
            is_training)
    else:
        cell = MultiRNNCell(
            [LSTMCell(hsize, is_training=is_training, keep_prob=keep_prob)
             for hsize in hidden_sizes]
        )
        hiddens, _ = bidirectional_dynamic_rnn(
            cell,
            inputs,
            time_major=True,
            dtype=dtype,
            swap_memory=True
        )
    return hiddens


def build_embed(inputs, features, index2words, keep_prob, is_training):
    embeddings = []
    for idx, (values, feature, index2word) in enumerate(zip(inputs, features, index2words)):
        if requires_vocab(feature):
            with tf.variable_scope("embedding_%d" % (idx,)):
                embedding = embedding_lookup(
                    values,
                    dim=feature["dimension"],
                    size=len(index2word),
                    dtype=tf.float32,
                    mask_negative=True
                )
                embeddings.append(embedding)
        elif requires_character_convolution(feature):
            embeddings.append(
                character_convolution(values, feature)
            )
        else:
            embeddings.append(tf.expand_dims(values, 2))
    return maybe_dropout(concat(embeddings, axis=2), keep_prob, is_training)


def crf_metrics(unary_scores, labels, transition_params, sequence_lengths,
                mask):
    """
    Computes CRF output metrics.
    Receives:
        unary_scores : batch-major order
        labels : batch-major order
        transition_params : nclasses x nclasses matrix.
        sequence_lengths : length of each time-sequence
        mask : batch-major example mask

    Returns:
        token_correct,
        token_correct_total,
        sentence_correct,
        sentence_correct_total
    """
    classes = unary_scores.get_shape()[-1].value
    decoded, scores = viterbi_decode(unary_scores,
                                     transition_params,
                                     sequence_lengths)

    tf.add_to_collection(UNARY_SCORES, unary_scores)
    tf.add_to_collection(DECODED, decoded)
    tf.add_to_collection(DECODED_SCORES, scores)

    equals_label = tf.equal(labels, decoded)
    token_correct = tf.reduce_sum(
        tf.cast(
            tf.logical_and(equals_label, mask),
            tf.int32
        )
    )
    token_correct_total = tf.reduce_sum(tf.cast(mask, tf.int32))
    tf.add_to_collection(TOKEN_CORRECT, token_correct)
    tf.add_to_collection(TOKEN_CORRECT_TOTAL, token_correct_total)
    sentence_correct, _ = compute_sentence_correct(equals_label, mask)
    sentence_correct_total = tf.reduce_sum(tf.cast(mask[:, 0], tf.int32))

    tf.add_to_collection(SENTENCE_CORRECT, sentence_correct)
    tf.add_to_collection(SENTENCE_CORRECT_TOTAL, sentence_correct_total)

    build_true_false_positives(decoded, mask, labels,
        classes, equals_label)

    return (token_correct, token_correct_total,
            sentence_correct, sentence_correct_total)


def build_true_false_positives(decoded, mask_batch_major, labels_batch_major,
                               classes, equals_label):
    masked_equals_label = tf.logical_and(equals_label, mask_batch_major)

    # now for each class compute tp, fp, fn
    # [nclasses x batch x time]
    masked_per_class = tf.logical_and(
        tf.equal(labels_batch_major[None, :, :], tf.range(classes)[:, None, None]),
        mask_batch_major)

    # correct, and on label
    correct = tf.reduce_sum(tf.cast(tf.logical_and(masked_per_class, equals_label[None, :, :]), tf.int32),
        axis=[1, 2])
    # predicted a particular class
    guessed = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(decoded[None, :, :], tf.range(classes)[:, None, None]), mask_batch_major), tf.int32),
        axis=[1, 2])
    total = tf.reduce_sum(tf.cast(masked_per_class, tf.int32), axis=[1, 2])
    tp, fp, fn = correct, guessed - correct, total - correct

    tf.add_to_collection(TRUE_POSITIVES, tp)
    tf.add_to_collection(FALSE_POSITIVES, fp)
    tf.add_to_collection(FALSE_NEGATIVES, fn)


def softmax_metrics(unary_scores, labels, mask):
    """
    Compute softmax output stats for correct/accuracy per-token/per-sentence.
    Receive
        unary_scores : time-major
        labels : time-major
        mask : time-major
    Returns:
        token_correct,
        token_correct_total,
        sentence_correct,
        sentence_correct_total
    """
    classes = unary_scores.get_shape()[-1].value
    unary_scores_batch_major = tf.transpose(unary_scores, [1, 0, 2])
    labels_batch_major = tf.transpose(labels, [1, 0])
    mask_batch_major = tf.transpose(mask, [1, 0])
    decoded = tf.cast(tf.argmax(unary_scores_batch_major, 2), labels.dtype)
    unary_probs_batch_major = tf.nn.softmax(unary_scores_batch_major)
    scores = tf.reduce_max(unary_probs_batch_major, 2)

    tf.add_to_collection(UNARY_SCORES, unary_probs_batch_major)
    tf.add_to_collection(DECODED, decoded)
    tf.add_to_collection(DECODED_SCORES, scores)

    equals_label = tf.equal(decoded, labels_batch_major)

    token_correct = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                equals_label,
                mask_batch_major
            ),
            tf.int32
        )
    )
    token_correct_total = tf.reduce_sum(tf.cast(mask, tf.int32))
    tf.add_to_collection(TOKEN_CORRECT, token_correct)
    tf.add_to_collection(TOKEN_CORRECT_TOTAL, token_correct_total)

    sentence_correct, sentence_correct_total = compute_sentence_correct(
        equals_label, mask_batch_major
    )
    tf.add_to_collection(SENTENCE_CORRECT, sentence_correct)
    tf.add_to_collection(SENTENCE_CORRECT_TOTAL, sentence_correct_total)

    build_true_false_positives(decoded, mask_batch_major, labels_batch_major,
        classes, equals_label)
    return (token_correct, token_correct_total,
            sentence_correct, sentence_correct_total)


def add_objective_names_types(objectives):
    for objective in objectives:
        with tf.variable_scope(objective["name"]):
            # store objective names in graph:
            tf.add_to_collection(OBJECTIVE_NAMES,
                tf.constant(objective["name"], name="objective_name")
            )
            tf.add_to_collection(OBJECTIVE_TYPES,
                tf.constant(objective["type"], name="objective_type")
            )


def build_loss(inputs, objectives, labels, labels_mask,
               labels_class_weights, fused, sequence_lengths,
               class_weights_normalize):
    """
    Compute loss function given the objectives.
    Assumes inputs are of the form [time, batch, features].

    Arguments:
    ----------
        inputs : tf.Tensor
        objectives : list<dict>, objective specs
        labels : list<tf.Tensor>
        labels_mask : list<tf.Tensor>
        labels_class_weights : list<tf.Tensor>
        sequence_lengths : tf.Tensor

    Returns:
        loss : tf.Tensor (scalar)
    """
    losses = []
    negative_log_likelihoods  = []
    sentence_corrects = []
    sentence_corrects_total = []
    token_corrects = []
    token_corrects_total = []
    max_output_vocab = max(len(obj["vocab"]) for obj in objectives)
    total_output_size = len(objectives) * max_output_vocab

    add_objective_names_types(objectives)

    if fused:
        with tf.variable_scope("FusedOutputs"):
            objective_labels = labels[0]
            mask = labels_mask[0]
            objective_class_weights = labels_class_weights[0]
            # perform all classifications at once:
            unary_scores = tf.contrib.layers.fully_connected(
                inputs, total_output_size,
                activation_fn=None
            )

            unary_scores = tf.reshape(unary_scores,
                                      [tf.shape(unary_scores)[0],
                                       tf.shape(unary_scores)[1],
                                       len(objectives),
                                       max_output_vocab])
            negative_log_likelihood = sparse_softmax_cross_entropy_with_logits(
                logits=unary_scores,
                labels=objective_labels
            )
            labels_mask_casted = tf.cast(mask, negative_log_likelihood.dtype)
            masked_negative_log_likelihood = negative_log_likelihood * labels_mask_casted
            if objective_class_weights is not None:
                class_weights_mask = tf.gather(
                        tf.reshape(objective_class_weights, [-1]),
                        objective_labels +
                        tf.reshape(tf.range(len(objectives)) * max_output_vocab, [1, 1, len(objectives)]))
                if class_weights_normalize:
                    masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood * class_weights_mask
                    num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted * class_weights_mask), 1e-6)
                    normed_loss = masked_weighed_negative_log_likelihood_sum / (num_predictions / len(objectives))
                else:
                    masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood * class_weights_mask
                    num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                    normed_loss = masked_weighed_negative_log_likelihood_sum / (num_predictions / len(objectives))
            else:
                masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood
                num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                normed_loss = masked_weighed_negative_log_likelihood_sum / (num_predictions / len(objectives))

            masked_negative_log_likelihood_sum = tf.reduce_sum(masked_negative_log_likelihood)
            losses.append(normed_loss)
            negative_log_likelihoods.append(masked_negative_log_likelihood_sum)

            for idx, objective in enumerate(objectives):
                with tf.variable_scope(objective["name"]):
                    (token_correct,
                     token_correct_total,
                     sentence_correct,
                     sentence_correct_total) = softmax_metrics(unary_scores[:, :, idx, :len(objective["vocab"])],
                                                               labels=objective_labels[:, :, idx],
                                                               mask=mask[:, :, idx])
                    token_corrects.append(token_correct)
                    token_corrects_total.append(token_correct_total)
                    sentence_corrects.append(sentence_correct)
                    sentence_corrects_total.append(sentence_correct_total)

    else:
        for objective, objective_labels, mask, objective_class_weights in zip(objectives, labels, labels_mask, labels_class_weights):
            with tf.variable_scope(objective["name"]):
                if objective["type"] == "crf":
                    unary_scores = tf.contrib.layers.fully_connected(
                        inputs,
                        len(objective["vocab"]),
                        activation_fn=None
                    )
                    unary_scores_batch_major = tf.transpose(unary_scores, [1, 0, 2])
                    labels_batch_major = tf.transpose(objective_labels, [1, 0])


                    padded_unary_scores_batch_major = tf.cond(tf.greater(tf.shape(unary_scores_batch_major)[1], 1),
                        lambda: unary_scores_batch_major,
                        lambda: tf.pad(unary_scores_batch_major, [[0, 0], [0, 1], [0, 0]]))
                    padded_labels_batch_major = tf.cond(tf.greater(tf.shape(labels_batch_major)[1], 1),
                        lambda: labels_batch_major,
                        lambda: tf.pad(labels_batch_major, [[0, 0], [0, 1]]))

                    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                        padded_unary_scores_batch_major, padded_labels_batch_major, sequence_lengths
                    )
                    labels_mask_casted = tf.cast(mask, log_likelihood.dtype)
                    masked_log_likelihood = (
                        log_likelihood * labels_mask_casted
                    )
                    masked_negative_log_likelihood_sum = -tf.reduce_sum(masked_log_likelihood)
                    num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                    losses.append(masked_negative_log_likelihood_sum / num_predictions)
                    negative_log_likelihoods.append(masked_negative_log_likelihood_sum)
                    sequence_mask = tf.logical_and(
                        tf.sequence_mask(sequence_lengths),
                        # pad the time dimension:
                        tf.expand_dims(mask, 1)
                    )

                    (token_correct,
                     token_correct_total,
                     sentence_correct,
                     sentence_correct_total) = crf_metrics(unary_scores_batch_major,
                                                           labels=labels_batch_major,
                                                           mask=sequence_mask,
                                                           transition_params=transition_params,
                                                           sequence_lengths=sequence_lengths)
                elif objective["type"] == 'softmax':
                    unary_scores = tf.contrib.layers.fully_connected(
                        inputs,
                        len(objective["vocab"]),
                        activation_fn=None
                    )
                    negative_log_likelihood = sparse_softmax_cross_entropy_with_logits(
                        logits=unary_scores,
                        labels=objective_labels
                    )
                    labels_mask_casted = tf.cast(mask, negative_log_likelihood.dtype)
                    masked_negative_log_likelihood = (
                        negative_log_likelihood * labels_mask_casted
                    )
                    if objective_class_weights is not None:
                        class_weights_mask = tf.gather(objective_class_weights, objective_labels)
                        masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood * class_weights_mask
                        masked_negative_log_likelihood_sum = tf.reduce_sum(masked_negative_log_likelihood)

                        if class_weights_normalize:
                            num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted * class_weights_mask), 1e-6)
                            normed_loss = masked_weighed_negative_log_likelihood_sum / num_predictions
                        else:
                            num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                            normed_loss = masked_weighed_negative_log_likelihood_sum / num_predictions
                    else:
                        masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood
                        masked_negative_log_likelihood_sum = tf.reduce_sum(masked_negative_log_likelihood)
                        num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                        normed_loss = masked_weighed_negative_log_likelihood_sum / num_predictions

                    losses.append(normed_loss)
                    negative_log_likelihoods.append(masked_negative_log_likelihood_sum)

                    (token_correct,
                     token_correct_total,
                     sentence_correct,
                     sentence_correct_total) = softmax_metrics(unary_scores,
                                                               labels=objective_labels,
                                                               mask=mask)
                else:
                    raise ValueError(
                        "unknown objective type %r" % (objective["type"],)
                    )
                token_corrects.append(token_correct)
                token_corrects_total.append(token_correct_total)
                sentence_corrects.append(sentence_correct)
                sentence_corrects_total.append(sentence_correct_total)
    # aggregate metrics for all objectives:
    total_loss = tf.reduce_sum(sum_list(losses))
    tf.summary.scalar("BatchLoss", total_loss)
    neg_log_likelihood_total = sum_list(negative_log_likelihoods)
    tf.summary.scalar("BatchNLL", neg_log_likelihood_total)
    tf.add_to_collection(NLL, neg_log_likelihood_total)
    tf.add_to_collection(NLL_TOTAL, tf.shape(inputs)[1])

    sentence_corrects_total = sum_list(sentence_corrects_total)
    sentence_corrects = sum_list(sentence_corrects)
    tf.add_to_collection(SENTENCE_CORRECT_ALL, sentence_corrects)
    tf.add_to_collection(SENTENCE_CORRECT_ALL_TOTAL, sentence_corrects_total)

    token_corrects_total = sum_list(token_corrects_total)
    token_corrects = sum_list(token_corrects)
    tf.add_to_collection(TOKEN_CORRECT_ALL, token_corrects)
    tf.add_to_collection(TOKEN_CORRECT_ALL_TOTAL, token_corrects_total)
    return total_loss


def build_model(name,
                trainable,
                features,
                feature_index2words,
                objectives,
                keep_prob,
                input_keep_prob,
                hidden_sizes,
                freeze_rate,
                freeze_rate_anneal,
                solver,
                cudnn,
                fused,
                faux_cudnn,
                class_weights,
                class_weights_normalize,
                class_weights_clipval,
                lr,
                weight_noise,
                anneal_rate,
                clip_norm):
    # mixed output fusing is currently unsupported
    if fused and any(obj["type"] != "softmax" for obj in objectives):
        raise ValueError("cannot fuse outputs and use non-softmax output.")
    # clear all existing collections to ensure every new collection is
    # is created fresh
    graph = tf.get_default_graph()
    for collection_name in graph.get_all_collection_keys():
        graph.clear_collection(collection_name)

    # build a model under the model's name to prevent collisions
    # when multiple models are restored simultaneously
    with tf.variable_scope(name):
        global_step = tf.Variable(0, trainable=False, name="global_step")
        tf.add_to_collection(GLOBAL_STEP, global_step)
        # model placeholders:
        (input_placeholders,
         labels,
         labels_mask,
         labels_class_weights,
         sequence_lengths,
         is_training) = build_inputs(features,
                                     objectives=objectives,
                                     fused=fused,
                                     class_weights=class_weights,
                                     class_weights_clipval=class_weights_clipval)
        embed = build_embed(input_placeholders,
                            features=features,
                            index2words=feature_index2words,
                            is_training=is_training,
                            keep_prob=input_keep_prob)
        hiddens = embed
        if len(hidden_sizes) > 0:
            hiddens = build_recurrent(hiddens,
                                      cudnn=cudnn,
                                      faux_cudnn=faux_cudnn,
                                      hidden_sizes=hidden_sizes,
                                      keep_prob=keep_prob,
                                      weight_noise=weight_noise,
                                      is_training=is_training)

        loss = build_loss(hiddens,
                          objectives=objectives,
                          fused=fused,
                          labels=labels,
                          labels_mask=labels_mask,
                          labels_class_weights=labels_class_weights,
                          class_weights_normalize=class_weights_normalize,
                          sequence_lengths=sequence_lengths)
        if trainable:
            learning_rate = tf.train.exponential_decay(lr, global_step,
                                                       33000, anneal_rate, staircase=True)

            if solver == "adam":
                optimizer = LazyAdamOptimizer(learning_rate)
            elif solver == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                raise ValueError("Unknown solver %r." % (solver))

            grad_vars = optimizer.compute_gradients(loss)
            if clip_norm > 0:
                grad_vars = [(grad if isinstance(grad, tf.IndexedSlices) else tf.clip_by_norm(grad, clip_norm), var) for grad, var in grad_vars]
            train_op = optimizer.apply_gradients(grad_vars, global_step=global_step)
        else:
            train_op = tf.no_op()
        tf.add_to_collection(TRAIN_OP, train_op)
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.merge_all())


def restore_session(session,
                    path,
                    replace_to=None,
                    replace_from=None,
                    verbose=False,
                    use_metagraph=True,
                    only_features=False):
    """
    Call restore on tf.train.Saver on a specific path to store all the
    variables of the current tensorflow session to a file for later restoring.

    Arguments:
        session : tf.Session
        path : str, place containing the session data to restore
        verbose : bool, print status messages.
        use_metagraph : bool, restore by re-creating saved metagraph.

    Returns:
        bool : success or failure of the restoration
    """
    makedirs(path, exist_ok=True)
    if not path.endswith("/"):
        path = path + "/"
    checkpoint = tf.train.get_checkpoint_state(path)
    if verbose:
        print("Looking for saved session under %r" % (path,), flush=True)
    if checkpoint is None or checkpoint.model_checkpoint_path is None:
        if verbose:
            print("No saved session found", flush=True)
        return False
    fname = basename(checkpoint.model_checkpoint_path)
    if verbose:
        print("Restoring saved session from %r" % (join(path, fname),), flush=True)

    if use_metagraph:
        param_saver = tf.train.import_meta_graph(join(path, fname + ".meta"),
            clear_devices=True)
        missing_vars = []
    else:
        if only_features:
            to_restore = {}
            whitelist = ["embedding", "/RNN/", "/RNNParams", "CharacterConvolution", "HighwayLayer"]
            for var in tf.global_variables():
                if any(keyword in var.name for keyword in whitelist):
                    to_restore[var.name[:-2]] = var
            param_saver = tf.train.Saver(to_restore)
        else:
            if replace_to is not None and replace_from is not None:
                to_restore = {}
                for var in tf.global_variables():
                    var_name = var.name[:var.name.rfind(":")]
                    old_name = var_name.replace(replace_to, replace_from)
                    to_restore[old_name] = var
                param_saver = tf.train.Saver(to_restore)
                missing_vars = []
            else:
                reader = tf.train.NewCheckpointReader(join(path, fname))
                saved_shapes = reader.get_variable_to_shape_map()
                found_vars = [var for var in tf.global_variables()
                              if var.name.split(':')[0] in saved_shapes]
                missing_vars = [var for var in tf.global_variables()
                                if var.name.split(':')[0] not in saved_shapes]
                param_saver = tf.train.Saver(found_vars)
    param_saver.restore(session, join(path, fname))
    session.run([var.initializer for var in missing_vars])
    return True


def bidirectional_dynamic_rnn(cell, inputs, dtype, time_major=True, swap_memory=False):
    with tf.variable_scope("forward"):
        out_fwd, final_fwd = tf.nn.dynamic_rnn(
            cell,
            inputs,
            time_major=time_major,
            dtype=dtype,
            swap_memory=swap_memory
        )

    if time_major:
        reverse_axis = 0
    else:
        reverse_axis = 1

    with tf.variable_scope("backward"):
        out_bwd, final_bwd = tf.nn.dynamic_rnn(
            cell,
            reverse(inputs, axis=reverse_axis),
            time_major=time_major,
            dtype=dtype,
            swap_memory=swap_memory
        )

    out_bwd = reverse(out_bwd, axis=reverse_axis)
    return concat([out_fwd, out_bwd], axis=2), (final_fwd, final_bwd)


def get_embedding_lookup(size, dim, dtype, reuse=None, trainable=True):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        W = tf.get_variable(
            name="embedding",
            shape=[size, dim],
            dtype=dtype,
            initializer=tf.random_uniform_initializer(
                -1.0 / math.sqrt(dim),
                1.0 / math.sqrt(dim)
            ),
            trainable=trainable
        )
        return W


def embedding_lookup(inputs,
                     size,
                     dim,
                     dtype,
                     reuse=None,
                     mask_negative=False,
                     trainable=True,
                     place_on_cpu_if_big=True):
    """
    Construct an Embedding layer that gathers
    elements from a matrix with `size` rows,
    and `dim` features using the indices stored in `x`.

    Arguments:
    ----------
        inputs : tf.Tensor, of integer type
        size : int, how many symbols in the lookup table
        dim : int, how many columns per symbol.
        dtype : data type for the lookup table (e.g. tf.float32)
        reuse : bool, (default None) whether the lookup table
            was already used before (thus this is weight sharing).
        mask_negative : bool, (default False) should -1s in the
            lookup input indicate padding (e.g. no lookup),
            and thus should those values be masked out post-lookup.
        trainable : bool (default True), whether the parameters of
            this lookup table can be backpropagated into (e.g.
            for Glove word vectors that are fixed pre-trained, this
            can be set to False).
        place_on_cpu_if_big : bool, if matrix is big, store it on cpu.
    Returns:
    --------
        tf.Tensor, result of tf.nn.embedding_lookup(LookupTable, inputs)
    """
    W = get_embedding_lookup(size, dim, dtype, reuse, trainable=trainable)
    if mask_negative:
        embedded = tf.nn.embedding_lookup(W, tf.maximum(inputs, 0))
        null_mask = tf.expand_dims(
            tf.cast(
                tf.not_equal(inputs, -1),
                dtype
            ),
            -1
        )
        return embedded * null_mask
    else:
        return tf.nn.embedding_lookup(W, inputs)


def _get_sharded_variable(name, shape, dtype, num_shards):
    """Get a list of sharded variables with the given dtype."""
    if num_shards > shape[0]:
        raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                         (shape, num_shards))
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards

    shards = []
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1
        shards.append(
            tf.get_variable(
                name + "_%d" % i,
                [current_size] + shape[1:],
                dtype=dtype
            )
        )
    return shards


def _get_concat_variable(name, shape, dtype, num_shards):
    """Get a sharded variable concatenated into one tensor."""
    sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
    if len(sharded_variable) == 1:
        return sharded_variable[0]

    concat_name = name + "/concat"
    concat_full_name = tf.get_variable_scope().name + "/" + concat_name + ":0"
    for value in tf.get_collection(tf.GraphKeys.CONCATENATED_VARIABLES):
        if value.name == concat_full_name:
            return value

    concat_variable = tf.concat_v2(sharded_variable, 0, name=concat_name)
    tf.add_to_collection(tf.GraphKeys.CONCATENATED_VARIABLES, concat_variable)
    return concat_variable


class SequenceModel(object):
    def __init__(self,
                 objectives,
                 features,
                 feature_index2words,
                 hidden_sizes,
                 keep_prob,
                 lr,
                 solver,
                 seed=1234,
                 input_keep_prob=0.7,
                 clip_norm=-1,
                 name="SequenceTagger",
                 cudnn=False,
                 anneal_rate=0.99,
                 trainable=True,
                 weight_noise=0.0,
                 class_weights_normalize=False,
                 faux_cudnn=False,
                 class_weights=False,
                 class_weights_clipval=1000.0,
                 freeze_rate=1.0,
                 fused=False,
                 freeze_rate_anneal=0.8,
                 create_variables=True):
        if fused and objectives[0]["type"] == "crf":
            fused = False

        self.keep_prob = keep_prob
        self.input_keep_prob = input_keep_prob
        self.hidden_sizes = hidden_sizes
        self.name = name
        self.objectives = objectives
        self.features = features
        self.feature_index2words = feature_index2words
        self.seed = seed
        self.lr = lr
        self.fused = fused
        self.weight_noise = weight_noise
        self.anneal_rate = anneal_rate
        self.clip_norm = clip_norm
        self.solver = solver
        self.class_weights_normalize = class_weights_normalize
        self.class_weights = class_weights
        self.class_weights_clipval = class_weights_clipval
        self.rng = np.random.RandomState(seed)
        self.cudnn = cudnn
        self.feature_word2index = [
            {w: k for k, w in enumerate(index2word)} if index2word is not None else None
            for index2word in self.feature_index2words
        ]
        self.label2index = [
            {w: k for k, w in enumerate(objective["vocab"])}
            for objective in self.objectives
        ]

        if create_variables:
            # 1) build graph here (TF functional code pattern)
            build_model(name=self.name,
                        trainable=trainable,
                        objectives=self.objectives,
                        features=self.features,
                        feature_index2words=self.feature_index2words,
                        hidden_sizes=self.hidden_sizes,
                        keep_prob=self.keep_prob,
                        solver=self.solver,
                        freeze_rate=freeze_rate,
                        class_weights_normalize=self.class_weights_normalize,
                        class_weights=self.class_weights,
                        class_weights_clipval=self.class_weights_clipval,
                        freeze_rate_anneal=freeze_rate_anneal,
                        cudnn=self.cudnn,
                        lr=self.lr,
                        fused=self.fused,
                        weight_noise=self.weight_noise,
                        anneal_rate=self.anneal_rate,
                        input_keep_prob=self.input_keep_prob,
                        faux_cudnn=faux_cudnn,
                        clip_norm=self.clip_norm)

        # 2) and use meta graph to recover these fields:
        self.recover_graph_variables()


    def recover_graph_variables(self):
        """Use TF meta graph to obtain key metrics
        and outputs from model."""
        self.labels = tf.get_collection(LABEL_PLACEHOLDERS)
        self.labels_mask = tf.get_collection(LABEL_MASK_PLACEHOLDERS)
        self.input_placeholders = tf.get_collection(INPUT_PLACEHOLDERS)
        self.sequence_lengths = tf.get_collection(SEQUENCE_LENGTHS)[0]
        self.decoded = tf.get_collection(DECODED)
        self.decoded_scores = tf.get_collection(DECODED_SCORES)
        self.unary_scores = tf.get_collection(UNARY_SCORES)

        self.token_correct = tf.get_collection(TOKEN_CORRECT)
        self.token_correct_total = tf.get_collection(TOKEN_CORRECT_TOTAL)

        self.sentence_correct = tf.get_collection(SENTENCE_CORRECT)
        self.sentence_correct_total = tf.get_collection(SENTENCE_CORRECT_TOTAL)

        self.token_correct_all = tf.get_collection(TOKEN_CORRECT_ALL)[0]
        self.token_correct_all_total = tf.get_collection(TOKEN_CORRECT_ALL_TOTAL)[0]
        self.sentence_correct_all = tf.get_collection(SENTENCE_CORRECT_ALL)[0]
        self.sentence_correct_all_total = tf.get_collection(SENTENCE_CORRECT_ALL_TOTAL)[0]

        self.true_positives = tf.get_collection(TRUE_POSITIVES)
        self.false_positives = tf.get_collection(FALSE_POSITIVES)
        self.false_negatives = tf.get_collection(FALSE_NEGATIVES)

        if len(self.true_positives) == 0 and len(self.token_correct) != 0:
            self.true_positives = [None for _ in self.token_correct]
            self.false_positives = [None for _ in self.token_correct]
            self.false_negatives = [None for _ in self.token_correct]

        if len(tf.get_collection(GLOBAL_STEP)) > 0:
            self.global_step = tf.get_collection(GLOBAL_STEP)[0]
        else:
            try:
                self.global_step = tf.get_default_graph().get_tensor_by_name(
                    self.name + "/" + "global_step:0")
            except KeyError:
                self.global_step = tf.Variable(0, trainable=False, name="global_step")
            tf.add_to_collection(GLOBAL_STEP, self.global_step)

        self.is_training = tf.get_collection(IS_TRAINING)[0]
        self.noop = tf.no_op()
        self.train_op = tf.get_collection(TRAIN_OP)[0]
        train_summaries = tf.get_collection(TRAIN_SUMMARIES)
        self.train_summaries = train_summaries[0] if len(train_summaries) > 0 else None

        self.nll = tf.get_collection(NLL)[0]
        self.nll_total = tf.get_collection(NLL_TOTAL)[0]
        self.saver = tf.train.Saver()



    @classmethod
    def overrideable_fields(cls):
        return [
            "keep_prob",
            "name",
            "lr",
            "clip_norm",
            "class_weights_normalize",
            "class_weights_clipval",
            "cudnn",
            "anneal_rate",
            "weight_noise",
            "input_keep_prob"
        ]

    @classmethod
    def fields_to_save(cls):
        return [
            "hidden_sizes",
            "objectives",
            "name",
            "cudnn",
            "class_weights",
            "features",
            "fused",
            "class_weights_normalize",
            "weight_noise",
            "anneal_rate",
            "feature_index2words",
            "solver",
            "lr",
            "clip_norm",
            "keep_prob",
            "input_keep_prob",
            "class_weights_clipval"
        ]

    def predict(self, session, feed_dict):
        feed_dict[self.is_training] = False
        outputs, outputs_probs = session.run(
            (self.decoded, self.decoded_scores), feed_dict
        )
        predictions_out = {}
        for value, val_prob, objective in zip(outputs, outputs_probs, self.objectives):
            predictions_out[objective["name"]] = (value, val_prob)
        return predictions_out

    def predict_proba(self, session, feed_dict):
        feed_dict[self.is_training] = False
        outputs = session.run(
            self.unary_scores, feed_dict
        )
        predictions_out = {}
        for value, objective in zip(outputs, self.objectives):
            predictions_out[objective["name"]] = value
        return predictions_out

    def save(self, session, path):
        makedirs(path, exist_ok=True)
        with open(join(path, "model.json"), "wt") as fout:
            save_dict = {}
            for field in type(self).fields_to_save():
                save_dict[field] = getattr(self, field)
            json.dump(save_dict, fout)

        with open(join(path, "rng.pkl"), "wb") as fout:
            pickle.dump(self.rng, fout)

        save_session(session, self.saver, path, verbose=True)

    @classmethod
    def load(cls, session, path, args=None, verbose=True, trainable=True,
             rebuild_graph=False, faux_cudnn=False, replace_to=None, replace_from=None):
        """Convenience method for using a tensorflow session to reload
        a previously saved + serialized model from disk."""
        with open(join(path, "model.json"), "rt") as fin:
            model_props = json.load(fin)

        # update fields based on CLI:
        if args is not None:
            ex_fields = explicitly_set_fields()
            for field in cls.overrideable_fields():
                if field in ex_fields:
                    model_props[field] = getattr(args, field)

        # prune old fields based on changes to saveable fields:
        relevant_props = {}
        for field in cls.fields_to_save():
            if field in model_props:
                relevant_props[field] = model_props[field]

        relevant_props["trainable"] = trainable
        relevant_props["faux_cudnn"] = faux_cudnn

        if rebuild_graph:
            print("Using rebuild_graph mode: creating a new graph.", flush=True)
            relevant_props["create_variables"] = True
            model = cls(**relevant_props)
            restore_session(
                session, path,
                replace_to=replace_to,
                replace_from=replace_from,
                verbose=verbose,
                use_metagraph=False
            )
        else:
            if model_props.get("cudnn", False):
                import tensorflow.contrib.cudnn_rnn
            relevant_props["create_variables"] = False
            restore_session(
                session, path,
                verbose=verbose,
                use_metagraph=True
            )
            model = cls(**relevant_props)

        rng_path = join(path, "rng.pkl")
        if exists(rng_path):
            # apply the saved random number generator to this
            # model:
            with open(rng_path, "rb") as fin:
                model.rng = pickle.load(fin)
        return model


def make_path_absolute(obj, basepath):
    copied = obj.copy()
    for key in ["path", "vocab"]:
        if key in copied:
            copied[key] = join(basepath, copied[key])
    return copied


class Config(object):
    def __init__(self, datasets, features, objectives,
                 wikidata_path, classification_path):
        assert(len(features) > 0)
        self.datasets = datasets
        self.features = features
        self.objectives = objectives
        self.classifications = None
        self.wikidata_path = wikidata_path
        self.classification_path = classification_path

        # build the objective names:
        self._named_objectives = [obj["name"] for obj in self.objectives]

    @classmethod
    def load(cls, path):
        with open(path, "rt") as fin:
            config = json.load(fin)
        config_dirname = dirname(path)
        return cls(
            datasets=[make_path_absolute(dataset, config_dirname) for dataset in config['datasets']],
            features=[make_path_absolute(feat, config_dirname) for feat in config['features']],
            objectives=[make_path_absolute(objective, config_dirname) for objective in config['objectives']],
            wikidata_path=config.get("wikidata_path", None),
            classification_path=(
                join(config_dirname, config.get("classification_path", None))
                if "classification_path" in config else None)
        )

    def load_dataset_separate(self, dataset_type):
        paths = [dataset for dataset in self.datasets if dataset["type"] == dataset_type]
        all_examples = {}
        for dataset in paths:
            _, extension = splitext(dataset["path"])
            if extension == ".h5" or extension == ".hdf5":
                if self.classifications is None:
                    if self.wikidata_path is None or self.classification_path is None:
                        raise ValueError("missing wikidata_path and "
                                         "classification_path, cannot "
                                         "construct H5Dataset.")
                    self.classifications = ClassificationHandler(
                        self.wikidata_path,
                        self.classification_path
                    )
                examples = H5Dataset(
                    dataset["path"],
                    dataset["x"],
                    dataset["y"],
                    self._named_objectives,
                    ignore_value=dataset.get('ignore', None),
                    classifications=self.classifications)
            else:
                examples = TSVDataset(
                    dataset["path"],
                    dataset["x"],
                    dataset["y"],
                    self._named_objectives,
                    comment=dataset.get('comment', '#'),
                    ignore_value=dataset.get('ignore', None),
                    retokenize=dataset.get('retokenize', False))
            title = dataset["path"].split('/')[-1].split(".")[0]
            name = title
            iteration = 1
            while name in all_examples:
                name = title + "-%d" % (iteration,)
                iteration += 1
            all_examples[name] = examples
        return all_examples

    def load_dataset(self, dataset_type, merge=True):
        datasets = self.load_dataset_separate(dataset_type)
        if merge:
            return CombinedDataset(list(datasets.values()))
        return datasets


def boolean_argument(parser, name, default):
    parser.add_argument("--" + name, action="store_true", default=default)
    parser.add_argument("--no" + name, action="store_false", dest=name)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--anneal_rate', type=float, default=0.99)
    parser.add_argument('--clip_norm', type=float, default=-1)
    parser.add_argument('--weight_noise', type=float, default=0.0)
    parser.add_argument('--hidden_sizes', type=int, nargs="*", default=[200, 200])
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--restore_input_features', type=str, default=None)
    parser.add_argument('--improvement_key', type=str, default="token_correct")
    parser.add_argument('--freeze_rate', type=float, default=1.0)
    parser.add_argument('--freeze_rate_anneal', type=float, default=0.8)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--test_every', type=int, default=10000,
        help="Number of training iterations after which testing should occur.")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_patience', type=int, default=10)
    parser.add_argument('--class_weights_clipval', type=float, default=1000.0)
    parser.add_argument('--device', type=str, default="gpu:0")
    parser.add_argument('--keep_prob', type=float, default=0.5)
    parser.add_argument('--input_keep_prob', type=float, default=0.7)
    parser.add_argument('--solver', type=str, default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--name", type=str, default="SequenceTagger")
    parser.add_argument("--old_name", type=str, default=None)
    boolean_argument(parser, "cudnn", True)
    boolean_argument(parser, "faux_cudnn", False)
    boolean_argument(parser, "class_weights", False)
    boolean_argument(parser, "rebuild_graph", False)
    boolean_argument(parser, "class_weights_normalize", False)
    boolean_argument(parser, "fused", True)
    boolean_argument(parser, "report_metrics_per_axis", True)
    boolean_argument(parser, "report_class_f1", False)
    return parser.parse_args(args=args)


def get_vocab(dataset, max_vocab=-1, extra_words=None):
    index2word = []
    occurrence = {}
    for el in dataset:
        if el not in occurrence:
            index2word.append(el)
            occurrence[el] = 1
        else:
            occurrence[el] += 1
    index2word = sorted(index2word, key=lambda x: occurrence[x], reverse=True)
    if max_vocab > 0:
        index2word = index2word[:max_vocab]
    if extra_words is not None:
        index2word = extra_words + index2word
    return index2word


def get_objectives(objectives, dataset):
    out = []
    for obj_idx, objective in enumerate(objectives):
        if "vocab" in objective:
            with open(objective["vocab"], "rt") as fin:
                vocab = fin.read().splitlines()
        else:
            vocab = get_vocab((w[obj_idx] for _, y in dataset for w in y if w[obj_idx] is not None), -1)

        out.append(
            {
                "vocab": vocab,
                "type": objective["type"],
                "name": objective["name"]
            }
        )
    return out


def merge_all_metrics(metrics):
    out = {}
    for key, metric in metrics.items():
        for subkey, submetric in metric.items():
            if len(key) > 0:
                out[key + "_" + subkey] = submetric
                if subkey not in out:
                    out[subkey] = submetric
                else:
                    out[subkey] += submetric
            else:
                out[subkey] = submetric
    return out


def log_outcome(logger, outcome, step, name):
    for k, v in sorted(outcome.items()):
        if "total" in k:
            continue
        else:
            total = outcome[k + "_total"]
            if total == 0:
                continue
            logger.log(k, v / total, step=step)
    logger.writer.flush()


def compute_f1(metrics, objectives, report_class_f1):
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total = 0
    for objective in objectives:
        name = objective["name"]
        key = "%s_true_positives" % (name,)
        if key not in metrics:
            continue
        tp = metrics[key]
        fp = metrics["%s_false_positives" % (name,)]
        fn = metrics["%s_false_negatives" % (name,)]
        del metrics[key]
        del metrics["%s_false_positives" % (name,)]
        del metrics["%s_false_negatives" % (name,)]



        precision = 1.* tp / np.maximum((tp + fp), 1e-6)
        recall = 1. * tp / np.maximum((tp + fn), 1e-6)
        f1 = 2.0 * precision * recall / np.maximum((precision + recall), 1e-6)

        support = tp + fn

        full_f1 = np.average(f1, weights=support) * 100.0
        full_recall = np.average(recall, weights=support) * 100.0
        full_precision = np.average(precision, weights=support) * 100.0

        total_f1 += full_f1
        total_recall += full_recall
        total_precision += full_precision
        total += 1
        if report_class_f1:
            print("F1 %s: %r" % (name, full_f1))
            print("Name\tF1\tTP\tFP\tFN")
            rows = zip([label for label, has_support in zip(objective["vocab"],
                                                            support > 0)
                        if has_support],
                       f1, tp, fp, fn)
            for val, f1_val, val_tp, val_fp, val_fn in rows:
                print("%s\t%r\t%d\t%d\t%d" % (
                    val, f1_val, val_tp, val_fp, val_fn))
            print("")
    if total > 0:
        metrics["F1"] = total_f1
        metrics["recall"] = total_recall
        metrics["precision"] = total_precision
        metrics["F1_total"] = total
        metrics["recall_total"] = total
        metrics["precision_total"] = total


def accuracy(model, session, datasets, batch_size, train,
             report_metrics_per_axis, report_class_f1,
             callback=None,
             callback_period=None, writer=None):
    pbar = get_progress_bar("train" if train else "validation", item="batches")
    if not isinstance(datasets, dict):
        datasets = {'':datasets}
    all_metrics_agg = {}

    if callback is not None:
        if callback_period is None:
            raise ValueError("callback_period cannot be None if "
                             "callback is used.")
    else:
        callback_period = None

    if train:
        train_op = model.train_op
    else:
        train_op = model.noop
    is_training = model.is_training
    metrics = {"nll": model.nll, "nll_total": model.nll_total}
    summaries = []

    if not train:
        metric_iter = zip(
            model.objectives,
            model.token_correct,
            model.token_correct_total,
            model.sentence_correct,
            model.sentence_correct_total,
            model.true_positives,
            model.false_positives,
            model.false_negatives
        )
        for metric_vars in metric_iter:
            (
                objective,
                token_correct,
                token_correct_total,
                sentence_correct,
                sentence_correct_total,
                true_positives,
                false_positives,
                false_negatives
            ) = metric_vars
            name = objective["name"]
            if report_metrics_per_axis:
                metrics["%s_token_correct" % (name,)] = token_correct
                metrics["%s_token_correct_total" % (name,)] = token_correct_total
                metrics["%s_sentence_correct" % (name,)] = sentence_correct
                metrics["%s_sentence_correct_total" % (name,)] = sentence_correct_total
            if true_positives is not None:
                metrics["%s_true_positives" % (name,)] = true_positives
                metrics["%s_false_positives" % (name,)] = false_positives
                metrics["%s_false_negatives" % (name,)] = false_negatives
        metrics["token_correct"] = model.token_correct_all
        metrics["token_correct_total"] = model.token_correct_all_total
        metrics["sentence_correct"] = model.sentence_correct_all
        metrics["sentence_correct_total"] = model.sentence_correct_all_total
        summaries = []
    else:
        if writer is not None and model.train_summaries is not None:
            summaries = model.train_summaries

    metrics_values = [v for _, v in sorted(metrics.items())]
    metrics_names = [name for name, _ in sorted(metrics.items())]
    outputs_val = [train_op, model.global_step, summaries, metrics_values]
    for title, dataset in datasets.items():
        batches = iter_batches_single_threaded(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            train=train,
            pbar=pbar
        )
        metrics_agg = {}
        iteration = 0
        for feed_dict in batches:
            feed_dict[is_training] = train
            _, step, summary_out, outputs = session.run(outputs_val, feed_dict)
            if writer is not None:
                writer.add_summary(summary_out, step)
            for key, value in zip(metrics_names, outputs[:len(metrics_names)]):
                if key not in metrics_agg:
                    metrics_agg[key] = value
                else:
                    metrics_agg[key] += value
            iteration += 1
            if callback_period is not None and iteration % callback_period == 0:
                callback(iteration)

            if np.isnan(metrics_agg['nll']):
                print("loss is NaN.", flush=True, file=sys.stderr)
                sys.exit(1)

        compute_f1(metrics_agg, model.objectives, report_class_f1)
        all_metrics_agg[title] = metrics_agg
        del batches
    return merge_all_metrics(all_metrics_agg)


def present_outcome(outcome, epoch, name):
    string_rows = []
    for k, v in sorted(outcome.items()):
        if "total" in k:
            continue
        else:
            total = outcome[k + "_total"]
            if total == 0:
                continue
            if "correct" in k:
                string_rows.append(
                    [
                        k,
                        "%.2f%%" % (100.0 * v / total),
                        "(%d correct / %d)" % (v, total)
                    ]
                )
            else:
                string_rows.append(
                    [
                        k,
                        "%.3f" % (v / total),
                        ""
                    ]
                )
    max_len_cols = [
        max(len(row[colidx]) for row in string_rows)
        for colidx in range(len(string_rows[0]))
    ] if len(string_rows) > 0 else []
    rows = []
    for row in string_rows:
        rows.append(
            " ".join(
                [col + " " * (max_len_cols[colidx] - len(col))
                 for colidx, col in enumerate(row)]
            )
        )
    return "\n".join(["Epoch {epoch}: {name}".format(epoch=epoch, name=name)] + rows)


def print_outcome(outcome, objectives, epoch, step, name, logger=None):
    outcome_report = present_outcome(outcome, epoch, name)
    if logger is not None:
        log_outcome(logger, outcome, step, name)
    print(outcome_report)



class SequenceTagger(object):
    def __init__(self, path, device="gpu", faux_cudnn=False, rebuild_graph=False):
        tf.reset_default_graph()
        session_conf = tf.ConfigProto(
            allow_soft_placement=True
        )
        self.session = tf.InteractiveSession(config=session_conf)
        with tf.device(device):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._model = SequenceModel.load(
                    self.session,
                    path,
                    args=None,
                    verbose=False,
                    trainable=False,
                    rebuild_graph=rebuild_graph,
                    faux_cudnn=faux_cudnn
                )

    @property
    def objectives(self):
        return self._model.objectives

    def predict_proba(self, tokens):
        blank_labels = tuple(None for _ in self._model.objectives)
        batches = list(iter_batches_single_threaded(
            model=self._model,
            dataset=[
                (tokens, [blank_labels for t in tokens])
            ],
            batch_size=1,
            train=False,
            autoresize=False
        ))
        outputs = []
        batches[0][self._model.is_training] = False
        probs_out = self._model.predict_proba(
            self.session, batches[0]
        )
        return probs_out


    def predict_proba_sentences(self, sentences):
        blank_labels = tuple(None for _ in self._model.objectives)
        batches = iter_batches_single_threaded(
            model=self._model,
            dataset=[
                (sentence, [blank_labels for t in sentence])
                for sentence in sentences
            ],
            batch_size=min(256, len(sentences)),
            train=False,
            autoresize=False
        )
        for batch in batches:
            batch[self._model.is_training] = False
            yield self._model.predict_proba(
                self.session, batch
            )

    def predict_topk_sentences(self, sentences, k=5):
        blank_labels = tuple(None for _ in self._model.objectives)
        batches = iter_batches_single_threaded(
            model=self._model,
            dataset=[
                (sentence, [blank_labels for t in sentence])
                for sentence in sentences
            ],
            batch_size=min(256, len(sentences)),
            train=False,
            autoresize=False
        )
        for batch in batches:
            outputs = self._model.predict_proba(
                self.session, batch
            )
            named_outputs = {}
            for objective in self._model.objectives:
                obj_name = objective["name"]
                tags, scores = outputs[obj_name]
                if objective["type"] == "crf":
                    named_outputs[obj_name] = [
                        [(token, [objective["vocab"][tag]], [score]) for token, tag in zip(tokens, tags)]
                        for tokens, tags, score in zip(sentences, tags, scores)
                    ]
                elif objective["type"] == 'softmax':
                    all_sent_scores = []

                    for tokens, scores in zip(sentences, scores):
                        sent_scores = []
                        for token, token_scores in zip(tokens, scores):
                            topk = np.argsort(token_scores)[::-1][:k]
                            sent_scores.append(
                                (
                                    token,
                                    [objective["vocab"][idx] for idx in topk],
                                    [token_scores[idx] for idx in topk]
                                )
                            )
                        all_sent_scores.append(sent_scores)
                    named_outputs[obj_name] = all_sent_scores
                else:
                    raise ValueError("unknown objective type %r." % (objective["type"],))
            yield named_outputs

    def tag_sentences(self, sentences):
        if len(sentences) == 0:
            return {
                objective["name"]: []
                for objective in self._model.objectives
            }
        blank_labels = tuple(None for _ in self._model.objectives)
        batches = list(iter_batches_single_threaded(
            self._model,
            [
                (sentence, [blank_labels for t in sentence])
                for sentence in sentences
            ],
            batch_size=min(256, len(sentences)),
            train=False,
            autoresize=False
        ))

        named_outputs = {}
        sentence_idx = 0

        for batch in batches:
            outputs = self._model.predict(self.session, batch)
            for objective in self._model.objectives:
                obj_name = objective["name"]
                if obj_name not in named_outputs:
                    named_outputs[obj_name] = []
                tags, scores = outputs[obj_name]
                nsentences = len(tags)
                if objective["type"] == "crf":
                    named_outputs[obj_name].extend([
                        [(token, objective["vocab"][tag], score) for token, tag in zip(tokens, tags)]
                        for tokens, tags, score in zip(sentences[sentence_idx:sentence_idx+nsentences], tags, scores)
                    ])
                elif objective["type"] == 'softmax':
                    named_outputs[obj_name].extend([
                        [(token, objective["vocab"][tag], score)
                         for token, tag, score in zip(tokens, tags, scores)]
                        for tokens, tags, scores in zip(sentences[sentence_idx:sentence_idx+nsentences], tags, scores)
                    ])
                else:
                    raise ValueError("unknown objective type %r." % (objective["type"],))
            sentence_idx += nsentences

        return named_outputs


def count_number_of_parameters():
    return int(sum([np.prod(var.get_shape().as_list())
                    for var in tf.trainable_variables()]))


class TestCallback(object):
    def __init__(self, model, session, dataset, epoch, args, logger):
        self.model = model
        self.session = session
        self.dataset = dataset
        self.epoch = epoch
        self.args = args
        self.logger = logger
        self.report_metrics_per_axis = args.report_metrics_per_axis
        self.report_class_f1 = args.report_class_f1

    def test(self, iteration):
        dev_outcome = accuracy(self.model, self.session, self.dataset, self.args.batch_size,
            train=False, report_metrics_per_axis=self.report_metrics_per_axis,
            report_class_f1=self.report_class_f1)
        print_outcome(dev_outcome, self.model.objectives,
            epoch="{}-{}".format(self.epoch, iteration),
            step=self.session.run(self.model.global_step),
            name="validation",
            logger=self.logger
        )
        if self.args.save_dir is not None:
            self.model.save(self.session, self.args.save_dir)


def compute_epoch(session, model, train_set,
                  validation_set, test_callback, epoch,
                  train_writer, test_writer,
                  args):
    test_callback.epoch = epoch
    train_outcome = accuracy(model,
                             session,
                             train_set,
                             args.batch_size,
                             train=True,
                             callback_period=args.test_every,
                             writer=train_writer.writer if train_writer is not None else None,
                             report_metrics_per_axis=args.report_metrics_per_axis,
                             report_class_f1=args.report_class_f1,
                             callback=test_callback.test)
    global_step = session.run(model.global_step)
    print_outcome(train_outcome,
                  model.objectives,
                  epoch=epoch,
                  name="train",
                  step=global_step,
                  logger=train_writer)
    dev_outcome = accuracy(
        model, session, validation_set, args.batch_size,
        train=False,
        report_metrics_per_axis=args.report_metrics_per_axis,
        report_class_f1=args.report_class_f1)
    print_outcome(dev_outcome,
                  model.objectives,
                  epoch=epoch,
                  step=global_step,
                  name="validation",
                  logger=test_writer)
    if args.save_dir is not None:
        model.save(session, args.save_dir)
    return dev_outcome


def main():
    args = parse_args()
    config = Config.load(args.config)
    validation_set = config.load_dataset("dev", merge=False)
    session_conf = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=session_conf) as session, tf.device(args.device):
        if args.load_dir is not None:
            model = SequenceModel.load(session, args.load_dir,
                args=args, rebuild_graph=args.rebuild_graph, faux_cudnn=args.faux_cudnn,
                replace_to=args.name,
                replace_from=args.old_name)
            dev_outcome = accuracy(
                model, session, validation_set, args.batch_size, train=False,
                report_metrics_per_axis=args.report_metrics_per_axis,
                report_class_f1=args.report_class_f1)
            print_outcome(dev_outcome,
                          model.objectives, 0,
                          name="loaded validation",
                          step=session.run(model.global_step),
                          logger=None)
            # dev_outcome = None
            if args.rebuild_graph and args.save_dir is not None:
                model.save(session, args.save_dir)
            train_set = config.load_dataset("train")
        else:
            # load classes and index2word from a file.
            dev_outcome = None
            train_set = config.load_dataset("train")
            model = SequenceModel(
                objectives=get_objectives(config.objectives, train_set),
                features=config.features,
                feature_index2words=get_feature_vocabs(config.features, train_set, ["<UNK>"]),
                lr=args.lr,
                anneal_rate=args.anneal_rate,
                weight_noise=args.weight_noise,
                freeze_rate=args.freeze_rate,
                freeze_rate_anneal=args.freeze_rate_anneal,
                clip_norm=args.clip_norm,
                hidden_sizes=args.hidden_sizes,
                solver=args.solver,
                fused=args.fused,
                class_weights_normalize=args.class_weights_normalize,
                class_weights=args.class_weights,
                class_weights_clipval=args.class_weights_clipval,
                keep_prob=args.keep_prob,
                input_keep_prob=args.input_keep_prob,
                name=args.name,
                cudnn=args.cudnn,
                faux_cudnn=args.faux_cudnn,
                create_variables=True)
            session.run(tf.global_variables_initializer())
            if args.restore_input_features is not None:
                restore_session(
                    session, args.restore_input_features,
                    verbose=True,
                    use_metagraph=False,
                    only_features=True)

        print("Model has {} trainable parameters.".format(count_number_of_parameters()), flush=True)
        best_dev_score = 0.0
        patience = 0
        best_epoch = 0
        best_outcome = None
        improvement_key = args.improvement_key
        if dev_outcome is not None:
            best_dev_score = dev_outcome[improvement_key]
            best_epoch = -1
            best_outcome = dev_outcome

        if args.save_dir is not None:
            train_writer = Logger(session, tf.summary.FileWriter(join(args.save_dir, "train")))
            test_writer = Logger(session, tf.summary.FileWriter(join(args.save_dir, "test")))
        else:
            train_writer, test_writer = None, None

        test_callback = TestCallback(model,
                                     session,
                                     validation_set,
                                     -1,
                                     args,
                                     logger=test_writer)
        if len(train_set) > 0:
            train_set.set_randomize(True)
            train_set.set_rng(model.rng)
            for epoch in range(args.max_epochs):
                dev_outcome = compute_epoch(
                    session, model,
                    train_set=train_set, validation_set=validation_set,
                    epoch=epoch, test_callback=test_callback,
                    train_writer=train_writer,
                    test_writer=test_writer,
                    args=args)

                if dev_outcome[improvement_key] > best_dev_score:
                    best_dev_score = dev_outcome[improvement_key]
                    best_epoch = epoch
                    best_outcome = dev_outcome
                    patience = 0
                    if args.save_dir is not None:
                        model.save(session, join(args.save_dir, "best"))
                else:
                    patience += 1
                    if patience >= args.max_patience:
                        print("No improvements for {} epochs. Stopping.".format(args.max_patience))
                        break
                del dev_outcome
        print_outcome(
            best_outcome,
            model.objectives,
            epoch=best_epoch,
            name="validation-best",
            step=session.run(model.global_step),
            logger=None)


if __name__ == "__main__":
    main()
