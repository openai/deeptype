"""
Obtain a learnability score for each type axis.
Trains a binary classifier for each type and
gets its AUC.

Usage
-----

```
python3 evaluate_learnability.py sample_data.tsv --out report.json --wikidata /path/to/wikidata
```

"""
import json
import time
import argparse

from os.path import dirname, realpath, join

SCRIPT_DIR = dirname(realpath(__file__))

import numpy as np
import tensorflow as tf

from sklearn import metrics
from collections import Counter

from wikidata_linker_utils.type_collection import TypeCollection, offset_values_mask
import wikidata_linker_utils.wikidata_properties as wprop
from wikidata_linker_utils.progressbar import get_progress_bar
from generator import prefetch_generator


def learnability(collection, lines, mask, truth_tables, qids, id2pos,
                 epochs=5, batch_size=128, max_dataset_size=-1,
                 max_vocab_size=10000, hidden_sizes=None, lr=0.001,
                 window_size=5, input_size=5, keep_prob=0.5,
                 verbose=True):
    if hidden_sizes is None:
        hidden_sizes = []
    tf.reset_default_graph()
    dset = list(get_windows(lines, mask, window_size, truth_tables, lambda x: id2pos[x]))
    if max_dataset_size > 0:
        dset = dset[:max_dataset_size]

    pos_num = np.zeros(len(qids))
    for _, labels in dset:
        pos_num += labels
    neg_num = np.ones(len(qids)) * len(dset) - pos_num
    pos_weight = (pos_num / (pos_num + neg_num))[None, :]

    vocab = ["<UNK>"] + [w for w, _ in Counter(lines[:, 0]).most_common(max_vocab_size)]
    inv_vocab = {w: k for k, w in enumerate(vocab)}
    with tf.device("gpu"):
        W = tf.get_variable(
            "W", shape=[len(vocab), input_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer()
        )
        indices = tf.placeholder(tf.int32, [None, window_size*2], name="indices")
        labels = tf.placeholder(tf.bool, [None, len(qids)], name="label")
        keep_prob_pholder = tf.placeholder_with_default(keep_prob, [])
        lookup = tf.reshape(tf.nn.embedding_lookup(
            W, indices
        ), [tf.shape(indices)[0], input_size * window_size*2])
        lookup = tf.nn.dropout(lookup, keep_prob_pholder)
        hidden = lookup
        for layer_idx, hidden_size in enumerate(hidden_sizes):
            hidden = tf.contrib.layers.fully_connected(
                hidden,
                num_outputs=hidden_size,
                scope="FC%d" % (layer_idx,)
            )
        out = tf.contrib.layers.fully_connected(
            hidden,
            num_outputs=len(qids),
            activation_fn=None)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=tf.cast(labels, tf.float32))
        cost = tf.where(tf.is_finite(cost), cost, tf.zeros_like(cost))
        cost_mean = tf.reduce_mean(
            (tf.cast(labels, tf.float32) * 1.0 / (pos_weight)) * cost +
            (tf.cast(tf.logical_not(labels), tf.float32) * 1.0 / (1.0 - pos_weight)) * cost
        )
        cost_sum = tf.reduce_sum(cost)
        size = tf.shape(indices)[0]
        noop = tf.no_op()
        correct = tf.reduce_sum(tf.cast(tf.equal(tf.greater_equal(out, 0), labels), tf.int32), 0)
        out_activated = tf.sigmoid(out)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_mean)
        session = tf.InteractiveSession()
        session.run(tf.global_variables_initializer())

    def accuracy(dataset, batch_size, train):
        epoch_correct = np.zeros(len(qids))
        epoch_nll = 0.0
        epoch_total = np.zeros(len(qids))
        op = train_op if train else noop
        all_labels = []
        all_preds = []
        for i in get_progress_bar("train" if train else "dev", item="batches")(range(0, len(dataset), batch_size)):
            batch_labels = [label for _, label in dataset[i:i+batch_size]]
            csum, corr, num_examples, preds, _ = session.run([cost_sum, correct, size, out_activated, op],
                feed_dict={
                    indices: [[inv_vocab.get(w, 0) for w in window] for window, _ in dataset[i:i+batch_size]],
                    labels: batch_labels,
                    keep_prob_pholder: keep_prob if train else 1.0
                })
            epoch_correct += corr
            epoch_nll += csum
            epoch_total += num_examples
            all_labels.extend(batch_labels)
            all_preds.append(preds)
        return (epoch_nll, epoch_correct, epoch_total, np.vstack(all_preds), np.vstack(all_labels))


    dataset_indices = np.arange(len(dset))
    train_indices = dataset_indices[:int(0.8 * len(dset))]
    dev_indices = dataset_indices[int(0.8 * len(dset)):]
    train_dataset = [dset[idx] for idx in train_indices]
    dev_dataset = [dset[idx] for idx in dev_indices]
    learnability = []
    for epoch in range(epochs):
        t0 = time.time()
        train_epoch_nll, train_epoch_correct, train_epoch_total, _, _ = accuracy(train_dataset, batch_size, train=True)
        t1 = time.time()
        if verbose:
            print("epoch %d train: %.3f%% in %.3fs" % (
                epoch, 100.0 * train_epoch_correct.sum() / train_epoch_total.sum(), t1 - t0),)
        t0 = time.time()
        dev_epoch_nll, dev_epoch_correct, dev_epoch_total, pred, y = accuracy(dev_dataset, batch_size, train=False)
        t1 = time.time()
        learnability = []
        for qidx in range(len(qids)):
            try:
                fpr, tpr, thresholds = metrics.roc_curve(y[:,qidx], pred[:,qidx], pos_label=1)
                auc = metrics.auc(fpr, tpr)
                if not np.isnan(auc):
                    average_precision_score = metrics.average_precision_score(y[:,qidx], pred[:,qidx])
                    learnability.append((qids[qidx],
                                         auc,
                                         average_precision_score,
                                         100.0 * dev_epoch_correct[qidx] / dev_epoch_total[qidx],
                                         int(pos_num[qidx]),
                                         int(neg_num[qidx])))
            except ValueError:
                continue
        if verbose:
            learnability = sorted(learnability, key=lambda x: x[1], reverse=True)
            print("epoch %d dev: %.3fs" % (epoch, t1-t0))
            for qid, auc, average_precision_score, acc, pos, neg in learnability:
                print("    %r AUC: %.3f, APS: %.3f, %.3f%% positive: %d, negative: %d" % (
                    collection.ids[qid], auc, average_precision_score, acc, pos, neg))
            print("")
    return learnability


def generate_training_data(collection, path):
    with open(path, "rt") as fin:
        lines = [row.split("\t")[:2] for row in fin.read().splitlines()]
    lines_arr = np.zeros((len(lines), 2), dtype=np.object)
    mask = np.zeros(len(lines), dtype=np.bool)
    for i, l in enumerate(lines):
        lines_arr[i, 0] = l[0]
        if len(l) > 1:
            lines_arr[i, 1] = collection.name2index[l[1]]
            mask[i] = True
    return lines_arr, mask


def get_proposal_sets(collection, article_ids, seed):
    np.random.seed(seed)
    relation = collection.relation(wprop.CATEGORY_LINK)

    relation_mask = offset_values_mask(relation.values, relation.offsets, article_ids)
    counts = np.bincount(relation.values[relation_mask])
    is_fp = collection.relation(wprop.FIXED_POINTS).edges() > 0
    is_fp = is_fp[:counts.shape[0]]
    counts = counts * is_fp
    topfields_fp = np.argsort(counts)[::-1][:(counts > 0).sum()]
    relation = collection.relation(wprop.INSTANCE_OF)

    relation_mask = offset_values_mask(relation.values, relation.offsets, article_ids)
    counts = np.bincount(relation.values[relation_mask])
    topfields_instance_of = np.argsort(counts)[::-1][:(counts > 0).sum()]

    np.random.shuffle(topfields_instance_of)
    np.random.shuffle(topfields_fp)

    return [(topfields_instance_of, wprop.INSTANCE_OF), (topfields_fp, wprop.CATEGORY_LINK)]  


def build_truth_tables(collection, lines, qids, relation_name):
    truth_tables = []
    all_ids = list(sorted(set(lines[:, 1])))
    id2pos = {idx: pos for pos, idx in enumerate(all_ids)}
    for qid in qids:
        truth_tables.append(collection.satisfy([relation_name], [qid])[all_ids])
        collection.reset_cache()
    truth_tables = np.stack(truth_tables, axis=1)
    qid_sums = truth_tables.sum(axis=0)
    kept_qids = []
    kept_dims = []
    for i, (qid, qid_sum) in enumerate(zip(qids, qid_sums)):
        if qid_sum != 0 and qid_sum != truth_tables.shape[0]:
            kept_qids.append(qid)
            kept_dims.append(i)
    truth_tables = truth_tables[:, kept_dims]
    return truth_tables, kept_qids, id2pos


def get_windows(lines, mask, window, truth_table, id_mapper):
    for i in np.where(mask)[0]:
        if i >= window and i < len(lines) - window:
            yield (lines[max(0, i - window):i + window, 0],
                   truth_table[id_mapper(lines[i, 1])])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=10000)
    parser.add_argument("--simultaneous_fields", type=int, default=512)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=5)
    parser.add_argument("--wikidata", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def generate_truth_tables(collection, lines_arr, proposal_sets, simultaneous_fields):
    for topfields, relation_name in proposal_sets:
        for i in range(0, len(topfields), simultaneous_fields):
            truth_tables, qids, id2pos = build_truth_tables(
                collection,
                lines_arr,
                qids=topfields[i:i+simultaneous_fields],
                relation_name=relation_name)
            yield (topfields[i:i+simultaneous_fields],
                   relation_name,
                   truth_tables,
                   qids,
                   id2pos)


def main():
    args = parse_args()
    collection = TypeCollection(args.wikidata, num_names_to_load=0)
    collection.load_blacklist(join(dirname(SCRIPT_DIR), "extraction", "blacklist.json"))
    lines_arr, mask = generate_training_data(collection, args.dataset)
    article_ids = np.array(list(set(lines_arr[:, 1])), dtype=np.int32)
    proposal_sets = get_proposal_sets(collection, article_ids, args.seed)
    report = []
    total = sum(len(topfields) for topfields, _ in proposal_sets)
    seen = 0
    t0 = time.time()
    data_source = generate_truth_tables(collection, lines_arr, proposal_sets,
                                        args.simultaneous_fields)

    for topfields, relation_name, truth_tables, qids, id2pos in prefetch_generator(data_source):
        # for each of these properties and given relation
        # construct the truth table for each item and discover
        # their 'learnability':
        seen += len(topfields)
        field_auc_scores = learnability(
            collection,
            lines_arr,
            mask,
            qids=qids,
            truth_tables=truth_tables,
            id2pos=id2pos,
            batch_size=args.batch_size,
            epochs=args.max_epochs,
            input_size=args.input_size,
            window_size=args.window_size,
            max_vocab_size=args.max_vocab_size,
            verbose=True)
        for qid, auc, average_precision_score, correct, pos, neg in field_auc_scores:
            report.append(
                {
                    "qid": collection.ids[qid],
                    "auc": auc,
                    "average_precision_score": average_precision_score,
                    "correct": correct,
                    "relation": relation_name,
                    "positive": pos,
                    "negative": neg
                }
            )
        with open(args.out, "wt") as fout:
            json.dump(report, fout)
        t1 = time.time()
        speed = seen / (t1 - t0)
        print("AUC obtained for %d / %d items (%.3f items/s)" % (seen, total, speed), flush=True)


if __name__ == "__main__":
    main()

