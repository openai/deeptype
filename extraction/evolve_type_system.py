import json
import argparse
import time
import random
import numpy as np

from evaluate_type_system import fix_and_parse_tags

from wikidata_linker_utils.json import load_config
from wikidata_linker_utils.type_collection import TypeCollection
from wikidata_linker_utils.progressbar import get_progress_bar
from wikidata_linker_utils.wikipedia import induce_wikipedia_prefix
from os.path import realpath, dirname, join, exists
from wikidata_linker_utils.fast_disambiguate import (
    beam_project, cem_project, ga_project
)

SCRIPT_DIR = dirname(realpath(__file__))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("--relative_to", default=None, type=str)
    parser.add_argument("--penalty", default=0.0005, type=float)
    parser.add_argument("--beam_width", default=8, type=float)
    parser.add_argument("--beam_search_subset", default=2000, type=int)
    parser.add_argument("--log", default=None, type=str)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--ngen", type=int, default=40)
    parser.add_argument("--method", type=str,
        choices=["cem", "greedy", "beam", "ga"],
        default="greedy")
    return parser.parse_args(args=args)


def load_aucs():
    paths = [
        "/home/jonathanraiman/en_field_auc_w10_e10.json",
        "/home/jonathanraiman/en_field_auc_w10_e10-s1234.json",
        "/home/jonathanraiman/en_field_auc_w5_e5.json",
        "/home/jonathanraiman/en_field_auc_w5_e5-s1234.json"
    ]
    aucs = {}
    for path in paths:
        with open(path, "rt") as fin:
            auc_report = json.load(fin)
            for report in auc_report:
                key = (report["qid"], report["relation"])
                if key in aucs:
                    aucs[key].append(report["auc"])
                else:
                    aucs[key] = [report["auc"]]
    for key in aucs.keys():
        aucs[key] = np.mean(aucs[key])
    return aucs

def greedy_disambiguate(tags):
    greedy_correct = 0
    total = 0
    for dest, other_dest, times_pointed in tags:
        total += 1
        if len(other_dest) == 1 and dest == other_dest[0]:
            greedy_correct += 1
        elif other_dest[np.argmax(times_pointed)] == dest:
            greedy_correct += 1
    return greedy_correct, total


def fast_disambiguate(tags, all_classifications):
    correct = 0
    total = 0
    for dest, other_dest, times_pointed in tags:
        total += 1
        if len(other_dest) == 1 and dest == other_dest[0]:
            correct += 1
        else:
            identities = np.all(all_classifications[other_dest, :] == all_classifications[dest, :], axis=1)
            matches = other_dest[identities]
            matches_counts = times_pointed[identities]
            if len(matches) == 1 and matches[0] == dest:
                correct += 1
            elif matches[np.argmax(matches_counts)] == dest:
                correct += 1
    return correct, total


def get_prefix(config):
    return config.prefix or induce_wikipedia_prefix(config.wiki)


MAX_PICKS = 400.0

def rollout(cached_satisfy, key2row, tags, aucs, ids, sample,
            penalty, greedy_correct):
    mean_auc = 0.0
    sample_sum = sample.sum()
    if sample_sum == 0:
        total = len(tags)
        return (greedy_correct / total,
                greedy_correct / total)
    if sample_sum > MAX_PICKS:
        return 0.0, 0.0
    all_classifications = None
    if sample_sum > 0:
        all_classifications = np.zeros((len(ids), int(sample_sum)), dtype=np.bool)
        col = 0
        for picked, (key, auc) in zip(sample, aucs):
            if picked:
                all_classifications[:, col] = cached_satisfy[key2row[key]]
                col += 1
                mean_auc += auc
        mean_auc = mean_auc / sample_sum
    correct, total = fast_disambiguate(tags, all_classifications)
    # here's the benefit of using types:
    improvement = correct - greedy_correct
    # penalty for using unreliable types:
    objective = (
        (greedy_correct + improvement * mean_auc) / total -
        # number of items is penalized
        sample_sum * penalty
    )
    return objective, correct / total


def get_cached_satisfy(collection, aucs, ids, mmap=False):
    path = join(SCRIPT_DIR, "cached_satisfy.npy")
    if not exists(path):
        cached_satisfy = np.zeros((len(aucs), len(ids)), dtype=np.bool)
        for row, (qid, relation_name) in get_progress_bar("satisfy", item="types")(enumerate(sorted(aucs.keys()))):
            cached_satisfy[row, :] = collection.satisfy([relation_name], [collection.name2index[qid]])[ids]
            collection._satisfy_cache.clear()
        np.save(path, cached_satisfy)
        if mmap:
            del cached_satisfy
            cached_satisfy = np.load(path, mmap_mode="r")
    else:
        if mmap:
            cached_satisfy = np.load(path, mmap_mode="r")
        else:
            cached_satisfy = np.load(path)
    return cached_satisfy


def main():
    args = parse_args()
    config = load_config(
        args.config,
        ["wiki",
         "language_path",
         "wikidata",
         "redirections",
         "classification"],
        defaults={
            "num_names_to_load": 0,
            "prefix": None,
            "sample_size": 100,
            "wiki": None,
            "fix_links": False,
            "min_count": 0,
            "min_percent": 0.0
        },
        relative_to=args.relative_to
    )
    if config.wiki is None:
        raise ValueError("must provide path to 'wiki' in config.")
    prefix = get_prefix(config)
    collection = TypeCollection(
        config.wikidata,
        num_names_to_load=config.num_names_to_load,
        prefix=prefix,
        verbose=True
    )
    collection.load_blacklist(join(SCRIPT_DIR, "blacklist.json"))

    fname = config.wiki
    test_tags = fix_and_parse_tags(config,
                                   collection,
                                   config.sample_size)
    aucs = load_aucs()
    ids = sorted(set([idx for doc_tags in test_tags
                      for _, tag in doc_tags if tag is not None
                      for idx in tag[2] if len(tag[2]) > 1]))
    id2pos = {idx: k for k, idx in enumerate(ids)}
    # use reduced identity system:
    remapped_tags = []
    for doc_tags in test_tags:
        for text, tag in doc_tags:
            if tag is not None:
                remapped_tags.append(
                    (id2pos[tag[1]] if len(tag[2]) > 1 else tag[1],
                     np.array([id2pos[idx] for idx in tag[2]]) if len(tag[2]) > 1 else tag[2],
                     tag[3]))
    test_tags = remapped_tags

    aucs = {key: value for key, value in aucs.items() if value > 0.5}
    print("%d relations to pick from with %d ids." % (len(aucs), len(ids)), flush=True)
    cached_satisfy = get_cached_satisfy(collection, aucs, ids, mmap=args.method=="greedy")
    del collection
    key2row = {key: k for k, key in enumerate(sorted(aucs.keys()))}

    if args.method == "greedy":
        picks, _ = beam_project(
            cached_satisfy,
            key2row,
            remapped_tags,
            aucs,
            ids,
            beam_width=1,
            penalty=args.penalty,
            log=args.log
        )
    elif args.method == "beam":
        picks, _ = beam_project(
            cached_satisfy,
            key2row,
            remapped_tags,
            aucs,
            ids,
            beam_width=args.beam_width,
            penalty=args.penalty,
            log=args.log
        )
    elif args.method == "cem":
        picks, _ = cem_project(
            cached_satisfy,
            key2row,
            remapped_tags,
            aucs,
            ids,
            n_samples=args.samples,
            penalty=args.penalty,
            log=args.log
        )
    elif args.method == "ga":
        picks, _ = ga_project(
            cached_satisfy,
            key2row,
            remapped_tags,
            aucs,
            ids,
            ngen=args.ngen,
            n_samples=args.samples,
            penalty=args.penalty,
            log=args.log
        )
    else:
        raise ValueError("unknown method %r." % (args.method,))
    with open(args.out, "wt") as fout:
        json.dump(picks, fout)


if __name__ == "__main__":
    main()
