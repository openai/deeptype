import sys
import pickle
import argparse
import requests
import marisa_trie
import traceback
import numpy as np

from os.path import join, dirname, realpath, exists
from os import stat
from collections import Counter
from itertools import product

from wikidata_linker_utils.anchor_filtering import clean_up_trie_source, acceptable_anchor
from wikidata_linker_utils.wikipedia import (
    load_wikipedia_docs, induce_wikipedia_prefix, load_redirections, transition_trie_index
)
from wikidata_linker_utils.json import load_config

from wikidata_linker_utils.offset_array import OffsetArray
from wikidata_linker_utils.repl import reload_run_retry, enter_or_quit
from wikidata_linker_utils.progressbar import get_progress_bar
from wikidata_linker_utils.type_collection import TypeCollection, get_name as web_get_name


SCRIPT_DIR = dirname(realpath(__file__))
PROJECT_DIR = dirname(SCRIPT_DIR)

INTERNET = True

def maybe_web_get_name(s):
    global INTERNET
    if INTERNET:
        try:
            res = web_get_name(s)
            return res
        except requests.exceptions.ConnectionError:
            INTERNET = False
    return s


class OracleClassification(object):
    def __init__(self, classes, classification, path):
        self.classes = classes
        self.classification = classification
        self.path = path
        self.contains_other = self.classes[-1] == "other"

    def classify(self, index):
        return self.classification[index]

def load_oracle_classification(path):
    with open(join(path, "classes.txt"), "rt") as fin:
        classes = fin.read().splitlines()
    classification = np.load(join(path, "classification.npy"))
    return OracleClassification(classes, classification, path)


def can_disambiguate(oracles, truth, alternatives,
                     times_pointed, count_threshold,
                     ignore_other=False, keep_other=False):
    ambig = np.ones(len(alternatives), dtype=np.bool)
    for oracle in oracles:
        truth_pred = oracle.classify(truth)
        alt_preds = oracle.classify(alternatives)
        if keep_other and oracle.contains_other:
            if truth_pred == len(oracle.classes) - 1:
                continue
            else:
                ambig = np.logical_and(
                    ambig,
                    np.logical_or(
                        np.equal(alt_preds, truth_pred),
                        np.equal(alt_preds, len(oracle.classes) - 1)
                    )
                )
        elif ignore_other and oracle.contains_other and np.any(alt_preds == len(oracle.classes) - 1):
            continue
        else:
            ambig = np.logical_and(ambig, np.equal(alt_preds, truth_pred))

    # apply type rules to disambiguate:
    alternatives_matching_type = alternatives[ambig]
    alternatives_matching_type_times_pointed = times_pointed[ambig]

    if len(alternatives_matching_type) <= 1:
        return alternatives_matching_type, alternatives_matching_type_times_pointed, False

    # apply rules for count thresholding:
    ordered_times_pointed = np.argsort(alternatives_matching_type_times_pointed)[::-1]
    top1count = alternatives_matching_type_times_pointed[ordered_times_pointed[0]]
    top2count = alternatives_matching_type_times_pointed[ordered_times_pointed[1]]
    if top1count > top2count + count_threshold and alternatives_matching_type[ordered_times_pointed[0]] == truth:
        return (
            alternatives_matching_type[ordered_times_pointed[0]:ordered_times_pointed[0]+1],
            alternatives_matching_type_times_pointed[ordered_times_pointed[0]:ordered_times_pointed[0]+1],
            True
        )
    return alternatives_matching_type, alternatives_matching_type_times_pointed, False


def disambiguate(tags, oracles):
    ambiguous = 0
    obvious = 0
    disambiguated_oracle = 0
    disambiguated_with_counts = 0
    disambiguated_greedy = 0
    disambiguated_with_background = 0
    count_threshold = 0
    ambiguous_tags = []
    obvious_tags = []
    non_obvious_tags = []

    disambiguated_oracle_ignore_other = 0
    disambiguated_oracle_keep_other = 0

    for text, tag in tags:
        if tag is None:
            continue
        anchor, dest, other_dest, times_pointed = tag
        if len(other_dest) == 1:
            obvious += 1
            obvious_tags.append((anchor, dest, other_dest, times_pointed))
        else:
            ambiguous += 1
            non_obvious_tags.append((anchor, dest, other_dest, times_pointed))

            if other_dest[np.argmax(times_pointed)] == dest:
                disambiguated_greedy += 1

            matching_tags, times_pointed_subset, used_counts = can_disambiguate(
                oracles, dest, other_dest, times_pointed, count_threshold
            )
            if len(matching_tags) <= 1:
                if used_counts:
                    disambiguated_with_counts += 1
                else:
                    disambiguated_oracle += 1
            else:
                ambiguous_tags.append(
                    (anchor, dest, matching_tags, times_pointed_subset)
                )

            matching_tags, times_pointed_subset, used_counts = can_disambiguate(
                oracles, dest, other_dest, times_pointed, count_threshold, ignore_other=True
            )
            if len(matching_tags) <= 1:
                disambiguated_oracle_ignore_other += 1

            matching_tags, times_pointed_subset, used_counts = can_disambiguate(
                oracles, dest, other_dest, times_pointed, count_threshold, keep_other=True
            )
            if len(matching_tags) <= 1:
                disambiguated_oracle_keep_other += 1

    report = {
        "ambiguous": ambiguous,
        "obvious": obvious,
        "disambiguated oracle": disambiguated_oracle,
        "disambiguated greedy": disambiguated_greedy,
        "disambiguated oracle + counts": disambiguated_oracle + disambiguated_with_counts,
        "disambiguated oracle + counts + ignore other": disambiguated_oracle_ignore_other,
        "disambiguated oracle + counts + keep other": disambiguated_oracle_keep_other
    }
    return (report, ambiguous_tags)


def disambiguate_batch(test_tags, train_tags, oracles):
    test_tags = test_tags
    total_report = {}
    ambiguous_tags = []
    for tags in get_progress_bar("disambiguating", item="articles")(test_tags):
        report, remainder = disambiguate(tags, oracles)
        ambiguous_tags.extend(remainder)
        for key, value in report.items():
            if key not in total_report:
                total_report[key] = value
            else:
                total_report[key] += value
    return total_report, ambiguous_tags


def obtain_tags(doc,
                wiki_trie,
                anchor_trie,
                trie_index2indices,
                trie_index2indices_counts,
                trie_index2indices_transitions,
                redirections,
                prefix,
                collection,
                first_names,
                min_count,
                min_percent):
    out_doc = []
    for anchor, dest_index in doc.links(wiki_trie, redirections, prefix):
        if dest_index is None:
            out_doc.append((anchor, None))
            continue
        anchor_stripped = anchor.strip()
        keep = False
        if len(anchor_stripped) > 0:
            anchor_stripped = clean_up_trie_source(anchor_stripped)
            if acceptable_anchor(anchor_stripped, anchor_trie, first_names):
                anchor_idx = anchor_trie[anchor_stripped]
                all_options = trie_index2indices[anchor_idx]
                all_counts = trie_index2indices_counts[anchor_idx]
                if len(all_options) > 0:
                    if trie_index2indices_transitions is not None:
                        old_dest_index = dest_index
                        dest_index = transition_trie_index(
                            anchor_idx, dest_index,
                            trie_index2indices_transitions,
                            all_options
                        )
                    if dest_index != -1:
                        new_dest_index = dest_index
                        keep = True
                        if keep and (min_count > 0 or min_percent > 0):
                            dest_count = all_counts[all_options==new_dest_index]
                            if dest_count < min_count or (dest_count / sum(all_counts)) < min_percent:
                                keep = False

                        if keep:
                            out_doc.append(
                                (
                                    anchor,
                                    (anchor_stripped, new_dest_index, all_options, all_counts)
                                )
                            )
        if not keep:
            out_doc.append((anchor, None))
    return out_doc


def add_boolean(parser, name, default):
    parser.add_argument("--%s" % (name,), action="store_true", default=default)
    parser.add_argument("--no%s" % (name,), action="store_false", dest=name)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--relative_to", type=str, default=None)
    parser.add_argument("--log", type=str, default=None)
    add_boolean(parser, "verbose", True)
    add_boolean(parser, "interactive", True)
    return parser


def parse_args(args=None):
    return get_parser().parse_args(args=args)


def summarize_disambiguation(total_report, file=None):
    if file is None:
        file = sys.stdout
    if total_report.get("ambiguous", 0) > 0:
        for key, value in sorted(total_report.items(), key=lambda x : x[1]):
            if "disambiguated" in key:
                print("%.3f%% disambiguated by %s (%d / %d)" % (
                        100.0 * value / total_report["ambiguous"],
                        key[len("disambiguated"):].strip(),
                        value, total_report["ambiguous"]
                    ), file=file
                )
        print("", file=file)
    for key, value in sorted(total_report.items(), key=lambda x : x[1]):
        if "disambiguated" in key:
            print("%.3f%% disambiguated by %s [including single choice] (%d / %d)" % (
                    100.0 * (
                        (value + total_report["obvious"]) /
                        (total_report["ambiguous"] + total_report["obvious"])
                    ),
                    key[len("disambiguated"):].strip(),
                    value + total_report["obvious"],
                    total_report["ambiguous"] + total_report["obvious"]
                ), file=file
            )
    print("", file=file)


def summarize_ambiguities(ambiguous_tags,
                          oracles,
                          get_name):
    class_ambiguities = {}
    for anchor, dest, other_dest, times_pointed in ambiguous_tags:
        class_ambig_name = []
        for oracle in oracles:
            class_ambig_name.append(oracle.classes[oracle.classify(dest)])
        class_ambig_name = " and ".join(class_ambig_name)
        if class_ambig_name not in class_ambiguities:
            class_ambiguities[class_ambig_name] = {
                "count": 1,
                "examples": [(anchor, dest, other_dest, times_pointed)]
            }
        else:
            class_ambiguities[class_ambig_name]["count"] += 1
            class_ambiguities[class_ambig_name]["examples"].append((anchor, dest, other_dest, times_pointed))
    print("Ambiguity Report:")
    for classname, ambiguity in sorted(class_ambiguities.items(), key=lambda x: x[0]):
        print("    %s" % (classname,))
        print("        %d ambiguities" % (ambiguity["count"],))

        common_bad_anchors = Counter([anc for anc, _, _, _ in ambiguity["examples"]]).most_common(6)
        anchor2example = {anc: (dest, other_dest, times_pointed) for anc, dest, other_dest, times_pointed in ambiguity["examples"]}

        for bad_anchor, count in common_bad_anchors:
            dest, other_dest, times_pointed = anchor2example[bad_anchor]
            truth_times_pointed = int(times_pointed[np.equal(other_dest, dest)])
            only_alt = [(el, int(times_pointed[k])) for k, el in enumerate(other_dest) if el != dest]
            only_alt = sorted(only_alt, key=lambda x: x[1], reverse=True)
            print("        %r (%d time%s)" % (bad_anchor, count, 's' if count != 1 else ''))
            print("            Actual: %r" % ((get_name(dest), truth_times_pointed),))
            print("            Others: %r" % ([(get_name(el), c) for (el, c) in only_alt[:5]]))
            print("")
        print("")


def get_prefix(config):
    return config.prefix or induce_wikipedia_prefix(config.wiki)



def fix_and_parse_tags(config, collection, size):
    trie_index2indices = OffsetArray.load(
        join(config.language_path, "trie_index2indices"),
        compress=True
    )
    trie_index2indices_counts = OffsetArray(
        np.load(join(config.language_path, "trie_index2indices_counts.npy")),
        trie_index2indices.offsets
    )
    if exists(join(config.language_path, "trie_index2indices_transition_values.npy")):
        trie_index2indices_transitions = OffsetArray(
            np.load(join(config.language_path, "trie_index2indices_transition_values.npy")),
            np.load(join(config.language_path, "trie_index2indices_transition_offsets.npy")),
        )
    else:
        trie_index2indices_transitions = None


    anchor_trie = marisa_trie.Trie().load(join(config.language_path, "trie.marisa"))
    wiki_trie = marisa_trie.RecordTrie('i').load(
        join(config.wikidata, "wikititle2wikidata.marisa")
    )
    prefix = get_prefix(config)
    redirections = load_redirections(config.redirections)
    docs = load_wikipedia_docs(config.wiki, size)

    while True:
        try:
            collection.load_blacklist(join(SCRIPT_DIR, "blacklist.json"))
        except (ValueError,) as e:
            print("issue reading blacklist, please fix.")
            print(str(e))
            enter_or_quit()
            continue
        break

    print("Load first_names")
    with open(join(PROJECT_DIR, "data", "first_names.txt"), "rt") as fin:
        first_names = set(fin.read().splitlines())

    all_tags = []
    for doc in get_progress_bar('fixing links', item='article')(docs):
        tags = obtain_tags(
            doc,
            wiki_trie=wiki_trie,
            anchor_trie=anchor_trie,
            trie_index2indices=trie_index2indices,
            trie_index2indices_counts=trie_index2indices_counts,
            trie_index2indices_transitions=trie_index2indices_transitions,
            redirections=redirections,
            prefix=prefix,
            first_names=first_names,
            collection=collection,
            min_count=config.min_count,
            min_percent=config.min_percent)
        if any(x is not None for _, x in tags):
            all_tags.append(tags)
    collection.reset_cache()
    return all_tags


def main():
    args = parse_args()
    config = load_config(args.config,
                         ["wiki",
                          "language_path",
                          "wikidata",
                          "redirections",
                          "classification",
                          "path"],
                         defaults={"num_names_to_load": 0,
                                   "prefix": None,
                                   "sample_size": 100,
                                   "wiki": None,
                                   "min_count": 0,
                                   "min_percent": 0.0},
                         relative_to=args.relative_to)
    if config.wiki is None:
        raise ValueError("must provide path to 'wiki' in config.")
    prefix = get_prefix(config)

    print("Load type_collection")
    collection = TypeCollection(
        config.wikidata,
        num_names_to_load=config.num_names_to_load,
        prefix=prefix,
        verbose=True)

    fname = config.wiki
    all_tags = fix_and_parse_tags(config, collection, config.sample_size)
    test_tags = all_tags[:config.sample_size]
    train_tags = all_tags[config.sample_size:]

    oracles = [load_oracle_classification(classification)
               for classification in config.classification]

    def get_name(idx):
        if idx < config.num_names_to_load:
            if idx in collection.known_names:
                return collection.known_names[idx] + " (%s)" % (collection.ids[idx],)
            else:
                return collection.ids[idx]
        else:
            return maybe_web_get_name(collection.ids[idx]) + " (%s)" % (collection.ids[idx],)

    while True:
        total_report, ambiguous_tags = disambiguate_batch(
            test_tags, train_tags, oracles)
        summarize_disambiguation(total_report)
        if args.log is not None:
            with open(args.log, "at") as fout:
                summarize_disambiguation(total_report, file=fout)
        if args.verbose:
            try:
                summarize_ambiguities(
                    ambiguous_tags,
                    oracles,
                    get_name
                )
            except KeyboardInterrupt as e:
                pass
        if args.interactive:
            enter_or_quit()
        else:
            break


if __name__ == "__main__":
    main()
