import argparse
import sys
import json
import time
import traceback

from os import makedirs
from os.path import join, dirname, realpath
from wikidata_linker_utils.repl import (
    enter_or_quit, reload_module,
    ALLOWED_RUNTIME_ERRORS,
    ALLOWED_IMPORT_ERRORS
)
from wikidata_linker_utils.logic import logical_ors
from wikidata_linker_utils.type_collection import TypeCollection
import wikidata_linker_utils.wikidata_properties as wprop


import numpy as np

SCRIPT_DIR = dirname(realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('wikidata', type=str,
        help="Location of wikidata properties.")
    parser.add_argument('classifiers', type=str, nargs="+",
        help="Filename(s) for Python script that classifies entities.")
    parser.add_argument('--export_classification', type=str, nargs="+",
        default=None,
        help="Location to save the result of the entity classification.")
    parser.add_argument('--num_names_to_load', type=int, default=20000000,
        help="Number of names to load from disk to accelerate reporting.")
    parser.add_argument('--language_path', type=str, default=None,
        help="Location of a language-wikipedia specific information set to "
             "provide language/wikipedia specific metrics.")
    parser.add_argument('--interactive', action="store_true", default=True,
        help="Operate in a REPL. Reload scripts on errors or on user prompt.")
    parser.add_argument('--nointeractive', action="store_false",
        dest="interactive", help="Run classification without REPL.")
    parser.add_argument('--use-cache', action="store_true",
        dest="use_cache", help="store satisfies in cache.")
    parser.add_argument('--nouse-cache', action="store_false",
        dest="use_cache", help="not store satisfies in cache.")
    return parser.parse_args()


def get_other_class(classification):
    if len(classification) == 0:
        return None
    return np.logical_not(logical_ors(
        list(classification.values())
    ))


def export_classification(classification, path):
    classes = sorted(list(classification.keys()))
    if len(classes) == 0:
        return
    makedirs(path, exist_ok=True)
    num_items = classification[classes[0]].shape[0]
    classid = np.zeros(num_items, dtype=np.int32)
    selected = np.zeros(num_items, dtype=np.bool)
    for index, classname in enumerate(classes):
        truth_table = classification[classname]
        selected = selected | truth_table
        classid = np.maximum(classid, truth_table.astype(np.int32) * index)

    other = np.logical_not(selected)
    if other.sum() > 0:
        classes_with_other = classes + ["other"]
        classid = np.maximum(classid, other.astype(np.int32) * len(classes))
    else:
        classes_with_other = classes

    with open(join(path, "classes.txt"), "wt") as fout:
        for classname in classes_with_other:
            fout.write(classname + "\n")

    np.save(join(path, "classification.npy"), classid)


def main():
    args = parse_args()
    should_export = args.export_classification is not None
    if should_export and len(args.export_classification) != len(args.classifiers):
        raise ValueError("Must have as many export filenames as classifiers.")
    collection = TypeCollection(
        args.wikidata,
        num_names_to_load=args.num_names_to_load,
        language_path=args.language_path,
        cache=args.use_cache
    )
    if args.interactive:
        alert_failure = enter_or_quit
    else:
        alert_failure = lambda: sys.exit(1)

    while True:
        try:
            collection.load_blacklist(join(SCRIPT_DIR, "blacklist.json"))
        except (ValueError,) as e:
            print("Issue reading blacklist, please fix.")
            print(str(e))
            alert_failure()
            continue

        classifications = []
        for class_idx, classifier_fname in enumerate(args.classifiers):
            while True:
                try:
                    classifier = reload_module(classifier_fname)
                except ALLOWED_IMPORT_ERRORS as e:
                    print("issue reading %r, please fix." % (classifier_fname,))
                    print(str(e))
                    traceback.print_exc(file=sys.stdout)
                    alert_failure()
                    continue

                try:
                    t0 = time.time()
                    classification = classifier.classify(collection)
                    classifications.append(classification)
                    if class_idx == len(args.classifiers) - 1:
                        collection.reset_cache()
                    t1 = time.time()
                    print("classification took %.3fs" % (t1 - t0,))
                except ALLOWED_RUNTIME_ERRORS as e:
                    print("issue running %r, please fix." % (classifier_fname,))
                    print(str(e))
                    traceback.print_exc(file=sys.stdout)
                    alert_failure()
                    continue
                break
        try:
            # show cardinality for each truth table:
            if args.interactive:
                mega_other_class = None
                for classification in classifications:
                    for classname in sorted(classification.keys()):
                        print("%r: %d members" % (classname, int(classification[classname].sum())))
                    print("")
                    summary = {}
                    for classname, truth_table in classification.items():
                        (members,) = np.where(truth_table)
                        summary[classname] = [collection.get_name(int(member)) for member in members[:20]]
                    print(json.dumps(summary, indent=4))

                    other_class = get_other_class(classification)
                    if other_class.sum() > 0:
                        # there are missing items:
                        to_report = (
                            classifier.class_report if hasattr(classifier, "class_report") else
                            [wprop.SUBCLASS_OF, wprop.INSTANCE_OF, wprop.OCCUPATION, wprop.CATEGORY_LINK]
                        )
                        collection.class_report(to_report, other_class, name="Other")
                        if mega_other_class is None:
                            mega_other_class = other_class
                        else:
                            mega_other_class = np.logical_and(mega_other_class, other_class)
                if len(classifications) > 1:
                    if mega_other_class.sum() > 0:
                        # there are missing items:
                        to_report = [wprop.SUBCLASS_OF, wprop.INSTANCE_OF, wprop.OCCUPATION, wprop.CATEGORY_LINK]
                        collection.class_report(to_report, mega_other_class, name="Other-combined")
            if should_export:
                assert(len(classifications) == len(args.export_classification)), (
                    "classification outputs missing for export."
                )
                for classification, savename in zip(classifications, args.export_classification):
                    export_classification(classification, savename)
        except KeyboardInterrupt as e:
            pass

        if args.interactive:
            enter_or_quit()
        else:
            break


if __name__ == "__main__":
    main()


